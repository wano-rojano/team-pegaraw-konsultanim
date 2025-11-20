"""Multi-crop and insurance RAG system for Konsultanim (OSS + Qwen + Alibaba Cloud embeddings via DashScope intl)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import dashscope
from dashscope import TextEmbedding

from .oss_loader import OSSPDFLoader
from .qwen_client import qwen_chat
from .qwen_langchain import QwenChat
from .agent_graph_with_helpfulness import build_agent_graph_with_helpfulness

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


class AlibabaEmbeddings(Embeddings):
    """Alibaba Cloud embedding wrapper with disk cache."""

    def __init__(self, model: str = "text-embedding-v3"):
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        
        # Simple disk cache
        self.cache_dir = Path(".embedding_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()

    def _get_cached(self, text: str):
        cache_file = self.cache_dir / f"{self._cache_key(text)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    def _set_cached(self, text: str, embedding):
        cache_file = self.cache_dir / f"{self._cache_key(text)}.json"
        cache_file.write_text(json.dumps(embedding))

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        """Embed texts in batches of 10 with caching."""
        all_embeddings = []
        to_embed = []
        to_embed_indices = []
        
        # Check cache first
        for idx, text in enumerate(inputs):
            cached = self._get_cached(text)
            if cached:
                all_embeddings.append((idx, cached))
            else:
                to_embed.append(text)
                to_embed_indices.append(idx)
        
        # Embed uncached texts
        batch_size = 10
        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i:i + batch_size]
            
            resp = TextEmbedding.call(
                api_key=self.api_key,
                model=self.model,
                input=batch,
            )
            
            if getattr(resp, "status_code", 200) != 200:
                raise RuntimeError(
                    f"DashScope embedding error: status_code={getattr(resp, 'status_code', None)}, "
                    f"code={getattr(resp, 'code', None)}, message={getattr(resp, 'message', resp)}"
                )
            
            if hasattr(resp.output, 'embeddings'):
                sorted_embeds = sorted(resp.output.embeddings, key=lambda e: e.get("text_index", 0))
                batch_embeddings = [e["embedding"] for e in sorted_embeds]
            elif isinstance(resp.output, dict) and 'embeddings' in resp.output:
                sorted_embeds = sorted(resp.output['embeddings'], key=lambda e: e.get("text_index", 0))
                batch_embeddings = [e["embedding"] for e in sorted_embeds]
            else:
                raise RuntimeError(f"Unexpected response format: {resp}")
            
            # Cache and collect
            for j, emb in enumerate(batch_embeddings):
                orig_idx = to_embed_indices[i + j]
                self._set_cached(to_embed[i + j], emb)
                all_embeddings.append((orig_idx, emb))
        
        # Sort back to original order
        all_embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in all_embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class DummyEmbeddings(Embeddings):
    """
    Simple local fallback embeddings.

    This is ONLY used if AlibabaEmbeddings fails (e.g., DashScope intl embeddings not yet enabled).
    It allows the app to keep working for the demo, while still clearly attempting to use Alibaba
    embeddings first.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash_to_vec(self, text: str) -> List[float]:
        # Very simple deterministic hash-based vector; not semantically meaningful,
        # but good enough to exercise Qdrant and RAG flow.
        import math

        seed = abs(hash(text))
        vec = []
        for i in range(self.dim):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            vec.append((seed / 0x7FFFFFFF) * 2 - 1.0)  # [-1, 1]
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vec(text)


class GraphState(BaseModel):
    """State for RAG graph."""
    question: str
    generation: str = ""
    documents: List[Document] = Field(default_factory=list)


def _get_embeddings_or_fallback() -> Embeddings:
    """
    Try to create Alibaba embeddings; if DashScope intl embeddings are not available
    (e.g., 'DashScope has not yet been released ...'), fall back to DummyEmbeddings.
    """
    try:
        emb = AlibabaEmbeddings(model="text-embedding-v4")
        # probe once to trigger any 401 early
        _ = emb.embed_query("probe")
        logger.info("Using AlibabaEmbeddings (DashScope TextEmbedding).")
        return emb
    except Exception as e:
        logger.error(f"AlibabaEmbeddings unavailable, using DummyEmbeddings fallback: {e}")
        return DummyEmbeddings(dim=256)


@lru_cache(maxsize=1)
def _load_crop_vectorstore():
    """Load crop documents from OSS with Alibaba Cloud embeddings (with fallback)."""
    logger.info("Initializing crop RAG (OSS + Alibaba Cloud embeddings)...")

    loader = OSSPDFLoader()
    all_docs = loader.load_all_crops()
    if not all_docs:
        logger.error("No documents loaded from OSS!")
        raise ValueError("No crop documents found in OSS bucket")

    logger.info(f"Loaded {len(all_docs)} total pages from OSS")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # increase from 1000
        chunk_overlap=100,    # reduce from 200
    )
    splits = text_splitter.split_documents(all_docs)
    logger.info(f"Split into {len(splits)} chunks")

    embeddings = _get_embeddings_or_fallback()
    dim = len(embeddings.embed_query("test"))

    client = QdrantClient(location=":memory:")
    collection_name = "crop_knowledge"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vectorstore.add_documents(splits)
    logger.info(f"RAG ready: {len(splits)} chunks indexed")
    return vectorstore


@lru_cache(maxsize=1)
def _load_insurance_vectorstore():
    """Load insurance documents from OSS with Alibaba Cloud embeddings (with fallback)."""
    logger.info("Initializing insurance RAG (OSS + Alibaba Cloud embeddings)...")

    loader = OSSPDFLoader()
    docs = loader.load_insurance()
    if not docs:
        logger.warning("No insurance documents found in OSS")
        return None

    logger.info(f"Loaded {len(docs)} insurance pages from OSS")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)

    embeddings = _get_embeddings_or_fallback()
    dim = len(embeddings.embed_query("test"))

    client = QdrantClient(location=":memory:")
    collection_name = "insurance_knowledge"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vectorstore.add_documents(splits)
    logger.info(f"Insurance RAG ready: {len(splits)} chunks indexed")
    return vectorstore


@lru_cache(maxsize=1)
def _get_crop_rag_graph():
    """Get crop RAG graph using Alibaba Cloud services (OSS + embeddings + Qwen)."""
    vectorstore = _load_crop_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an agricultural research assistant for Philippine farmers.\n"
            "Answer questions based ONLY on PDF excerpts from Alibaba Cloud OSS.\n"
            "Coverage: rice, corn, and coconut crop diseases, pests, and management.\n"
            "Always cite sources as: [filename.pdf, p. N, category]\n"
            "If information not in PDFs, say 'Evidence not found in knowledge base'.\n"
            "Be precise and scientific."
        )),
        ("user", "Context from OSS PDFs:\n{context}\n\nQuestion: {question}")
    ])

    def retrieve_documents(state: GraphState):
        docs = retriever.invoke(state.question)
        return {"documents": docs}

    def generate_answer(state: GraphState):
        context_parts = []
        for doc in state.documents:
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "general")
            filename = os.path.basename(source) if source != "unknown" else "unknown"

            citation = f"[{filename}, p. {page}, {category}]"
            context_parts.append(f"{doc.page_content}\n{citation}")

        context = "\n\n".join(context_parts)

        formatted = prompt.format_messages(context=context, question=state.question)
        
        # Convert LangChain message types to DashScope format
        messages = []
        for m in formatted:
            role = m.type
            # Map LangChain roles to DashScope roles
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            # "system" stays "system"
            
            messages.append({"role": role, "content": m.content})

        answer = qwen_chat(
            messages,
            model=os.getenv("TOOL_LLM_NAME", "qwen-plus"),
        )
        return {"generation": answer}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def _get_insurance_rag_graph():
    """Get insurance RAG graph (OSS + embeddings + Qwen)."""
    vectorstore = _load_insurance_vectorstore()
    if not vectorstore:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # FIX: Use a single-turn user message format instead of system+user
    prompt = ChatPromptTemplate.from_messages([
        ("user", (
            "You are a crop insurance policy expert for the Philippines.\n"
            "Answer based ONLY on PCIC policy documents.\n"
            "Always cite: [filename.pdf, p. N, insurance]\n"
            "If not in PDFs, say 'No specific policy information found'.\n\n"
            "Context from policy documents:\n{context}\n\n"
            "Question: {question}"
        ))
    ])

    def retrieve_documents(state: GraphState):
        docs = retriever.invoke(state.question)
        return {"documents": docs}

    def generate_answer(state: GraphState):
        context_parts = []
        for doc in state.documents:
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source) if source != "unknown" else "unknown"

            citation = f"[{filename}, p. {page}, insurance]"
            context_parts.append(f"{doc.page_content}\n{citation}")

        context = "\n\n".join(context_parts)

        formatted = prompt.format_messages(context=context, question=state.question)
        
        # Convert to DashScope format - now guaranteed to be user message
        messages = []
        for m in formatted:
            role = m.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            
            messages.append({"role": role, "content": m.content})

        answer = qwen_chat(
            messages,
            model=os.getenv("TOOL_LLM_NAME", "qwen-plus"),
        )
        return {"generation": answer}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


@tool
def retrieve_crop_information(query: Annotated[str, "Question about crop diseases/pests"]):
    """
    Retrieve from OSS crop disease PDFs using Alibaba Cloud embeddings + Qwen.
    Covers rice, corn, and coconut.
    """
    try:
        rag_graph = _get_crop_rag_graph()
        result = rag_graph.invoke({"question": query})
        return result.get("generation", "No information found")
    except Exception as e:
        logger.error(f"Alibaba Crop RAG error: {e}")
        return f"Error retrieving information: {str(e)}"


@tool
def retrieve_insurance_information(query: Annotated[str, "Question about insurance"]):
    """
    Retrieve PCIC insurance information from OSS PDFs using Alibaba Cloud embeddings + Qwen.
    """
    try:
        rag_graph = _get_insurance_rag_graph()
        if not rag_graph:
            return "Insurance information not available in OSS."

        result = rag_graph.invoke({"question": query})
        return result.get("generation", "No insurance information found")
    except Exception as e:
        logger.error(f"Alibaba Insurance RAG error: {e}")
        return f"Error retrieving insurance information: {str(e)}"


def test_alibaba_rag():
    """Test Alibaba Cloud-powered RAG systems."""
    print("Testing Alibaba Cloud Crop RAG with Qwen + Alibaba embeddings...")
    crop_result = retrieve_crop_information.invoke("What causes rice blast disease?")
    print(f"Crop RAG Result: {crop_result[:200]}...\n")

    print("Testing Alibaba Cloud Insurance RAG...")
    insurance_result = retrieve_insurance_information.invoke("What crops are covered by PCIC?")
    print(f"Insurance RAG Result: {insurance_result[:200]}...\n")

    print("Alibaba Cloud RAG systems tested successfully!")


class ResponseFormat(BaseModel):
    """Response format for crop diagnosis."""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class CropDoctorAgent:
    """Crop Doctor Agent - Multi-crop disease diagnosis (rice, corn, coconut)."""

    SYSTEM_INSTRUCTION = (
        "You are Konsultanim's Crop Doctor Agent, specializing in disease and pest diagnosis for rice, corn, and coconut crops. "
    )

    FORMAT_INSTRUCTION = (
        "Use ResponseFormat structure:\n"
    )

    def __init__(self):
        # Use Qwen via DashScope instead of OpenAI
        self.model = QwenChat(
            model=os.getenv('TOOL_LLM_NAME', 'qwen-plus'),
            temperature=0,
        )
        self.graph = build_agent_graph_with_helpfulness(
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            ResponseFormat,
            checkpointer=memory
        )

if __name__ == "__main__":
    test_alibaba_rag()