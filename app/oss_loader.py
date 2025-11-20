"""Load PDF documents from Alibaba Cloud OSS."""
import os
import logging
from typing import List
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from .oss_config import OSSConfig

logger = logging.getLogger(__name__)


class OSSPDFLoader:
    """Load PDFs from Alibaba Cloud OSS bucket."""
    
    def __init__(self):
        self.oss = OSSConfig()
    
    def load_from_prefix(self, prefix: str, category: str = None) -> List[Document]:
        """
        Load all PDFs from an OSS folder prefix.
        
        Args:
            prefix: OSS folder path (e.g., 'rice/', 'corn/', 'insurance/')
            category: Metadata category (e.g., 'rice', 'insurance')
        
        Returns:
            List of Document objects with enhanced metadata
        """
        documents = []
        pdf_files = self.oss.list_pdfs(prefix)
        
        logger.info(f"Loading {len(pdf_files)} PDFs from OSS prefix: {prefix}")
        
        for idx, pdf_info in enumerate(pdf_files, 1):
            tmp_path = None
            try:
                with NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp_path = tmp.name
                    self.oss.download_to_temp(pdf_info['key'], tmp_path)
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()

                for doc in docs:
                    doc.metadata['source'] = pdf_info['key']
                    doc.metadata['oss_bucket'] = self.oss.bucket_name
                    doc.metadata['file_size'] = pdf_info['size']
                    doc.metadata['last_modified'] = str(pdf_info['last_modified'])
                    if category:
                        doc.metadata['category'] = category
                    else:
                        doc.metadata['category'] = prefix.split('/')[0] if '/' in prefix else 'general'

                documents.extend(docs)
                logger.info(f"  [{idx}/{len(pdf_files)}] {pdf_info['key']}: {len(docs)} pages")
            except Exception as e:
                logger.error(f"  [{idx}/{len(pdf_files)}] ❌ {pdf_info['key']}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {tmp_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} total pages from {prefix}")
        return documents
    
    def load_all_crops(self) -> List[Document]:
        """Load all crop-related PDFs (rice, corn, coconut)."""
        categories = ['rice', 'corn', 'coconut']
        all_docs = []
        
        for category in categories:
            docs = self.load_from_prefix(f'{category}/', category)
            all_docs.extend(docs)
            logger.info(f"{category.upper()}: {len(docs)} pages loaded")
        
        return all_docs
    
    def load_insurance(self) -> List[Document]:
        """Load insurance policy PDFs."""
        return self.load_from_prefix('insurance/', 'insurance')