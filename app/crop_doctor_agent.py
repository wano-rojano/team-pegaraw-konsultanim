"""Konsultanim Crop Doctor Agent - Rice, Corn, and Coconut disease diagnosis."""
import os
import logging
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .qwen_langchain import QwenChat
from .agent_graph_alibaba import build_agent_graph_with_helpfulness_alibaba

memory = MemorySaver()
logger = logging.getLogger(__name__)


class ResponseFormat(BaseModel):
    """Response format for crop diagnosis."""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class CropDoctorAgent:
    """Crop Doctor Agent - Multi-crop disease diagnosis (rice, corn, coconut)."""

    SYSTEM_INSTRUCTION = (
        "You are Konsultanim's Crop Doctor Agent, specializing in disease and pest diagnosis for rice, corn, and coconut crops. "
        "You have access to: (1) crop-specific PDF libraries on Alibaba Cloud OSS covering diseases, pests, and management; "
        "(2) web search; (3) academic research databases (PubMed, arXiv).\n\n"
        
        "**TOOL-USE POLICY:**\n"
        "1. ALWAYS use retrieve_crop_information FIRST - it queries Alibaba Cloud OSS for rice, corn, and coconut PDFs\n"
        "2. After getting tool results, IMMEDIATELY provide your complete answer - do NOT call more tools\n"
        "3. Use web/academic search ONLY if PDFs completely lack evidence\n"
        "4. Cite sources: [filename.pdf, p. N, crop_type] or [URL] or (arXiv:ID)\n"
        "5. Never invent citations - if no source found, state 'Evidence not found in OSS library'\n\n"
        
        "Before answering, check if you have enough context. If details are missing, ask concise clarifying questions first. "
        "Key details to request for diagnosis: crop type (rice/corn/coconut), province/region in Philippines, growth stage, cultivar/variety if known, "
        "symptoms (plant part: leaf/sheath/stem/panicle/grain/trunk/root; lesion color/shape/size; presence of mycelia/spores/exudates), "
        "field distribution and incidence/severity, recent weather (humidity, temperature, rainfall), field history/rotation, "
        "recent inputs (fertilizer, pesticide, seed treatments), and irrigation/drainage.\n\n"
        
        "**RESPONSE STRUCTURE (when sufficient info available):**\n"
        "1. **Likely Diagnosis**: Disease/pest name, confidence level (low/medium/high)\n"
        "2. **Differential Diagnoses**: How to distinguish\n"
        "3. **Immediate Actions**: Prioritized steps (cultural first, then others)\n"
        "4. **Integrated Management**:\n"
        "   - Cultural controls\n"
        "   - Mechanical controls\n"
        "   - Biological controls (BCAs)\n"
        "   - Chemical controls (active ingredients only, no brands; include resistance management, PPE, label compliance)\n"
        "5. **Monitoring**: What to watch and thresholds\n"
        "6. **Additional Information Needed**: If uncertainty remains\n"
        "7. **Sources**: All citations with [filename.pdf, p. N, crop] format\n\n"
        
        "**CRITICAL**: Once you call retrieve_crop_information and get results, analyze them and provide your COMPLETE answer immediately. "
        "Do NOT make additional tool calls unless the farmer asks a follow-up question."
    )

    FORMAT_INSTRUCTION = (
        "Use ResponseFormat structure:\n"
        "- status='input_required': List missing info as bullet questions\n"
        "- status='error': Brief, actionable error message\n"
        "- status='completed': Full diagnosis with all sections listed in SYSTEM_INSTRUCTION\n"
        "Always cite ACTUAL sources retrieved from Alibaba Cloud OSS - never invent.\n"
        "After tool results are available, provide your complete answer WITHOUT calling more tools."
    )

    def __init__(self):
        self.model = QwenChat(
            model=os.getenv('TOOL_LLM_NAME', 'qwen-plus'),
            temperature=0,
        )
        self.graph = build_agent_graph_with_helpfulness_alibaba(
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            ResponseFormat,
            checkpointer=memory
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {
            'messages': [('user', query)],
            'iteration_count': 0
        }
        
        config = {
            'configurable': {'thread_id': context_id},
            'recursion_limit': 50
        }

        logger.info(f"[CropDoctor] Starting stream for query: {query[:100]}...")

        try:
            for item in self.graph.stream(inputs, config, stream_mode='values'):
                message = item['messages'][-1]
                
                if isinstance(message, AIMessage) and message.tool_calls:
                    tool_name = message.tool_calls[0]['name']
                    
                    if 'crop' in tool_name or 'retrieve' in tool_name:
                        feedback = '📚 Searching crop disease database (Alibaba Cloud OSS)...'
                    elif 'weather' in tool_name:
                        feedback = '🌦️ Fetching weather data...'
                    elif 'search' in tool_name:
                        feedback = '🔍 Searching additional sources...'
                    else:
                        feedback = f'⚙️ Using {tool_name.replace("_", " ")}...'

                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': feedback,
                    }
                elif isinstance(message, ToolMessage):
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': 'Analyzing results with Qwen...',
                    }
        except Exception as e:
            logger.error(f"[CropDoctor] Stream error: {e}", exc_info=True)
            # Return error response
            yield {
                'is_task_complete': False,
                'require_user_input': True,
                'content': f'Error processing request: {str(e)}',
            }
            return

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        
        logger.info(f"[CropDoctor] Getting final response. Has structured_response: {structured_response is not None}")
        
        if structured_response:
            logger.info(f"[CropDoctor] Structured response status: {structured_response.status}")
            logger.info(f"[CropDoctor] Message length: {len(structured_response.message)}")
        
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        logger.warning("[CropDoctor] No valid structured response - returning error")
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'Unable to process request. Please provide more details about your crop issue.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
