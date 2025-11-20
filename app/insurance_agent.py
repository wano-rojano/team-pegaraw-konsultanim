"""Konsultanim Insurance Agent - Crop insurance policy assistance."""
import os
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .qwen_langchain import QwenChat
from .agent_graph_alibaba import build_agent_graph_with_helpfulness_alibaba  # CHANGED

memory = MemorySaver()

class InsuranceResponseFormat(BaseModel):
    """Response format for insurance queries."""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class InsuranceAgent:
    """Insurance Agent - Crop insurance policy guidance."""

    SYSTEM_INSTRUCTION = (
        "You are Konsultanim's Crop Insurance Advisor, helping Filipino farmers understand "
        "and navigate crop insurance policies for rice, corn, and coconut.\n\n"
        
        "**TOOL-USE POLICY:**\n"
        "1. ALWAYS use retrieve_insurance_information FIRST for policy questions\n"
        "2. Use web search ONLY for current rates/deadlines not in PDFs\n"
        "3. Cite all sources: [policy_document.pdf, p. N] or [URL]\n"
        "4. If information not in library, state: 'Please contact [relevant insurance provider]'\n\n"
        
        "**KEY TOPICS YOU HANDLE:**\n"
        "- Policy coverage (what's covered, exclusions)\n"
        "- Eligibility requirements\n"
        "- Claims process and documentation\n"
        "- Crop-specific policies (rice, corn, coconut)\n"
        "- Government subsidy programs (PCIC)\n\n"
        
        "**INFORMATION TO REQUEST:**\n"
        "- Crop type (rice/corn/coconut)\n"
        "- Farm size (hectares)\n"
        "- Location (province)\n"
        "- Specific question (coverage/claim/enrollment/etc.)\n\n"
        
        "**RESPONSE STRUCTURE:**\n"
        "1. **Policy Information**: Coverage, requirements, or process\n"
        "2. **Action Steps**: What farmer needs to do (prioritized)\n"
        "3. **Required Documents**: List what to prepare\n"
        "4. **Deadlines**: Key dates (if applicable)\n"
        "5. **Contact Information**: Where to submit/inquire\n"
        "6. **Sources**: All citations\n\n"
        
        "**IMPORTANT:**\n"
        "- Clarify subsidy eligibility clearly\n"
        "- For claims, stress documentation importance\n"
        "- Never guarantee claim approval - explain process only\n"
        "- Refer complex cases to actual insurance officers"
    )

    FORMAT_INSTRUCTION = (
        "Use InsuranceResponseFormat structure:\n"
        "- status='input_required': Ask for crop type, location, or specific question\n"
        "- status='error': Explain issue (e.g., 'Information not in policy library')\n"
        "- status='completed': Full response with all sections\n"
        "Cite ONLY sources actually retrieved from Alibaba Cloud OSS."  # CHANGED
    )

    def __init__(self):
        # Use Qwen via DashScope
        self.model = QwenChat(
            model=os.getenv('TOOL_LLM_NAME', 'qwen-plus'),
            temperature=0,
        )
        self.graph = build_agent_graph_with_helpfulness_alibaba(  # CHANGED
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            InsuranceResponseFormat,
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

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_name = message.tool_calls[0]['name']
                
                if 'insurance' in tool_name or 'retrieve' in tool_name:
                    feedback = '📋 Checking insurance policy documents (Alibaba Cloud OSS)...'
                elif 'search' in tool_name:
                    feedback = '🔍 Searching for updated information...'
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
                    'content': 'Processing policy information with Qwen...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        
        if structured_response and isinstance(structured_response, InsuranceResponseFormat):
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

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'Unable to process insurance query. Please try again.',
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']