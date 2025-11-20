"""LangChain wrapper for Qwen with proper tool calling support."""
import os
import json
from typing import Any, List, Optional, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from pydantic import Field

import dashscope


class QwenChat(BaseChatModel):
    """Qwen chat model via DashScope API with tool calling support."""
    
    model: str = Field(default="qwen-plus")
    temperature: float = Field(default=0.7)
    _tools: Optional[List[BaseTool]] = None
    
    @property
    def _llm_type(self) -> str:
        return "qwen-dashscope"
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to DashScope format."""
        converted = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, ToolMessage):
                # Skip tool messages for now - Qwen handles them differently
                continue
            else:
                role = "user"
            
            content = m.content
            # Handle dict content (from structured output)
            if isinstance(content, dict):
                content = json.dumps(content)
            
            converted.append({"role": role, "content": content})
        return converted
    
    def _convert_tools_to_functions(self) -> Optional[List[dict]]:
        """Convert LangChain tools to Qwen function calling format."""
        if not self._tools:
            return None
        
        functions = []
        for tool in self._tools:
            # Extract schema from tool
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema()
                parameters = {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", [])
                }
            else:
                parameters = {"type": "object", "properties": {}}
            
            functions.append({
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": parameters
            })
        
        return functions
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from Qwen."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        
        converted_messages = self._convert_messages(messages)
        
        # Build request params
        params = {
            "api_key": api_key,
            "model": self.model,
            "messages": converted_messages,
            "result_format": "message",
            "temperature": self.temperature,
        }
        
        # Add tools if bound
        tools_param = self._convert_tools_to_functions()
        if tools_param:
            params["tools"] = [{"type": "function", "function": f} for f in tools_param]
        
        try:
            resp = dashscope.Generation.call(**params)
            
            if getattr(resp, "status_code", 200) != 200:
                raise RuntimeError(
                    f"DashScope error: status_code={getattr(resp, 'status_code', None)}, "
                    f"code={getattr(resp, 'code', None)}, message={getattr(resp, 'message', resp)}"
                )
            
            # Extract response
            choice = resp.output.choices[0]
            message_content = choice.message.content or ""
            
            # Check for tool calls - handle both dict and object formats
            tool_calls = []
            raw_tool_calls = getattr(choice.message, 'tool_calls', None)
            
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    # Handle dict format (from DashScope API)
                    if isinstance(tc, dict):
                        function_data = tc.get('function', {})
                        tool_calls.append({
                            "name": function_data.get('name', ''),
                            "args": json.loads(function_data.get('arguments', '{}')),
                            "id": tc.get('id', f"call_{function_data.get('name', 'unknown')}"),
                            "type": "tool_call"
                        })
                    # Handle object format (if DashScope returns objects)
                    else:
                        tool_calls.append({
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments),
                            "id": getattr(tc, 'id', f"call_{tc.function.name}"),
                            "type": "tool_call"
                        })
            
            # Create AIMessage with tool calls
            message = AIMessage(
                content=message_content,
                additional_kwargs={
                    "tool_calls": tool_calls if tool_calls else None
                }
            )
            
            # Add tool_calls attribute for LangGraph compatibility
            if tool_calls:
                message.tool_calls = tool_calls
            
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise RuntimeError(f"Qwen API call failed: {e}")
    
    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "QwenChat":
        """Bind tools to the model."""
        new_instance = self.__class__(**self.dict())
        new_instance._tools = tools
        return new_instance
    
    def with_structured_output(self, schema: Any, **kwargs: Any) -> "QwenChat":
        """Return model configured to output structured data."""
        # For now, return self - structured output is handled by the agent graph
        return self
    
    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "tools_count": len(self._tools) if self._tools else 0
        }
