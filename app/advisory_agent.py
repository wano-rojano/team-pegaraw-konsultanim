import os
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from .qwen_langchain import QwenChat
from .agent_graph_alibaba import build_agent_graph_with_helpfulness_alibaba

memory = MemorySaver()


class AdvisoryResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class AdvisoryAgent:
    """Advisory Agent - provides weather forecasts, and disease risk asssessments with RAG support."""

    SYSTEM_INSTRUCTION = (
        "You are Konsultanim's Agricultural Advisory Agent for rice, corn, and coconut farmers in the Philippines. "
        "Your role is to provide evidence-based farming advisories based on weather forecasts and disease risk assessment. "
        "\n\n"
        "**TOOL-USE POLICY:**\n"
        "For weather-based advisories, follow this sequence:\n"
        "1. Use get_weather_forecast to obtain current and forecast weather data\n"
        "2. Use retrieve_crop_information to query Alibaba Cloud OSS PDFs about:\n"  # CHANGED
        "   - Which diseases are favored by the observed weather conditions for the farmer's crop (rice/corn/coconut)\n"
        "   - Evidence-based relationships between weather (temp, humidity, rainfall) and disease development\n"
        "   - Preventive management strategies for identified risks\n"
        "3. Use web search or academic search ONLY if PDF library lacks specific information\n"
        "\n\n"
        "**CRITICAL**\n"
        "- Never make disease risk assessments without consulting the OSS PDF library first\n"  # CHANGED
        "- Always cite the source of disease-weather relationships: [filename.pdf, p. N]\n"
        "- If PDFs don't establish a clear relationship, state 'Evidence not found in OSS library'\n"  # CHANGED
        "- Base all disease risk assessments on documented research, not general assumptions\n"
        "\n\n"
        "**Advisory Structure:**\n"
        "1) Weather Summary (data from get_weather_forecast with citation)\n"
        "2) Disease Risk Assessment (ONLY if evidence found in PDFs, with citations)\n"
        "3) Evidence-based Recommendations (from PDF library with citations)\n"
        "4) Immediate Actions (prioritized, with timing)\n"
        "5) Monitoring Guidelines\n"
        "6) Sources (cite all tools used)\n"
        "\n\n"
        "**Citation Standards:**\n"
        "- Weather data: [Open-Meteo Weather API]\n"
        "- Disease information: [filename.pdf, p. N]\n"
        "- If no PDF evidence: State 'No specific evidence found in OSS library'\n"  # CHANGED
        "\n\n"
        "Be scientifically cautious. If the PDF library doesn't document a weather-disease relationship, "
        "don't claim one exists. Advise farmers to monitor for symptoms and consult local extension officers."
    )

    FORMAT_INSTRUCTION = (
        "Respond using the ResponseFormat structure.\n"
        "\n"
        "Set status='input_required' if you need more information from the farmer. "
        "In message, list what information is needed.\n"
        "\n"
        "Set status='error' if there's a problem accessing data or tools. "
        "In message, explain what went wrong.\n"
        "\n"
        "Set status='completed' when you have a complete advisory. "
        "In message, provide the full advisory with all sections:\n"
        "- Weather Summary\n"
        "- Disease Risk Assessment (if applicable)\n"
        "- Recommendations\n"
        "- Immediate Actions\n"
        "- Monitoring Guidelines\n"
        "- Sources\n"
        "\n"
        "Always cite sources that were actually retrieved from Alibaba Cloud OSS. Never invent citations."  # CHANGED
    )

    def __init__(self):
        # Use Qwen via DashScope
        self.model = QwenChat(
            model=os.getenv('TOOL_LLM_NAME', 'qwen-plus'),
            temperature=0,
        )
        
        self.graph = build_agent_graph_with_helpfulness_alibaba(
            self.model,
            self.SYSTEM_INSTRUCTION,
            self.FORMAT_INSTRUCTION,
            AdvisoryResponseFormat,
            checkpointer=memory
        )

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        """Stream advisory responses - same pattern as CropDoctorAgent."""
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
            
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                tool_name = message.tool_calls[0]['name']
                if 'weather' in tool_name:
                    feedback = '🌦️ Fetching weather forecast...'
                elif 'retrieve' in tool_name or 'rag' in tool_name:
                    feedback = '📚 Searching disease management documents (Alibaba Cloud OSS)...'
                elif 'search' in tool_name:
                    feedback = '🔍 Searching for additional information...'
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

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        """Extract final advisory response from agent state."""
        current_state = self.graph.get_state(config)
        
        structured_response = current_state.values.get('structured_response')
        
        if structured_response and hasattr(structured_response, 'status') and hasattr(structured_response, 'message'):
            status = structured_response.status
            message = structured_response.message
            
            if status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': message,
                }
            elif status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': message,
                }
            elif status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': message,
                }
        
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your advisory request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def format_weather_summary(self, data: dict, days_ahead: int) -> str:
        """Format the weather summary section of the advisory."""
        matching_province = data.get('matching_province', 'the specified location')
        past_rain = data.get('past_rain', 0)
        forecast_rain = data.get('forecast_rain', 0)
        forecast_tmin = data.get('forecast_tmin', 0)
        forecast_tmax = data.get('forecast_tmax', 0)
        forecast_tavg = data.get('forecast_tavg', 0)
        forecast_humidity = data.get('forecast_humidity', 0)

        return f"""Weather Summary for {matching_province}, Philippines

**Past 7 Days:**
- Total rainfall: {past_rain:.1f} mm

**Next {min(days_ahead, 16)} Days Forecast:**
- Expected rainfall: {forecast_rain:.1f} mm
- Temperature range: {forecast_tmin:.1f}°C (min) to {forecast_tmax:.1f}°C (max)
- Average temperature: {forecast_tavg:.1f}°C
- Average relative humidity: {forecast_humidity:.1f}%

**For Disease Risk Assessment:**
Use the retrieve_crop_information tool to query Alibaba Cloud OSS about diseases associated with these conditions for the farmer's crop (rice, corn, or coconut):
- Temperature: {forecast_tavg:.1f}°C
- Humidity: {forecast_humidity:.1f}%
- Rainfall: {forecast_rain:.1f} mm over {min(days_ahead, 16)} days

Source: [Open-Meteo Weather API]"""