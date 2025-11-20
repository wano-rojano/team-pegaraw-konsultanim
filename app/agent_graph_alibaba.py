"""Agent graph builder with Alibaba Cloud tools."""
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing import Literal, Dict, Any
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
import logging

from .tools import get_tool_belt_alibaba

logger = logging.getLogger(__name__)


def build_agent_graph_with_helpfulness_alibaba(
    model,
    system_instruction: str,
    format_instruction: str,
    response_format: type[BaseModel],
    checkpointer: BaseCheckpointSaver = None
):
    """Build agent graph with Alibaba Cloud tools and strong stopping logic."""
    
    # Get Alibaba Cloud tool belt
    tools = get_tool_belt_alibaba()
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: MessagesState) -> Dict[str, Any]:
        messages = state['messages']
        iteration_count = state.get('iteration_count', 0)
        max_iterations = 5
        
        logger.info(f"[AgentGraph] Iteration {iteration_count + 1}/{max_iterations}")
        
        # Force completion if max iterations reached
        if iteration_count >= max_iterations:
            logger.warning(f"[AgentGraph] Max iterations reached, forcing completion")
            
            # Extract last meaningful content
            last_content = "Based on the information gathered, please provide more specific details about your crop issue."
            for msg in reversed(messages):
                content = getattr(msg, 'content', '')
                if isinstance(content, str) and len(content) > 100 and 'tool' not in content.lower():
                    last_content = content
                    break
            
            structured_response = response_format(
                status='completed',
                message=last_content
            )
            return {
                'messages': messages,
                'structured_response': structured_response,
                'iteration_count': iteration_count
            }
        
        # Add system instruction with STRONG stopping directive
        stop_directive = (
            "\n\n**CRITICAL STOPPING RULE**: "
            "After you receive tool results (especially from retrieve_crop_information), "
            "you MUST immediately provide your final answer. "
            "DO NOT call any more tools. Analyze the retrieved information and respond directly."
        )
        
        system_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get('role') == 'system']
        if not system_messages:
            full_instruction = f"{system_instruction}\n\n{format_instruction}{stop_directive}"
            messages = [{'role': 'system', 'content': full_instruction}] + list(messages)
        
        try:
            response = model_with_tools.invoke(messages)
            new_iteration_count = iteration_count + 1
            
            has_tool_calls = bool(getattr(response, 'tool_calls', None))
            logger.info(f"[AgentGraph] Has tool_calls: {has_tool_calls}")
            
            if not has_tool_calls and new_iteration_count >= 2:
                logger.info("[AgentGraph] No tool calls after tool use - forcing completion")
                
                content = response.content if hasattr(response, 'content') else str(response)
                
                # If response is too short, extract from tool results
                if len(content) < 150:
                    logger.info("[AgentGraph] Response too short, extracting from tool results")
                    for msg in reversed(messages):
                        msg_content = getattr(msg, 'content', '')
                        if isinstance(msg_content, str) and len(msg_content) > 150:
                            # Check if it's a tool result (not a user query)
                            if not any(keyword in msg_content.lower() for keyword in ['my', 'i have', 'what is', 'how to']):
                                content = f"Based on the retrieved information:\n\n{msg_content}"
                                break
                
                structured_response = response_format(
                    status='completed',
                    message=content
                )
                
                return {
                    'messages': [response],
                    'structured_response': structured_response,
                    'iteration_count': new_iteration_count
                }
            
            if has_tool_calls and new_iteration_count > 1:
                current_tool = response.tool_calls[0]['name'] if response.tool_calls else None
                
                # Look for recent tool calls
                recent_tools = []
                for msg in reversed(messages[-5:]):  # Check last 5 messages
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        recent_tools.extend([tc['name'] for tc in msg.tool_calls])
                
                if current_tool in recent_tools:
                    logger.warning(f"[AgentGraph] Tool '{current_tool}' called repeatedly - forcing completion")
                    structured_response = response_format(
                        status='completed',
                        message="Please provide more specific details about your crop issue so I can assist you better."
                    )
                    return {
                        'messages': [response],
                        'structured_response': structured_response,
                        'iteration_count': new_iteration_count
                    }
            
            return {
                'messages': [response],
                'iteration_count': new_iteration_count
            }
            
        except Exception as e:
            logger.error(f"[AgentGraph] Error: {e}", exc_info=True)
            error_msg = f"Error processing request: {str(e)}"
            structured_error = response_format(
                status='error',
                message=error_msg
            )
            return {
                'messages': [AIMessage(content=error_msg)],
                'structured_response': structured_error,
                'iteration_count': iteration_count + 1
            }
    
    def should_continue(state: MessagesState) -> Literal['tools', 'end']:
        messages = state['messages']
        last_message = messages[-1]
        iteration_count = state.get('iteration_count', 0)
        
        # Always end if structured response exists
        if state.get('structured_response'):
            logger.info("[AgentGraph] Structured response exists - ENDING")
            return 'end'
        
        # Always end if max iterations reached
        if iteration_count >= 5:
            logger.info("[AgentGraph] Max iterations (5) reached - ENDING")
            return 'end'
        
        # Check for tool calls
        has_tool_calls = (
            (hasattr(last_message, 'tool_calls') and last_message.tool_calls) or
            (hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'))
        )
        
        if has_tool_calls:
            logger.info(f"[AgentGraph] Tool calls detected at iteration {iteration_count + 1}")
            return 'tools'
        
        logger.info("[AgentGraph] No tool calls - ENDING")
        return 'end'
    
    # Build graph
    workflow = StateGraph(MessagesState)
    workflow.add_node('agent', call_model)
    workflow.add_node('tools', ToolNode(tools))
    workflow.set_entry_point('agent')
    
    workflow.add_conditional_edges(
        'agent',
        should_continue,
        {
            'tools': 'tools',
            'end': END
        }
    )
    workflow.add_edge('tools', 'agent')
    
    compiled = workflow.compile(checkpointer=checkpointer)
    logger.info("[AgentGraph] Graph compiled with aggressive stopping (max 5 iterations)")
    return compiled