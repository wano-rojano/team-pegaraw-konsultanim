"""Konsultanim Chainlit Chat Interface - Multi-crop agricultural assistant."""
import os
import json
import httpx
import chainlit as cl
from uuid import uuid4
from dotenv import load_dotenv
import logging

load_dotenv()

BASE_URL = os.getenv("A2A_BASE_URL", "http://localhost:10000")

logger = logging.getLogger(__name__)

def format_response(text: str) -> str:
    """Format the agent response for better readability."""
    replacements = {
        # Crop Doctor formatting
        "Likely Diagnosis:": "🔬 **Likely Diagnosis:**",
        "Key Symptoms:": "🔍 **Key Symptoms:**",
        "Differential Diagnoses:": "📋 **Differential Diagnoses:**",
        "Immediate actions:": "⚡ **Immediate Actions:**",
        "Integrated management:": "🚫 **Integrated Management:**",
        "Monitoring": "🕵🏼 **Monitoring:**",
        "Additional Info Needed:": "❓ **Additional Information Needed:**",
        "Sources:": "📚 **Sources:**",
        
        # Advisory Agent formatting
        "Weather Summary:": "🌦️ **Weather Summary:**",
        "Disease Risk Assessment:": "⚠️ **Disease Risk Assessment:**",
        "Evidence-based Recommendations:": "📄 **Evidence-based Recommendations:**",
        "Monitoring Guidelines:": "🕵🏼 **Monitoring Guidelines:**",
        
        # Insurance Agent formatting
        "Query Summary:": "📑 **Query Summary:**",
        "Policy Information:": "📄 **Policy Information:**",
        "Action Steps:": "📋 **Action Steps:**",
        "Required Documents:": "📚 **Required Documents:**",
        "Contact Information:": "📞 **Contact Information:**",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with Konsultanim."""
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        transport=httpx.AsyncHTTPTransport(retries=3),
    )
    cl.user_session.set("httpx", httpx_client)
    
    # Multi-crop quick actions
    actions = [
        cl.Action(
            name="rice_diseases", 
            value="rice_diseases", 
            label="🌾 Rice Diseases", 
            payload={}
        ),
        cl.Action(
            name="corn_pests", 
            value="corn_pests", 
            label="🌽 Corn Pests", 
            payload={}
        ),
        cl.Action(
            name="coconut_diseases", 
            value="coconut_diseases", 
            label="🌴 Coconut Diseases", 
            payload={}
        ),
        cl.Action(
            name="weather_advisory", 
            value="weather_advisory", 
            label="🌦️ Weather Advisory", 
            payload={}
        ),
        cl.Action(
            name="crop_insurance", 
            value="crop_insurance", 
            label="📑 Crop Insurance", 
            payload={}
        ),
    ]
    
    welcome_message = (
        "🤖 **Welcome to Konsultanim!**\n\n"
        "I'm your AI-powered agricultural assistant. I can help with:\n\n"
        "🔬 **Disease Diagnosis** - Rice, corn, and coconut diseases & pests\n"
        "🚫 **Integrated Pest Management** - Evidence-based IPM strategies\n"
        "📑 **Crop Insurance** - Policy guidance and claims assistance\n"
        "🌦️ **Weather Forecast** - Advisories and disease risk assessments\n\n"
        "📚 *I use curated references and academic papers, weather data, and official policy documents to provide accurate advice.*\n\n"
        "**Quick Actions:** Click the buttons below or ask me anything!"
    )
    
    await cl.Message(content=welcome_message, actions=actions).send()

# ==================== QUICK ACTION CALLBACKS ====================

@cl.action_callback("rice_diseases")
async def on_rice_diseases(action):
    """Quick action for rice disease queries."""
    await cl.Message(
        content=(
            "🌾 **Rice Disease and Pest Diagnosis and Management**\n\n"
            "To help diagnose rice diseases, please provide:\n"
            "• **Symptoms**: Describe what you see (leaf spots or lesions, discoloration, etc.)\n"
            "• **Location**: Your province/region\n"
            "• **Growth stage**: Vegetative, reproductive, ripening, etc.\n"
            "• **Field conditions**: Irrigation type, recent weather\n\n"
            "**Example:** *'My rice plants have greenish gray lesions on leaves, Laguna, reproductive stage, irrigated field'*"
        )
    ).send()

@cl.action_callback("corn_pests")
async def on_corn_pests(action):
    """Quick action for corn pest queries."""
    await cl.Message(
        content=(
            "🌽 **Corn Disease and Pest Diagnosis and Management**\n\n"
            "To help with corn issues, please describe:\n"
            "• **Problem**: Pest damage, disease symptoms, or general concern\n"
            "• **Growth stage**: Vegetative, reproductive, maturing\n"
            "• **Location**: Your province/region\n"
            "• **Observations**: What you see on leaves, stalks, or ears\n\n"
            "**Example:** *'Corn plants have holes in leaves, Isabela, vegetative stage'*"
        )
    ).send()

@cl.action_callback("coconut_diseases")
async def on_coconut_diseases(action):
    """Quick action for coconut disease queries."""
    await cl.Message(
        content=(
            "🌴 **Coconut Disease and Pest Diagnosis and Management**\n\n"
            "For coconut-related questions, please share:\n"
            "• **Issue**: Disease symptoms, pest damage, or general care question\n"
            "• **Tree age**: Seedling, young palm, mature tree\n"
            "• **Location**: Your province/region\n"
            "• **Symptoms**: Yellowing, wilting, trunk damage, etc.\n\n"
            "**Example:** *'Yellow spots on leaves and rounded coconuts, Quezon, mature trees, coastal area'*"
        )
    ).send()

@cl.action_callback("weather_advisory")
async def on_weather_advisory(action):
    """Quick action for weather advisory."""
    await cl.Message(
        content=(
            "🌦️ **Weather-Based Farming Advisory**\n\n"
            "I can provide weekly weather forecasts and disease risk assessments.\n\n"
            "Please specify:\n"
            "• **Location**: Province or municipality\n"
            "• **Crop**: Rice, corn, or coconut\n"
            "• **Time frame**: Weekly, 7-day, or 16-day forecast\n\n"
            "**Example:** *'Weekly weather advisory for rice in Laguna'*"
        )
    ).send()

@cl.action_callback("crop_insurance")
async def on_crop_insurance(action):
    """Quick action for crop insurance queries."""
    await cl.Message(
        content=(
            "📑 **Crop Insurance Assistance**\n\n"
            "I can help you understand:\n"
            "• Coverage and eligibility requirements\n"
            "• Claims process and documentation\n"
            "**Example questions:**\n"
            "• *'How do I file a crop insurance claim for rice?'*\n"
            "• *'What crops are covered by PCIC insurance?'*\n"
            "• *'What documents do I need for corn insurance enrollment?'*"
        )
    ).send()

# ==================== MESSAGE HANDLER ====================

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    httpx_client: httpx.AsyncClient = cl.user_session.get("httpx")
    
    # Show thinking indicator
    thinking_msg = cl.Message(content="🤔 Processing your query...")
    await thinking_msg.send()
    
    # Prepare A2A protocol payload
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message.content}],
                "message_id": uuid4().hex,
            }
        },
        "id": str(uuid4())
    }
    
    # Add context if exists
    context_id = cl.user_session.get("context_id")
    if context_id:
        payload["params"]["context_id"] = context_id
    
    try:
        # Send request to A2A server
        resp = await httpx_client.post(f"{BASE_URL}/", json=payload)
        resp.raise_for_status()
        data = resp.json()
        await thinking_msg.remove()

        text = None
        task_id = None
        new_context_id = None
        
        # Parse response
        if "result" in data:
            result = data["result"]
            
            if isinstance(result, dict):
                task_id = result.get("id")
                new_context_id = result.get("contextId")
                
                # Extract from artifacts (completed tasks)
                if "artifacts" in result and result.get("status", {}).get("state") == "completed":
                    artifacts = result["artifacts"]
                    for artifact in artifacts:
                        if artifact.get("name") == "result" and "parts" in artifact:
                            parts = artifact["parts"]
                            texts = [p.get("text") for p in parts if p.get("kind") == "text" and p.get("text")]
                            text = "\n".join(texts) if texts else None
                            if text:
                                break
                
                # Fallback: extract from status message
                if not text and "status" in result and "message" in result["status"]:
                    status_msg = result["status"]["message"]
                    if isinstance(status_msg, dict) and "parts" in status_msg:
                        parts = status_msg.get("parts", [])
                        texts = [p.get("text") for p in parts if p.get("kind") == "text" and p.get("text")]
                        text = "\n".join(texts) if texts else None
                        
        # Check for JSON-RPC error
        if not text and "error" in data:
            error = data["error"]
            text = f"❌ Server error: {error.get('message', 'Unknown error')}"

        # Send formatted response
        if text:
            if "error processing" in text.lower() or "try again" in text.lower():
                await cl.Message(
                    content=(
                        f"⚠️ {text}\n\n"
                        "💡 **Try asking more specific questions like:**\n"
                        "• What causes rice blast disease?\n"
                        "• How to manage corn borer in Nueva Ecija?\n"
                        "• Symptoms of coconut Cadang-Cadang disease\n"
                        "• Weekly weather advisory for Laguna\n"
                        "• How to file crop insurance claim?"
                    )
                ).send()
            else:
                formatted_text = format_response(text)
                await cl.Message(content=formatted_text).send()
        else:
            await cl.Message(
                content=(
                    f"⚠️ Couldn't parse response. Raw data:\n"
                    f"```json\n{json.dumps(data, indent=2)}\n```"
                )
            ).send()

        # Save context for conversation continuity
        if task_id:
            cl.user_session.set("task_id", task_id)
        if new_context_id:
            cl.user_session.set("context_id", new_context_id)
        
    except httpx.HTTPError as e:
        await thinking_msg.remove()
        await cl.Message(content=f"❌ Connection error: {e}\n\nMake sure the A2A server is running at `{BASE_URL}`").send()
    except Exception as e:
        await thinking_msg.remove()
        logger.error(f"Unexpected error: {e}", exc_info=True)
        await cl.Message(content=f"❌ Unexpected error: {e}").send()

@cl.on_chat_end
async def on_chat_end():
    """Clean up on chat end."""
    httpx_client: httpx.AsyncClient = cl.user_session.get("httpx")
    if httpx_client:
        await httpx_client.aclose()