from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import httpx
import asyncio
import os
import logging
from uuid import uuid4
from datetime import datetime, timedelta
import json

from .cron_routes import register_cron_routes 

logger = logging.getLogger(__name__)

class SMSHandler:
    def __init__(self):
        self.base_url = os.getenv("A2A_BASE_URL", "http://localhost:10000")
        # 320 seconds timeout for complex agent processing
        self.client = httpx.AsyncClient(timeout=320.0)
        
        # Twilio for sending
        self.twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        self.from_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Store conversation contexts with timestamps
        self.conversations = {}    

    def get_conversation_context(self, phone_number: str) -> str:
        """Get or create conversation context with cleanup"""
        now = datetime.now()
        
        # Clean old conversations (older than 24 hours)
        day_ago = now - timedelta(days=1)
        old_numbers = [
            num for num, data in self.conversations.items() 
            if data['last_active'] < day_ago
        ]
        for num in old_numbers:
            del self.conversations[num]
        
        # Get or create context
        if phone_number in self.conversations:
            self.conversations[phone_number]['last_active'] = now
            return self.conversations[phone_number]['context_id']
        
        context_id = str(uuid4())
        self.conversations[phone_number] = {
            'context_id': context_id,
            'last_active': now
        }
        return context_id

    async def process_farmer_sms(self, message: str, phone_number: str) -> str:
        """Process farmer SMS through your Konsultanim A2A agent"""
        
        # Get conversation context
        context_id = self.get_conversation_context(phone_number)
        
        # Add SMS-specific instruction
        sms_instruction = "SMS query from farmer - provide concise, actionable response suitable for text message (max 1600 characters). Be direct and actionable. Query: "
        final_message = sms_instruction + message

        # Send to agent via A2A protocol
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": final_message}],
                    "message_id": uuid4().hex,
                }
            },
            "id": str(uuid4())
        }
        
        if context_id:
            payload["params"]["context_id"] = context_id
        
        try:
            logger.info(f"Sending request to A2A server: {self.base_url}")
            logger.info(f"Processing SMS from {phone_number}: {message}")

            # Add explicit timeout handling with asyncio for long waits
            resp = await asyncio.wait_for(
                self.client.post(f"{self.base_url}/", json=payload),
                timeout=300.0  # 300 second timeout
            )

            logger.info(f"A2A server response status: {resp.status_code}")

            resp.raise_for_status()
            data = resp.json()
            
            # Extract response
            response_text = self.extract_response_from_data(data)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing SMS from {phone_number}: {e}")
            return "Konsultanim: I'm having trouble right now. Please try again later."

    def extract_response_from_data(self, data: dict) -> str:
        """Extract response text from A2A response data"""
        
        if "result" in data and "artifacts" in data["result"]:
            artifacts = data["result"]["artifacts"]
            for artifact in artifacts:
                if artifact.get("name") == "result":
                    parts = artifact.get("parts", [])
                    texts = [p.get("text") for p in parts if p.get("text")]
                    if texts:
                        full_text = "\n".join(texts)
                        # Truncate if too long for SMS
                        if len(full_text) > 1600:
                            return full_text[:1550] + "... [Response truncated for SMS]"
                        return full_text
        
        logger.warning(f"Unexpected A2A response format: {data}")
        
        return (
            "Konsultanim received your question. "
            "Can you provide more details about the symptoms, crop type, location, and growth stage? "
            "Example: 'Brown diamond spots on rice leaves, Laguna, reproductive stage'"
        )
    

    def close(self):
        """Cleanup resources"""
        if hasattr(self.client, 'aclose'):
            asyncio.create_task(self.client.aclose())

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

sms_gateway = SMSHandler()

register_cron_routes(app)

@app.route("/sms", methods=['POST'])
def sms_webhook():
    """Twilio SMS webhook handler"""
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')
    
    resp = MessagingResponse()
    
    if not incoming_msg:
        resp.message("Konsultanim ready! Ask me about rice, corn, or coconut diseases, weather, or crop insurance.")
        return str(resp)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                sms_gateway.process_farmer_sms(incoming_msg, from_number)
            )
        finally:
            loop.close()
        
        resp.message(response)
        
    except Exception as e:
        app.logger.error(f"SMS processing error: {e}")
        resp.message("Konsultanim: I'm having technical difficulties. Please try again later.")
    
    return str(resp)
