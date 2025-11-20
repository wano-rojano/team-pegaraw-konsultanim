"""Konsultanim A2A Server - Multi-crop agricultural AI assistant with Alibaba Cloud."""
import logging
import os
import sys

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from .agent_executor_alibaba import KonsultanimAgentExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='0.0.0.0')
@click.option('--port', 'port', default=int(os.environ.get('A2A_PORT', 10000)))
def main(host, port):
    """Start Konsultanim A2A server with Alibaba Cloud integration."""
    try:
        if not os.getenv('DASHSCOPE_API_KEY'):
            raise MissingAPIKeyError('DASHSCOPE_API_KEY environment variable not set.')
        
        if not os.getenv('OSS_ACCESS_KEY_ID') or not os.getenv('OSS_ACCESS_KEY_SECRET'):
            raise MissingAPIKeyError('OSS credentials not set.')

        logger.info(f"Starting Konsultanim server on {host}:{port}")

        capabilities = AgentCapabilities(
            streaming=True,
            push_notifications=True,
            text_input=True,
            file_upload=False,
            web_browsing=True,
            data_analysis=False
        )

        skills = [
            AgentSkill(
                id='crop_disease_diagnosis',
                name='Multi-Crop Disease Diagnosis (Alibaba Cloud)',
                description='Diagnose diseases and pests for rice, corn, and coconut crops using Alibaba Cloud OSS + Qwen',
                tags=['agriculture', 'pathology', 'diagnosis', 'alibaba-cloud'],
                examples=['My rice plants have brown spots on leaves, what disease is this?'],
            ),
            AgentSkill(
                id='crop_management',
                name='Integrated Pest Management',
                description='IPM strategies for rice, corn, and coconut',
                tags=['agronomy', 'ipm', 'management'],
                examples=['How do I manage coconut Cadang-Cadang disease?'],
            ),
            AgentSkill(
                id='farming_advisory',
                name='Weather-Based Farming Advisory',
                description='Weather forecasts and disease risk assessments',
                tags=['weather', 'advisory', 'climate'],
                examples=['Weekly weather advisory for rice in Laguna'],
            ),
            AgentSkill(
                id='crop_insurance',
                name='Crop Insurance Assistance',
                description='Guidance on crop insurance policies, claims, and enrollment',
                tags=['insurance', 'policy', 'PCIC'],
                examples=['How do I file a crop insurance claim for corn?'],
            ),
        ]

        agent_card = AgentCard(
            name='Konsultanim (Alibaba Cloud)',
            description=(
                'AI-powered agricultural assistant for Filipino farmers powered by Alibaba Cloud. '
                'Provides disease diagnosis, integrated pest management, '
                'weather-based advisories, and crop insurance guidance '
                'for rice, corn, and coconut crops. Evidence-based recommendations '
                'using scientific literature from OSS, real-time weather data, and insurance policy documents.'
            ),
            url=f'http://{host}:{port}/',
            version='2.0.0',
            defaultInputModes=['text'],
            defaultOutputModes=['text'],
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=KonsultanimAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'Server startup error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
