"""Agent executor with Alibaba Cloud integration - intelligent routing."""
import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    TaskState,
    Part,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from .crop_doctor_agent import CropDoctorAgent
from .advisory_agent import AdvisoryAgent
from .insurance_agent import InsuranceAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KonsultanimAgentExecutor(AgentExecutor):
    """Konsultanim multi-agent executor with Alibaba Cloud services."""

    def __init__(self):
        self.crop_doctor = CropDoctorAgent()
        self.advisory_agent = AdvisoryAgent()
        self.insurance_agent = InsuranceAgent()
        self.current_task_id = None

    def _route_query(self, query: str) -> str:
        """Determine which agent to use based on query content."""
        query_lower = query.lower()
        
        # Insurance keywords
        insurance_keywords = [
            'insurance', 'policy', 'claim', 'premium', 'coverage', 
            'pcic', 'enroll', 'indemnity', 'subsidy'
        ]
        if any(keyword in query_lower for keyword in insurance_keywords):
            return 'insurance'
        
        # Advisory keywords
        advisory_keywords = [
            'weather', 'forecast', 'advisory', 'weekly', 'climate',
            'when to plant', 'planting schedule'
        ]
        if any(keyword in query_lower for keyword in advisory_keywords):
            return 'advisory'
        
        # Default to crop doctor for disease/pest/diagnosis queries
        return 'crop_doctor'

    def _validate_request(self, context: RequestContext):
        """Validate incoming request."""
        if not context.message:
            return InvalidParamsError(message='No message provided')
        if not context.message.parts:
            return InvalidParamsError(message='Message has no parts')
        return None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        self.current_task_id = task.id
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Route to appropriate agent
            agent_type = self._route_query(query)
            logger.info(f"[Konsultanim] Routing to {agent_type} agent")
            
            if agent_type == 'insurance':
                agent = self.insurance_agent
                logger.info("[Konsultanim] Using Insurance Agent (Qwen + OSS)")
            elif agent_type == 'advisory':
                agent = self.advisory_agent
                logger.info("[Konsultanim] Using Advisory Agent (Qwen + OSS)")
            else:
                agent = self.crop_doctor
                logger.info("[Konsultanim] Using Crop Doctor Agent (Qwen + OSS)")
            
            # Stream response
            async for item in agent.stream(query, task.context_id):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='result',
                    )
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(
                            'Task completed successfully.',
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break

        except Exception as e:
            logger.error(f"Execution error: {e}")
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(
                    f'Error processing request: {str(e)}',
                    task.context_id,
                    task.id,
                ),
                final=True,
            )
            raise ServerError(error=InternalError(message=str(e)))
        finally:
            self.current_task_id = None

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Cancel the current task execution.
        
        This method is required by the AgentExecutor abstract base class.
        """
        if self.current_task_id:
            logger.info(f"[Konsultanim] Cancelling task: {self.current_task_id}")
            updater = TaskUpdater(event_queue, self.current_task_id, context.context_id)
            await updater.update_status(
                TaskState.cancelled,
                new_agent_text_message(
                    'Task cancelled by user.',
                    context.context_id,
                    self.current_task_id,
                ),
                final=True,
            )
            self.current_task_id = None
        else:
            logger.warning("[Konsultanim] No active task to cancel")
