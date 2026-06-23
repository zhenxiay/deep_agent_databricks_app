import logging
from typing import Any, AsyncGenerator, AsyncIterator, Optional

from databricks.sdk import WorkspaceClient
from databricks_langchain.chat_models import json
from langchain.messages import AIMessageChunk, ToolMessage
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)


def get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def get_user_workspace_client() -> WorkspaceClient:
    token = get_request_headers().get("x-forwarded-access-token")
    return WorkspaceClient(token=token, auth_type="pat")


def get_databricks_host_from_env() -> Optional[str]:
    try:
        w = WorkspaceClient()
        return w.config.host
    except Exception as e:
        logging.exception(f"Error getting databricks host from env: {e}")
        return None


async def process_agent_astream_events(
    async_stream: AsyncIterator[Any],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Generic helper to process agent stream events and yield ResponsesAgentStreamEvent objects.
    """
    async for event in async_stream:
        # Check if the stream event tuple is valid
        if not isinstance(event, (list, tuple)) or len(event) < 2:
            continue

        event_type, event_value = event[0], event[1]

        if event_type == "updates":
            # Ensure event_value is a dictionary (e.g., {node_name: state_update})
            if not isinstance(event_value, dict):
                continue
                
            for node_data in event_value.values():
                # Guard: Ensure node_data is a dict before calling .get()
                if isinstance(node_data, dict):
                    messages = node_data.get("messages", [])
                    if isinstance(messages, list) and len(messages) > 0:
                        for msg in messages:
                            if isinstance(msg, ToolMessage) and not isinstance(msg.content, str):
                                msg.content = json.dumps(msg.content)
                        
                        for item in output_to_responses_items_stream(messages):
                            yield item

        elif event_type == "messages":
            # Ensure event_value is a list/tuple as expected by your logic
            if isinstance(event_value, (list, tuple)) and len(event_value) > 0:
                try:
                    chunk = event_value[0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **create_text_delta(delta=content, item_id=chunk.id)
                        )
                except Exception as e:
                    logging.exception(f"Error processing agent stream event: {e}")
