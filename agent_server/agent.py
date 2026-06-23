import logging
from typing import AsyncGenerator, Optional

import litellm
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from mlflow.genai.agent_server import invoke, stream
import mlflow.litellm
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.skills import SkillMiddleware
from agent_server.tools.toolkit import bash
from agent_server.agent_backend import (
    setup_filesystem_backend,
    load_remote_skills,
)
from agent_server.subagents.stock_agent import stock_subagent 
from agent_server.subagents.thai_agent import thai_recipe_subagent
from agent_server.subagents.uc_basketball_agent import uc_basketball_subagent
from agent_server.utils import (
    get_session_id,
    process_agent_astream_events,
)

################## Initialization ################################

#mlflow.langchain.autolog()
mlflow.litellm.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

################ Agents #################################

system_prompt = (
    "You are the leader of a team of agents. "
    "Your task is to understand user's request and distribute it to the correct agent."
)

# Define subagents for deep agent

subagents = [thai_recipe_subagent, stock_subagent, uc_basketball_subagent]

############## Load remote skills #################################
load_remote_skills()

############## Agent Initialization #################################

async def init_deep_agent(workspace_client: Optional[WorkspaceClient] = None):
    '''
    This function initializes a deep agent with multiple subagents, each having access to different tools and information. 
    The main agent is responsible for understanding the user's request and delegating tasks to the appropriate subagent based on their expertise. 
    The system prompt guides the main agent in its role as a leader of the subagents.
    '''

    return create_deep_agent(
        subagents=subagents,
        model=ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct"), 
        checkpointer=MemorySaver(),
        backend=setup_filesystem_backend(),
        middleware=[SkillMiddleware()],
        skills=["/skills/"],
        tools=[bash],
        system_prompt=system_prompt
        )

@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)

@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    # By default, uses service principal credentials.
    # For on-behalf-of user authentication, use get_user_workspace_client() instead:
    #   agent = await init_agent(workspace_client=get_user_workspace_client())
    agent = await init_deep_agent()
    messages = {"messages": to_chat_completions_input([i.model_dump() for i in request.input])}

    async for event in process_agent_astream_events(
        agent.astream(
            input=messages, 
            config={"configurable": {"thread_id": session_id}},
            stream_mode=["updates", "messages"])
    ):
        yield event
