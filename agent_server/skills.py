'''
This module defines the skills that can be progressively disclosed to the agent. 
Each skill has a name, a brief description for the system prompt, and detailed content with instructions for when and how to use the skill. 
The agent can choose to load a skill's full content into its context when it determines that the skill is relevant to the user's query.
'''

from typing import TypedDict, Callable, Awaitable
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langchain_core.tools import tool

class Skill(TypedDict):
    """A skill that can be progressively disclosed to the agent."""
    name: str  # Unique identifier for the skill
    description: str  # 1-2 sentence description to show in system prompt
    content: str  # Full skill content with detailed instructions

SKILLS: list[Skill] = [
    {
        "name": "azure-pricing",
        "description": "Fetches real-time Azure retail pricing using the Azure Retail Prices API (prices.azure.com).",
        "content": """# Azure Pricing Skill

Use this skill to retrieve real-time Azure retail pricing data from the public Azure Retail Prices API. No authentication is required.

## When to Use This Skill

- User asks about the cost of an Azure service (e.g., "How much does a D4s v5 VM cost?")
- User wants to compare pricing across regions or SKUs
- User needs a cost estimate for a workload or architecture
- User mentions Azure pricing, Azure costs, or Azure billing
- User asks about reserved instance vs. pay-as-you-go pricing
- User wants to know about savings plans or spot pricing

## API Endpoint

```
GET https://prices.azure.com/api/retail/prices?api-version=2023-01-01-preview
```

Append `$filter` as a query parameter using OData filter syntax. Always use `api-version=2023-01-01-preview` to ensure savings plan data is included.""",
    },
]

@tool
def load_local_skill(skill_name: str) -> str:
    """Load the full content of a skill into the agent's context.

    Use this when you need detailed information about how to handle a specific
    type of request. This will provide you with comprehensive instructions,
    policies, and guidelines for the skill area.

    Args:
        skill_name: The name of the skill to load (e.g., "expense_reporting", "travel_booking")
    """
    # Find and return the requested skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"Loaded skill: {skill_name}\n\n{skill['content']}"

    # Skill not found
    available = ", ".join(s["name"] for s in SKILLS)
    return f"Skill '{skill_name}' not found. Available skills: {available}"

class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    # Register the load_skill tool as a class variable
    tools = [load_local_skill]

    def __init__(self):
        """Initialize and generate the skills prompt from SKILLS."""
        # Build skills prompt from the SKILLS list
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)
    
    # Override the model call to inject the skills prompt into the system message
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        # Append to system message content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)
    
    # Async version of the model call wrapper
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        # Append to system message content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return await handler(modified_request)