import os

from databricks.sdk import WorkspaceClient
from langchain_core.tools import tool

sp_workspace_client = WorkspaceClient()


@tool(description="Show all tables in Unity Catalog. Catalog & Schema: basket_intelligence.default")
def show_tables_from_uc():
    '''
    This tool retrieves all tables available in the specified catalog and schema.
    '''

    # Query the table using a SQL warehouse resource
    result = sp_workspace_client.statement_execution.execute_statement(
        warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),  # Requires a SQL warehouse resource,
        catalog="basket_intelligence",
        schema="default",
        statement="SHOW TABLES",
    )

    return result


@tool(description="Get data from Unity Catalog. Catalog & Schema: basket_intelligence.default")
def get_basketball_stats_from_uc(query: str):
    '''
    Get basketball stats from Unity Catalog.
    '''

    # Query the table using a SQL warehouse resource
    result = sp_workspace_client.statement_execution.execute_statement(
        warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),  # Requires a SQL warehouse resource,
        catalog="basket_intelligence",
        schema="default",
        statement=query,
    )

    return result


uc_agent_prompt = (
    "You are a data analyst. "
    "Use the unity catalog tool to help answer user queries."
)


uc_basketball_subagent = {
    "name": "unity-catalog-agent",
    "description": "Used to perform analytic tasks with basketball data from unity catalog",
    "system_prompt": uc_agent_prompt,
    "tools": [show_tables_from_uc, get_basketball_stats_from_uc],
    #"model": "openai:gpt-5.4",  # Optional override, defaults to main agent model
}
