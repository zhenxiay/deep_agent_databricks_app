'''
This file defines the toolkit of tools that the Databricks Agent Server can use to perform various tasks. 
'''
import subprocess
from langchain_core.tools import tool

@tool
def bash(command: str):
    """Execute a bash command and return the output.

    Args:
        command (str): The bash command to execute.
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip() or "(empty)"