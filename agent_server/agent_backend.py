'''
This module defines the deep agent backend for the Databricks Agent Server. 
'''
import os
from urllib.request import urlopen
from deepagents.backends import FilesystemBackend

def setup_filesystem_backend():
    '''
    Sets up the filesystem backend for the deep agent.
    '''
    AGENT_WORKSPACE = os.getcwd()

    #load_remote_skills(AGENT_WORKSPACE)

    backend = FilesystemBackend(
        root_dir=AGENT_WORKSPACE,
        virtual_mode=True
    )
    
    return backend

def load_remote_skills():
    """
    Downloads a remote skill and saves it directly to the local filesystem
    so the FilesystemBackend can provide it to the agent.
    """
    skill_url = "https://raw.githubusercontent.com/github/awesome-copilot/refs/heads/main/skills/az-cost-optimize/SKILL.md"

    # 1. Download the content
    with urlopen(skill_url) as response:
       skill_content = response.read().decode('utf-8')
   
    # 2. Define the local path relative to your agent's root directory
    # Note: We remove the leading slash from the key to make it a valid local path
    root_dir = os.getcwd()
    relative_path = "skills/az-cost-optimize/SKILL.md"
    full_path = os.path.join(root_dir, relative_path)

    # 3. Create the necessary folders on your disk
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    # 4. Write the content as a physical Markdown file
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(skill_content)
   
    print(f"Skill successfully integrated at: {full_path}")