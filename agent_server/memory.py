'''
This module sets up an in-memory store for the agent's memory.
 It initializes the store with some seed data that mimics the content of memory files (like AGENTS.md). 
 The `StateBackend` is also initialized to manage the state of the agent.
'''
from langgraph.store.memory import InMemoryStore

def setup_ram_memory():

    store = InMemoryStore()

    # Populate the store with initial memories (seed data)
    # This mimics the content of your memory files (e.g., AGENTS.md)
    store.put(
        namespace=("assistant_id",), 
        key="AGENTS.md", 
        value={"content": "The agent is helpful and prefers concise answers."}
    )
    return store