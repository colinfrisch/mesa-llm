import os
from collections import deque
from dataclasses import dataclass

from mesa.agent import Agent

from mesa_llm.module_llm import ModuleLLM


@dataclass
class MemoryEntry:
    type: str
    content: str
    step: int
    metadata: dict


class Memory:
    """
    Create a memory object that stores the agent's short and long term memory

    Attributes:
        agent : the agent that the memory belongs to

    Memory is composed of
        - A short term memory who stores the n (int) most recent interactions (observations, planning, discussions)
        - A long term memory that is a summary of the memories that are removed from short term memory (summary
        completes itself as it goes)

    """

    def __init__(
        self,
        agent: Agent,
        short_term_capacity: int = 5,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        llm_model: str | None = "openai/gpt-4o-mini",
    ):
        """
        Initialize the memory

        Args:
            short_term_capacity : the number of interactions to store in the short term memory
            api_key : the API key for the LLM
            llm_model : the model to use for the summarization
            agent : the agent that the memory belongs to
        """
        self.agent = agent
        self.llm = ModuleLLM(api_key=api_key, model=llm_model)
        self.capacity = short_term_capacity
        self.short_term_memory = deque(maxlen=self.capacity)
        self.long_term_memory = ""

    def add_to_memory(
        self, type: str, content: str, step: int, metadata: dict | None = None
    ):
        """
        Add a new entry to the memory
        """
        new_entry = MemoryEntry(type, content, step, metadata)
        self.short_term_memory.append(new_entry)

    def get_short_term_memory(self) -> list[MemoryEntry]:
        """
        Get the short term memory
        """
        return list(self.short_term_memory)

    def update_long_term_memory(self):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """

        short_term_memory_dict = [
            self.convert_entry_to_dict(entry) for entry in self.short_term_memory
        ]

        prompt = f"""
            Short term memory: {short_term_memory_dict}
            Long term memory: {self.long_term_memory}
            """

        system_prompt = """
        You are a helpful assistant that summarizes the short term memory into a long term memory.
        The long term memory should be a summary of the short term memory that is concise and informative.
        If the short term memory is empty, return the long term memory unchanged.
        If the long term memory is not empty, update it to include the new information from the short term memory.
        """

        self.long_term_memory = self.llm.generate(
            self.short_term_memory, self.long_term_memory, prompt, system_prompt
        )

    def convert_entry_to_dict(self, entry: MemoryEntry) -> dict:
        """
        Convert a memory entry to a dictionary
        """
        return {
            "type": entry.type,
            "content": entry.content,
            "step": entry.step,
            "metadata": entry.metadata,
        }
