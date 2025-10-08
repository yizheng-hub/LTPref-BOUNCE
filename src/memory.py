import os
import json
from datetime import datetime
import logging

memory_logger = logging.getLogger(__name__)
memory_logger.setLevel(logging.ERROR)
if not memory_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[AgentMemory %(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    memory_logger.addHandler(handler)

class AgentMemory:
    def __init__(self, config: dict, base_folder: str = 'intermediate_files'):
        """
        Initializes the AgentMemory with the given configuration and memory folder.

        Args:
            config (dict): Configuration dictionary for the agent, containing 'agent_id'.
            base_folder (str): The base directory to store agent memories.
        """
        self.agent_id = config['agent_id']
        self.base_folder = base_folder
        self.memory_folder = os.path.join(base_folder, self.agent_id)
        self.file_path = os.path.join(self.memory_folder, 'memory.json')
        self.memory_data = self._initialize_memory()

    def _initialize_memory(self) -> dict:
        """
        Initializes the memory folder and 'memory.json' file if they don't exist.
        Loads existing memory if the file is found.

        Returns:
            dict: The loaded or newly initialized memory data.
        """
        os.makedirs(self.memory_folder, exist_ok=True)

        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        
        # If file doesn't exist, create a new memory structure
        memory_data = {
            'agent_id': self.agent_id,
            'memory': [],
            'metadata': {
               'capacity': 1000,  # Maximum number of memories to store by default
            }
        }

        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(memory_data, file, indent=4, ensure_ascii=False) # ensure_ascii=False for Chinese chars
        return memory_data
    
    def save_memory(self):
        """
        Saves the current memory data to 'memory.json'.
        """
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.memory_data, file, indent=4, ensure_ascii=False)

    def add_memory(self, data: dict):
        """
        Adds a new memory entry to the memory data.

        Args:
            data (dict): The actual memory data (e.g., user input and response).
        """
        new_memory = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self.memory_data['memory'].append(new_memory)

        if len(self.memory_data['memory']) > self.memory_data['metadata'].get('capacity', 100):
            self.memory_data['memory'].pop(0) # Remove the oldest memory entry if capacity exceeded
        
        self.save_memory()

    def query_memory(self, keyword: str) -> list[dict]:
        """
        Queries the memory data for entries containing the specified keyword in their data.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            list[dict]: A list of memory entries containing the keyword.
        """
        return [memory for memory in self.memory_data['memory'] if keyword in str(memory['data'])] # Convert data to string for broad search

    def get_recent_memory(self, n: int = 5) -> list[dict]:
        """
        Returns the most recent n memory entries.

        Args:
            n (int): The number of recent entries to retrieve.

        Returns:
            list[dict]: A list of recent memory entries.
        """
        return self.memory_data['memory'][-n:]

    def get_all_memory(self) -> list[dict]:
        """
        Returns all memory entries.

        Returns:
            list[dict]: A list of all memory entries.
        """
        return self.memory_data['memory']

    def clear_memory(self):
        """
        Clears all memory entries and saves the empty memory.
        """
        self.memory_data['memory'] = []
        self.save_memory()
