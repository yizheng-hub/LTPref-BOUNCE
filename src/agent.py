import json
import logging
from openai import OpenAI
from memory import AgentMemory

class Agent:
    def __init__(self, config_path: str, memory_folder: str = 'intermediate_files'):
        """
        Initializes the Agent with the given configuration path and memory folder.

        Args:
            config_path (str): Path to the agent's JSON configuration file.
            memory_folder (str): Folder to store agent memories.
        """
        self.config = self.load_config(config_path)
        self.agent_id = self.config['agent_id']
        self.model = self.config['model']
        self.model_type = self.config['training_setting']['model_type']
        self.api_key = self.config['training_setting']['api_key']
        # Default base_url can be adjusted if using a different OpenAI-compatible API
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.zzz-api.top/v1") 
        # self.tokenizer = None # Not used in this API-based setup, but kept for consistency
        # self.processor = None # Not used
        self.system_prompt = self.config['system_message']
        self.memory = AgentMemory(self.config, memory_folder)
        self.logger = self._init_logger()

    def load_config(self, config_path: str) -> dict:
        """
        Loads the configuration from the given JSON file path.

        Args:
            config_path (str): Path to the JSON configuration file.

        Returns:
            dict: The loaded configuration dictionary.
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def _init_logger(self) -> logging.Logger:
        """
        Initializes a basic logger for the agent.
        """
        logger = logging.getLogger(self.agent_id)
        logger.setLevel(logging.INFO) # Default to ERROR level
        # If no handlers are attached, add a console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f"[{self.agent_id} %(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def log_error(self, message: str):
        """
        Logs an error message for the agent.

        Args:
            message (str): The error message.
        """
        self.logger.error(message)

    def generate_response(self, user_input: str, few_shot_examples: list = None) -> str:
        """
        Generates a response using the configured LLM.

        Args:
            user_input (str): The user's input/query.
            few_shot_examples (list): A list of few-shot examples (dictionaries with 'role' and 'content').

        Returns:
            str: The generated response from the LLM.
        """
        messages = [{'role': 'system', 'content': self.system_prompt}]
        
        # Few-shot examples are typically structured as alternating user/assistant messages
        # Example format: [{'role': 'user', 'content': 'Example input'}, {'role': 'assistant', 'content': 'Example output'}]
        if few_shot_examples:
            messages.extend(few_shot_examples)

        messages.append({'role': 'user', 'content': user_input})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                timeout=self.config['training_setting'].get('timeout', 10), # Use .get() with default
                max_tokens=self.config['training_setting']['max_new_tokens'],
                temperature=self.config['training_setting']['temperature'],
                top_p=self.config['training_setting']['top_p']
            )
            
            if not response or not response.choices:
                self.log_error(f"API call returned no valid choices: {response}")
                return "Cannot generate response (no choices)."
            
            ai_response = response.choices[0].message.content
            self.memory.add_memory({'user': user_input, 'response': ai_response})
            return ai_response

        except Exception as e:
            self.log_error(f"API call failed: {str(e)}")
            return "Service temporarily unavailable."