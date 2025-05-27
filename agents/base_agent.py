from abc import ABC, abstractmethod
from crewai import Agent, LLM
import os
from dotenv import load_dotenv
import logging
import asyncio
from litellm import completion
import litellm
import requests
import json

# Enable litellm debug mode
litellm._turn_on_debug()

logger = logging.getLogger(__name__)

class BaseSecurityAgent(ABC):
    def __init__(self, llm, graph_rag):
        # Load environment variables
        load_dotenv()
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")
        
        hf_model_path = os.getenv("HF_MODEL_PATH")
        if not hf_model_path:
            raise ValueError("HF_MODEL_PATH environment variable is not set")
        
        # Get Neo4j credentials from environment
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable is not set")
        
        logger.info(f"Connecting to Neo4j at {neo4j_uri} as user {neo4j_user}")
        
        # Configure LiteLLM with HuggingFace
        self.llm = LLM(
            model=f"huggingface/{hf_model_path}",
            api_key=hf_token,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            do_sample=True
        )
        
        # Store HuggingFace token for direct API calls
        self.hf_token = hf_token
        
        # Use the provided graph_rag instance directly
        self.graph_rag = graph_rag
        
        self.agent = self._create_agent()
    
    async def _call_llm(self, prompt: str) -> str:
        """Helper method to call LLM with proper error handling"""
        try:
            # Format the prompt according to Llama 3.1's chat format
            formatted_prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant that provides detailed and accurate responses.
<</SYS>>

{prompt} [/INST]"""
            
            logger.info(f"Calling LLM with prompt: {formatted_prompt[:100]}...")
            
            # Use LiteLLM completion
            response = completion(
                model=f"huggingface/{os.getenv('HF_MODEL_PATH')}",
                messages=[{"role": "user", "content": formatted_prompt}],
                api_key=self.hf_token,
                temperature=0.7,
                max_tokens=512,
                top_p=0.9,
                do_sample=True
            )
            
            if not response or not response.choices:
                raise Exception("Empty response from LLM")
            
            generated_text = response.choices[0].message.content
            logger.info("Successfully received response from LLM")
            return generated_text
                
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            logger.error(f"Full error details: {e.__dict__}")
            return f"Error generating response: {str(e)}"
    
    @abstractmethod
    def _create_agent(self):
        pass
    
    @abstractmethod
    async def process(self, input_data):
        pass
