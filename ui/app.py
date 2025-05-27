import os
os.environ["TRITON_DISABLED"] = "1"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"  # Disable flash-attention
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting
os.environ["LITELLM_PROVIDER"] = "huggingface"  # Set default provider
os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_CACHE"] = "true"  # Enable caching
os.environ["LITELLM_MAX_RETRIES"] = "3"  # Set max retries
os.environ["LITELLM_TIMEOUT"] = "60"  # Set timeout in seconds

import streamlit as st
from agents import CISOAgent, SecurityAnalystAgent, IncidentResponderAgent
from rag.graph_cyrag import GraphCyRAG
import pandas as pd
from pathlib import Path
import torch
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

async def init_models():
    load_dotenv()

    """Initialize models and store them in session state"""
    if 'models_initialized' not in st.session_state:
        try:
            # Basic configuration
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN environment variable is not set")
            
            base_model_path = os.getenv("HF_MODEL_PATH")
            if not base_model_path:
                raise ValueError("HF_MODEL_PATH environment variable is not set")
            
            fine_tuned_path = Path("./fine_tuned_model").absolute()
            
            # Verify model directories
            response_path = fine_tuned_path / "response"
            
            if not all(path.exists() for path in [fine_tuned_path, response_path]):
                raise FileNotFoundError(
                    f"Model directories not found. Please ensure the following paths exist:\n"
                    f"- {fine_tuned_path}\n"
                    f"- {response_path}"
                )
            
            # Initialize models one by one with proper error handling
            models = {}
            
            # Load base model and tokenizer with memory-optimized configuration
            try:
                logger.info("Loading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_path,
                    token=hf_token,
                    trust_remote_code=True
                )
                
                # Memory-optimized model loading configuration
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                # Simplified device map - keep everything in memory
                device_map = "auto" if device == "cuda" else "cpu"

                max_memory = None
                if device == "cuda":
                    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                    target_memory_bytes = int(gpu_memory_bytes * 0.8)  # 80% of total memory
                    target_memory_gb = target_memory_bytes / (1024 ** 3)  # Convert to GB
                    max_memory = {0: f"{target_memory_gb:.1f}GB"}  # Format as string (e.g., "8.0GB")
                    logger.info(f"Setting max_memory to {max_memory[0]} for GPU device 0")
                
                model_kwargs = {
                    "token": hf_token,
                    "trust_remote_code": True,
                    "device_map": device_map,
                    "quantization_config": quantization_config if device == "cuda" else None,
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                    "low_cpu_mem_usage": True,
                    "use_cache": False,
                    "use_flash_attention_2": False,
                    "attn_implementation": "eager",
                    "max_memory": max_memory
                }
                
                logger.info("Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    **model_kwargs
                )
                
                # Try to load fine-tuned model if available
                if response_path.exists():
                    try:
                        # Check if it's a LoRA adapter
                        if (response_path / "adapter_config.json").exists():
                            logger.info(f"Loading LoRA adapter from {response_path}")
                            adapter_config = PeftConfig.from_pretrained(response_path)
                            base_model = PeftModel.from_pretrained(
                                base_model,
                                response_path,
                                token=hf_token,
                                is_trainable=False,
                                config=adapter_config
                            )
                            logger.info("Successfully loaded LoRA adapter")
                    except Exception as e:
                        logger.warning(f"Could not load fine-tuned model: {str(e)}. Using base model only.")
                else:
                    logger.info("Fine-tuned model not found. Using base model only.")
                
                base_model.eval()
                models['base_model'] = base_model
                models['tokenizer'] = tokenizer
                
            except Exception as e:
                raise RuntimeError(f"Failed to initialize base model: {str(e)}")
            
            # GraphCyRAG with simplified configuration
            try:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD")

                if not neo4j_password:
                    raise ValueError("NEO4J_PASSWORD environment variable is not set")
                
                models['graph_rag'] = GraphCyRAG(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    llm=models['base_model']
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize GraphCyRAG: {str(e)}")
            
            # Initialize agents with simplified configuration
            try:
                # Configure CrewAI with HuggingFace provider
                os.environ["CREWAI_PROVIDER"] = "huggingface"
                os.environ["CREWAI_MODEL"] = base_model_path
                os.environ["CREWAI_TOKEN"] = hf_token
                
                models['ciso'] = CISOAgent(models['base_model'], models['graph_rag'])
                models['analyst'] = SecurityAnalystAgent(models['base_model'], models['graph_rag'])
                models['responder'] = IncidentResponderAgent(models['base_model'], models['graph_rag'])
            except Exception as e:
                raise RuntimeError(f"Failed to initialize agents: {str(e)}")
            
            # Store all models in session state
            for key, value in models.items():
                st.session_state[key] = value
            
            st.session_state.models_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.info("""
            Please ensure:
            1. All required environment variables are set:
               - HF_TOKEN
               - NEO4J_URI (optional, defaults to bolt://localhost:7687)
               - NEO4J_USER (optional, defaults to neo4j)
               - NEO4J_PASSWORD
            
            2. Model directories exist:
               - ./fine_tuned_model/
               - ./fine_tuned_model/response/
            
            3. You have proper permissions and internet connection
            
            You can set the environment variables using:
            ```bash
            set HF_TOKEN="your_token"
            set NEO4J_URI="bolt://localhost:7687"
            set NEO4J_USER="neo4j"
            set NEO4J_PASSWORD="your_password"
            ```
            """)
            return False
    return True

async def main():
    st.set_page_config(
        page_title="Cybersecurity Training Scenario Generator",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("Cybersecurity Training Scenario Generator")
    
    with st.spinner("Loading models..."):
        if not await init_models():
            return
    
    st.subheader("Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        threat_type = st.text_input("Enter Threat Type (e.g., SQL Injection)", "SQL Injection")
    with col2:
        cwe_id = st.text_input("Enter CWE ID (e.g., CWE-89)", "CWE-89")
    
    if st.button("Generate Training Scenario"):
        with st.spinner("Generating scenario..."):
            try:
                threat_paths = await st.session_state.graph_rag.fetch_threat_path(cwe_id)
                attack_context = await st.session_state.graph_rag.get_attack_context(threat_type)
                
                scenario = {
                    "Threat Type": threat_type,
                    "CWE ID": cwe_id,
                    "Attack Scenario": await st.session_state.analyst.process(attack_context),
                    "Strategic Recommendations": await st.session_state.ciso.process(attack_context),
                    "Response Steps": await st.session_state.responder.process(attack_context)
                }
                
                st.subheader("Generated Training Scenario")
                for key, value in scenario.items():
                    st.write(f"**{key}**:")
                    st.write(value)
                    st.write("---")
                
                df = pd.DataFrame([scenario])
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Scenario",
                    csv,
                    "scenario.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Error generating scenario: {str(e)}")
    
    if 'threat_paths' in locals():
        st.subheader("Threat Path Visualization")
        st.write("CVE to ATT&CK Mapping:")
        for path in threat_paths:
            st.write(path)
    
    if 'graph_rag' in st.session_state:
        await st.session_state.graph_rag.close()

if __name__ == "__main__":
    asyncio.run(main())
