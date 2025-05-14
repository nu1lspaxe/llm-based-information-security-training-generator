from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import os
import logging
from pathlib import Path
from peft import PeftModel, PeftConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AttackSimulator:
    def __init__(
        self,
        base_model_path: str,
        fine_tuned_path: str,
        hf_token: Optional[str] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure memory settings
        if self.device == "cuda":
            # Use 80% of available GPU memory
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()
        
        # Load base model and tokenizer
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Configure model loading with basic memory optimization
            model_kwargs = {
                "token": hf_token,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "use_cache": False,  # Disable KV cache to reduce memory usage
                "use_flash_attention_2": False,  # Disable flash attention
                "use_sdpa": False  # Disable scaled dot product attention
            }
            
            logger.info("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs
            )
            
            # Move model to device
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            # Try to load fine-tuned model if available
            fine_tuned_path = Path(fine_tuned_path)
            if fine_tuned_path.exists():
                try:
                    # Check if it's a LoRA adapter
                    if (fine_tuned_path / "adapter_config.json").exists():
                        logger.info(f"Loading LoRA adapter from {fine_tuned_path}")
                        # Load adapter config first
                        adapter_config = PeftConfig.from_pretrained(fine_tuned_path)
                        # Load adapter
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            fine_tuned_path,
                            token=hf_token,
                            device_map={"": self.device} if self.device == "cuda" else None,
                            is_trainable=False,
                            config=adapter_config
                        )
                        logger.info("Successfully loaded LoRA adapter")
                    else:
                        # Try loading as a regular model
                        model_files = list(fine_tuned_path.glob("*.bin"))
                        if model_files:
                            logger.info("Loading fine-tuned model weights...")
                            try:
                                state_dict = torch.load(
                                    str(model_files[0]),
                                    map_location=self.device
                                )
                                if isinstance(state_dict, dict):
                                    self.model.load_state_dict(state_dict, strict=False)
                                    logger.info("Successfully loaded fine-tuned model")
                                else:
                                    logger.warning("Invalid state dict format. Using base model only.")
                            except PermissionError:
                                logger.warning(f"Permission denied when loading model from {model_files[0]}. Trying to fix permissions...")
                                try:
                                    # Try to fix permissions
                                    os.chmod(str(model_files[0]), 0o666)
                                    state_dict = torch.load(
                                        str(model_files[0]),
                                        map_location=self.device
                                    )
                                    if isinstance(state_dict, dict):
                                        self.model.load_state_dict(state_dict, strict=False)
                                        logger.info("Successfully loaded fine-tuned model after fixing permissions")
                                    else:
                                        logger.warning("Invalid state dict format. Using base model only.")
                                except Exception as e:
                                    logger.warning(f"Could not fix permissions: {str(e)}. Using base model only.")
                        else:
                            logger.warning(f"No model files found in {fine_tuned_path}")
                except Exception as e:
                    logger.warning(f"Could not load fine-tuned model: {str(e)}. Using base model only.")
            else:
                logger.info("Fine-tuned model not found. Using base model only.")
            
            self.model.eval()
            logger.info("Model initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def generate_attack_scenario(self, context: str) -> str:
        prompt = f"""Generate a detailed attack scenario based on the following context:
        Context: {context}
        
        Attack Scenario:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Attack Scenario:")[-1].strip()
