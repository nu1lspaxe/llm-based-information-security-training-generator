import os
os.environ["TRITON_DISABLED"] = "1"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

class ResponseSimulator:
    def __init__(
        self,
        base_model_path: str,
        fine_tuned_path: str,
        hf_token: Optional[str] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use mirror site
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            token=hf_token,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            use_flash_attention_2=False,  # Disable flash-attention
            attn_implementation="eager"  # Use eager mode
        )
        
        # Load fine-tuned model
        if os.path.exists(fine_tuned_path):
            try:
                # Load state dict with safe loading parameters
                state_dict = torch.load(
                    fine_tuned_path,
                    map_location=self.device,
                    weights_only=True
                )
                if isinstance(state_dict, dict):
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    print("Warning: Invalid state dict format. Continuing with base model only.")
            except Exception as e:
                print(f"Warning: Could not load fine-tuned model: {str(e)}")
                print("Continuing with base model only.")
        
        self.model.eval()
    
    def generate_response_plan(self, context: str) -> str:
        prompt = f"""Generate a detailed incident response plan based on the following context:
        Context: {context}
        
        Response Plan:"""
        
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
        return response.split("Response Plan:")[-1].strip()