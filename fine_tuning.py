import hashlib
import json
import torch
import importlib
import os
import logging
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from huggingface_hub import scan_cache_dir
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Triton
os.environ["TRITON_DISABLED"] = "1"

load_dotenv()

def check_model_cache(model_path):
    cache_info = scan_cache_dir()
    model_id = model_path
    cache_status = {
        "model_path": model_path,
        "model_id": model_id,
        "found_in_cache": False,
        "cache_path": None,
        "size_on_disk": None,
        "last_modified": None
    }
    logger.info(f"Checking cache for model: {model_path}")
    logger.info(f"Looking for model ID: {model_id}")
    logger.info(f"HF_HOME: {os.getenv('HF_HOME', 'Not set (using default)')}")
    logger.info(f"Default cache path: {os.path.expanduser('~/.cache/huggingface/hub')}")
    if not cache_info.repos:
        logger.info("No models found in cache")
    else:
        for repo in cache_info.repos:
            logger.info(f"Model: {repo.repo_id}")
            logger.info(f"  Path: {repo.repo_path}")
            logger.info(f"  Size: {repo.size_on_disk / (1024*1024*1024):.2f} GB")
            logger.info(f"  Last modified: {repo.last_modified}")
    for repo in cache_info.repos:
        if repo.repo_id == model_id:
            cache_status.update({
                "found_in_cache": True,
                "cache_path": repo.repo_path,
                "size_on_disk": repo.size_on_disk,
                "last_modified": repo.last_modified
            })
            logger.info(f"✅ Model found in cache: {repo.repo_path}")
            return cache_status
    logger.info("❌ Model not found in cache. Will download from Hugging Face Hub")
    return cache_status

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise ValueError(f"Failed to load JSONL file: {e}")
    return data

def prepare_dataset(jsonl_data, role_type):
    prompts = []
    responses = []
    for item in jsonl_data:
        try:
            prompt = item['prompt'][0] if isinstance(item['prompt'], list) else item['prompt']
            response = item['response'][0] if isinstance(item['response'], list) else item['response']
            if role_type == "attack":
                prompt = f"Generate an attack scenario based on: {prompt}"
            elif role_type == "response":
                prompt = f"Generate a response strategy for: {prompt}"
            elif role_type == "ciso":
                prompt = f"Provide strategic recommendations for: {prompt}"
            elif role_type == "analyst":
                prompt = f"Analyze the security implications of: {prompt}"
            elif role_type == "responder":
                prompt = f"Create an incident response plan for: {prompt}"
            prompts.append(str(prompt))
            responses.append(str(response))
        except (KeyError, TypeError) as e:
            logger.warning(f"Skipping invalid entry: {item}, error: {e}")
            continue
    if not prompts:
        raise ValueError("No valid data entries found in JSONL file.")
    combined_texts = [f"{p}\nResponse: {r}" for p, r in zip(prompts, responses)]
    return Dataset.from_dict({"text": combined_texts})

def tokenize_dataset(dataset, tokenizer, max_length=512, cache_dir="./tokenized_dataset"):
    def get_cache_key():
        params = {
            "max_length": max_length,
            "tokenizer_name": tokenizer.name_or_path,
            "dataset_size": len(dataset),
        }
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    cache_key = get_cache_key()
    cache_metadata_path = os.path.join(cache_dir, "cache_metadata.json")

    if os.path.exists(cache_dir) and os.path.exists(cache_metadata_path):
        try:
            with open(cache_metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get("cache_key") == cache_key:
                logger.info(f"Loading tokenized dataset from {cache_dir}")
                return load_from_disk(cache_dir)
            else:
                logger.info(f"Cache at {cache_dir} is outdated, removing it...")
                import shutil
                shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}, removing cache...")
            import shutil
            shutil.rmtree(cache_dir)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=1
    )
    
    logger.info(f"Saving tokenized dataset to {cache_dir}")
    tokenized_dataset.save_to_disk(cache_dir)
    with open(cache_metadata_path, 'w') as f:
        json.dump({"cache_key": cache_key}, f)
    return tokenized_dataset

def fine_tuning(model_path: str, jsonl_path: str, output_path="./fine_tuned_model", role_type="attack"):
    batch_size = 1
    max_length = 128
    lora_rank = 2
    lora_alpha = 8
    lora_dropout = 0.05

    logger.info("=== GPU Information ===")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info("=======================")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    if device == "cuda:0":
        try:
            bnb_version = importlib.metadata.version("bitsandbytes")
            logger.info(f"bitsandbytes version: {bnb_version}")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError("bitsandbytes is not installed. Install it with: pip install bitsandbytes")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BitsAndBytesConfig: {e}")

    cache_status = check_model_cache(model_path)

    try:
        logger.info(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True
        )
        logger.info(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=os.getenv("HF_TOKEN"),
            quantization_config=quantization_config if device == "cuda:0" else None,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_flash_attention_2=False,
            attn_implementation="eager",
            use_cache=False,
        )
        model.config.use_cache = False
        if device == "cuda:0":
            model = model.to(device)
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Troubleshooting steps:")
        logger.info("1. Check internet connection")
        logger.info("2. Verify Hugging Face token")
        logger.info("3. Ensure model path is correct")
        logger.info("4. Try mirror site: https://hf-mirror.com")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    try:
        jsonl_data = load_jsonl(jsonl_path)
        jsonl_data = jsonl_data[:100]
        dataset = prepare_dataset(jsonl_data, role_type)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)
        if "labels" not in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.map(
                lambda x: {"labels": x["input_ids"]},
                batched=True,
                remove_columns=["input_ids", "attention_mask"]
            )
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check LoRA configuration.")
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=os.path.join(output_path, role_type),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        fp16=True if device == "cuda:0" else False,
        logging_steps=1,
        save_steps=25,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_names=["labels"],
        max_grad_norm=0.3,
        warmup_steps=10,
        no_cuda=device != "cuda:0",
        use_cpu=device == "cpu"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    try:
        trainer.train()
        if device == "cuda:0":
            logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"Current GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    except RuntimeError as e:
        logger.error(f"Training failed: {e}")
        if "CUDA out of memory" in str(e):
            logger.info("Try reducing batch_size or increasing gradient_accumulation_steps.")
        raise

    model.save_pretrained(os.path.join(output_path, role_type))
    tokenizer.save_pretrained(os.path.join(output_path, role_type))
    logger.info(f"Model saved to {os.path.join(output_path, role_type)}")

if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        logger.error("HF_TOKEN not set")
        logger.info("Set it using: set HF_TOKEN='your_token'")
        exit(1)
    
    roles = ["attack", "response", "ciso", "analyst", "responder"]
    
    model_path = os.getenv("HF_MODEL_PATH")
    if not model_path:
        logger.error("HF_MODEL_PATH not set")
        logger.info("Set it using: set HF_MODEL_PATH='your_model_path'")
        exit(1)

    jsonl_path = "./security_dataset.jsonl"
    for role in roles:
        logger.info(f"Training model for role: {role}")
        fine_tuning(
            model_path=model_path,
            jsonl_path=jsonl_path,
            role_type=role
        )