import hashlib
import json
import torch
import importlib
import os
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise ValueError(f"Failed to load JSONL file: {e}")
    return data

def prepare_dataset(jsonl_data):
    prompts = []
    responses = []
    for item in jsonl_data:
        try:
            prompt = item['prompt'][0] if isinstance(item['prompt'], list) else item['prompt']
            response = item['response'][0] if isinstance(item['response'], list) else item['response']
            prompts.append(str(prompt))
            responses.append(str(response))
        except (KeyError, TypeError) as e:
            print(f"Skipping invalid entry: {item}, error: {e}")
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
                print(f"Loading tokenized dataset from {cache_dir}")
                return load_from_disk(cache_dir)
            else:
                print(f"Cache at {cache_dir} is outdated, removing it...")
                import shutil
                shutil.rmtree(cache_dir)
        except Exception as e:
            print(f"Error loading cache metadata: {e}, removing cache...")
            import shutil
            shutil.rmtree(cache_dir)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Pad to max_length
            max_length=max_length,
            return_tensors=None,  # Return lists, not tensors (let collator handle tensor conversion)
        )
        tokenized["labels"] = tokenized["input_ids"].copy()  # Copy input_ids to labels
        return tokenized
    
    print(f"Tokenizing dataset (this may take ~65 minutes for 1.78M examples)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=1,  # Avoid multiprocessing issues on Windows
    )
    
    print(f"Saving tokenized dataset to {cache_dir}")
    tokenized_dataset.save_to_disk(cache_dir)
    with open(cache_metadata_path, 'w') as f:
        json.dump({"cache_key": cache_key}, f)

    return tokenized_dataset

def fine_tuning(model_path: str, jsonl_path: str, output_path="./fine_tuned_model"):
    batch_size = 2
    max_length = 512
    lora_rank = 8
    lora_alpha = 32
    lora_dropout = 0.1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    try:
        bnb_version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError("bitsandbytes is not installed. Install it with: pip install bitsandbytes")

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize BitsAndBytesConfig: {e}. Ensure bitsandbytes is compiled with CUDA support.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True
        )
        model.config.use_cache = False
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    try:
        jsonl_data = load_jsonl(jsonl_path)
        dataset = prepare_dataset(jsonl_data)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)

        print(tokenized_dataset[0]) 
        print(tokenized_dataset.column_names)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return

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

    print("Trainable parameters:")
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            print(f"{name}: {param.shape}")
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check LoRA configuration.")
    
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Windows
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=None,
    )

    # class DebugTrainer(Trainer):
    #         def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #             print("Inputs to compute_loss:", inputs.keys())
    #             for key, value in inputs.items():
    #                 print(f"{key} shape: {value.shape}, requires_grad: {value.requires_grad}")
                
    #             outputs = model(**inputs)
    #             loss = outputs.loss
    #             print(f"Loss: {loss.item()}, requires_grad: {loss.requires_grad}")
                
    #             return (loss, outputs) if return_outputs else loss

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except RuntimeError as e:
        print(f"Training failed: {e}")
        if "CUDA out of memory" in str(e):
            print("Try reducing batch_size or increasing gradient_accumulation_steps.")
        return

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    fine_tuning(
        model_path = "C:/Users/nu1ls/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        jsonl_path = "./security_dataset.jsonl",
    )