# A LLM-based Information Security Training Generator

```bash
pip install huggingface_hub streamlit langchain langchain-ollama langchain-community chromadb requests ratelimit chardet scapy ijson peft datasets torch bitsandbytes

# Enable GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```bash
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./llama3-8b --local-dir-use-symlinks False
```