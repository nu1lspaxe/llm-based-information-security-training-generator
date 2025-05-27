import os
from dotenv import load_dotenv
import streamlit.web.cli as stcli
import sys
import torch
import subprocess
from pathlib import Path
from setup_directories import setup_model_directories
from huggingface_hub import login

def main():
    # Load environment variables
    load_dotenv()

    # HuggingFace login
    login(token=os.getenv("HF_TOKEN"))
    
    # Configure PyTorch
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    
    # Setup model directories
    if not setup_model_directories():
        print("Failed to setup model directories. Please run setup_directories.py manually.")
        sys.exit(1)
    
    # Get the absolute path to run.py
    run_py_path = str(Path(__file__).parent / "run.py")
   
    # Run Streamlit with proper configuration
    try:
        # First try using subprocess
        streamlit_cmd = [
            sys.executable, "-m", "streamlit", "run",
            run_py_path,
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost",
            "--browser.serverPort", "8501"
        ]
        
        process = subprocess.Popen(streamlit_cmd)
        process.wait()
    except Exception as e:
        print(f"Error using subprocess: {e}")
        print("Trying alternative method...")
        
        # Fallback to direct command
        try:
            cmd = f"streamlit run {run_py_path} --server.address 0.0.0.0 --server.port 8501"
            os.system(cmd)
        except Exception as e:
            print(f"Error using direct command: {e}")
            print("Please try running the following command manually:")
            print(f"streamlit run {run_py_path} --server.address 0.0.0.0 --server.port 8501")
            sys.exit(1)

if __name__ == "__main__":
    main()
