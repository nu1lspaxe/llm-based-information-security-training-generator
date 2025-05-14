import os
import sys
import asyncio
import nest_asyncio
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Fix torch.classes.__path__ issue
torch.classes.__path__ = []

# Configure for WSL
if not torch.cuda.is_available():
    torch.set_num_threads(4)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def main():
    try:
        # Import the Streamlit app
        from ui.app import main as app_main
        
        # Configure asyncio for WSL
        if sys.platform == 'linux':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.set_debug(True)
        
        await app_main()
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Create and run the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close() 