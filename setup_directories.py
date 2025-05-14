import os
from pathlib import Path
import logging
import subprocess
import sys
import stat
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_admin():
    """Check if running with admin/root privileges"""
    try:
        if platform.system() == 'Windows':
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        else:
            return os.geteuid() == 0
    except:
        return False

def run_as_admin():
    """Request admin/root privileges"""
    if not is_admin():
        logger.info("Requesting administrator privileges...")
        if platform.system() == 'Windows':
            import ctypes
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        else:
            logger.info("Please run this script with sudo")
        sys.exit()

def set_full_permissions(path):
    """Set full permissions for a path"""
    try:
        if platform.system() == 'Windows':
            # Windows-specific permissions
            username = os.getenv('USERNAME', 'Everyone')
            cmd = f'icacls "{path}" /grant {username}:(OI)(CI)F /T /Q'
            subprocess.run(cmd, shell=True, check=True)
            cmd = f'attrib -R "{path}" /S /D'
            subprocess.run(cmd, shell=True, check=True)
        else:
            # Unix-like systems permissions
            os.chmod(str(path), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
    except Exception as e:
        logger.error(f"Could not set permissions for {path}: {e}")
        return False
    return True

def setup_model_directories():
    """Set up model directories with proper permissions"""
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.absolute()
        model_dir = project_root / "fine_tuned_model"
        
        # Create directories if they don't exist
        directories = [
            model_dir,
            model_dir / "attack",
            model_dir / "response"
        ]
        
        for directory in directories:
            if not directory.exists():
                logger.info(f"Creating directory: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
            
            # Set permissions
            if not set_full_permissions(directory):
                logger.error(f"Failed to set permissions for {directory}")
                return False
            
            # Create a placeholder file
            placeholder = directory / ".placeholder"
            if not placeholder.exists():
                placeholder.touch()
                if not set_full_permissions(placeholder):
                    logger.error(f"Failed to set permissions for placeholder file in {directory}")
                    return False
        
        # Verify permissions
        for directory in directories:
            try:
                test_file = directory / "test_write.tmp"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                logger.error(f"Could not verify write permissions for {directory}: {e}")
                return False
        
        logger.info("Model directories setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up model directories: {str(e)}")
        return False

if __name__ == "__main__":
    if not is_admin():
        logger.warning("Script is not running with administrator privileges. Some operations may fail.")
        logger.warning("Please run this script as administrator/root for best results.")
    
    if setup_model_directories():
        print("Model directories setup completed successfully!")
    else:
        print("Failed to setup model directories. Please check the logs for details.")
        print("Try running this script as administrator/root.") 