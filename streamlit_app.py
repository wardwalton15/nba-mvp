"""
Entry point for Streamlit Cloud deployment.
"""
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main app
from src.dashboard.app import main

main()
