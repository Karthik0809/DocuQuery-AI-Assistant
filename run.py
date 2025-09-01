#!/usr/bin/env python3
"""
Simple launcher script for DocuQuery AI Assistant
Run this file to start the application
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ui import launch
    
    if __name__ == "__main__":
        print("Starting DocuQuery AI Assistant...")
        launch()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)
