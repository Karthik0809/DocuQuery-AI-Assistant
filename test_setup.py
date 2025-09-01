#!/usr/bin/env python3
"""
Test script to verify the DocuQuery AI Assistant setup
Run this to check if all modules can be imported correctly
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing config.py: {e}")
        return False
    
    try:
        import utils
        print("✓ utils.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing utils.py: {e}")
        return False
    
    try:
        import document_processor
        print("✓ document_processor.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing document_processor.py: {e}")
        return False
    
    try:
        import rag_engine
        print("✓ rag_engine.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing rag_engine.py: {e}")
        return False
    
    try:
        import llm_interface
        print("✓ llm_interface.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing llm_interface.py: {e}")
        return False
    
    try:
        import export_manager
        print("✓ export_manager.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing export_manager.py: {e}")
        return False
    
    try:
        import main
        print("✓ main.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing main.py: {e}")
        return False
    
    try:
        import ui
        print("✓ ui.py imported successfully")
    except Exception as e:
        print(f"✗ Error importing ui.py: {e}")
        return False
    
    return True

def test_main_class():
    """Test if the main chatbot class can be instantiated"""
    print("\nTesting main class instantiation...")
    
    try:
        from main import EnhancedRAGChatbot
        rag = EnhancedRAGChatbot()
        print("✓ EnhancedRAGChatbot instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Error instantiating EnhancedRAGChatbot: {e}")
        return False

def test_config():
    """Test if configuration values are accessible"""
    print("\nTesting configuration...")
    
    try:
        from config import DEFAULT_GEMINI_MODEL, DEFAULT_TOP_K
        print(f"✓ Configuration loaded: Model={DEFAULT_GEMINI_MODEL}, TopK={DEFAULT_TOP_K}")
        return True
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def main():
    """Run all tests"""
    print("DocuQuery AI Assistant - Setup Test")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test main class
    if not test_main_class():
        success = False
    
    # Test configuration
    if not test_config():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! The setup is working correctly.")
        print("You can now run the application with: python run.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return success

if __name__ == "__main__":
    main()
