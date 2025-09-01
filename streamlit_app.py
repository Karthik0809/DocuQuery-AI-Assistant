import streamlit as st
import subprocess
import sys
import os
import threading
import time
import requests
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Enhanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered document question-answering system</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🚀 Quick Start")
    
    # Check if Gradio app is running
    gradio_running = False
    try:
        response = requests.get("http://localhost:7860", timeout=2)
        if response.status_code == 200:
            gradio_running = True
    except:
        pass
    
    if gradio_running:
        st.sidebar.success("✅ Gradio app is running!")
        st.sidebar.markdown("**Access your app:**")
        st.sidebar.markdown("[Open Gradio Interface](http://localhost:7860)")
        
        # Main content when Gradio is running
        st.markdown("""
        <div class="success-box">
            <h3>🎉 Your RAG Chatbot is Ready!</h3>
            <p>Click the link in the sidebar to access the full Gradio interface with all features:</p>
            <ul>
                <li>📄 Document upload and processing</li>
                <li>🤖 AI-powered Q&A</li>
                <li>🗣️ Voice input and output</li>
                <li>🌐 Multi-language support</li>
                <li>📊 Export functionality</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.sidebar.warning("⚠️ Gradio app not running")
        
        # Start Gradio app button
        if st.sidebar.button("🚀 Launch Gradio App", type="primary"):
            with st.spinner("Starting Gradio app..."):
                try:
                    # Start Gradio app in a separate thread
                    def start_gradio():
                        os.system(f"{sys.executable} ui.py")
                    
                    thread = threading.Thread(target=start_gradio, daemon=True)
                    thread.start()
                    
                    # Wait for app to start
                    time.sleep(5)
                    
                    # Check if it's running
                    try:
                        response = requests.get("http://localhost:7860", timeout=5)
                        if response.status_code == 200:
                            st.success("✅ Gradio app started successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to start Gradio app")
                    except:
                        st.error("❌ Failed to start Gradio app")
                        
                except Exception as e:
                    st.error(f"❌ Error starting app: {str(e)}")
        
        # Manual instructions
        st.markdown("""
        <div class="info-box">
            <h3>📋 How to Start Your RAG Chatbot</h3>
            <p><strong>Option 1:</strong> Click the "Launch Gradio App" button in the sidebar</p>
            <p><strong>Option 2:</strong> Run manually in terminal:</p>
            <code>python ui.py</code>
            <p><strong>Option 3:</strong> Use the launcher script:</p>
            <code>python run.py</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("## ✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📄 Document Processing
        - Multi-modal PDF support
        - OCR for scanned documents
        - Intelligent text extraction
        - Document sectioning
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Capabilities
        - Advanced RAG pipeline
        - Multi-language support
        - Voice input/output
        - Real-time Q&A
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Export & Analytics
        - Export to PDF/Word/TXT
        - Performance metrics
        - Conversation history
        - Debug information
        """)
    
    # Setup instructions
    st.markdown("## 🛠️ Setup Instructions")
    
    with st.expander("📋 Prerequisites"):
        st.markdown("""
        1. **Python 3.8+** installed
        2. **Google Gemini API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
        3. **Tesseract OCR** (for scanned documents)
        4. **All dependencies** installed via `pip install -r requirements.txt`
        """)
    
    with st.expander("🔧 Installation"):
        st.markdown("""
        ```bash
        # Clone the repository
        git clone https://github.com/Karthik0809/Enhanced-RAG-Chatbot.git
        cd Enhanced-RAG-Chatbot
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Test setup
        python test_setup.py
        
        # Run the app
        python run.py
        ```
        """)
    
    with st.expander("🌐 Deployment Options"):
        st.markdown("""
        ### Free Deployment Options:
        
        1. **Streamlit Cloud** (Current) - Free tier available
        2. **Hugging Face Spaces** - Free for public repos
        3. **Railway** - Free tier with limitations
        4. **Render** - Free tier available
        5. **Heroku** - Free tier discontinued, but alternatives exist
        
        ### For Global Access:
        - All platforms above provide public URLs
        - No local setup required for users
        - Automatic HTTPS and CDN
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit, Gradio, and Google Gemini</p>
        <p>Repository: <a href="https://github.com/Karthik0809/Enhanced-RAG-Chatbot" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
