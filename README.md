# DocuQuery AI Assistant - Enhanced RAG Chatbot

A sophisticated AI-powered document question-answering system designed specifically for mortgage and financial documents. This application uses advanced Retrieval-Augmented Generation (RAG) techniques to provide accurate, context-aware answers from PDF documents.

## Features

### Core Functionality
- **Multi-Modal PDF Processing**: Handles both digital and scanned PDFs with OCR
- **Advanced RAG Pipeline**: Combines multiple retrieval methods for superior accuracy
- **Domain-Specific Optimization**: Tailored for mortgage and financial documents
- **Real-time Q&A**: Instant answers with confidence scoring
- **Document Sectioning**: Intelligent document segmentation for targeted queries

### Advanced Features
- **Multi-Language Support**: 12+ languages with automatic translation
- **Speech Interface**: Voice input and text-to-speech capabilities
- **Export Functionality**: Export conversations to PDF, Word, or text formats
- **Performance Metrics**: Comprehensive system monitoring and analytics
- **Error Recovery**: Graceful degradation and automatic error handling

### Technical Capabilities
- **Hierarchical Chunking**: Multi-level text segmentation for optimal retrieval
- **Hybrid Search**: Combines semantic, keyword, and TF-IDF search
- **Query Expansion**: Automatic query enhancement with synonyms
- **Result Reranking**: Semantic reranking for improved relevance
- **Extractive QA**: Precise answer extraction using fine-tuned models

## Architecture

The application is organized into modular components:

```
├── config.py              # Configuration and constants
├── utils.py               # Utility functions and helper classes
├── document_processor.py  # PDF processing and OCR
├── rag_engine.py          # RAG pipeline and vector search
├── llm_interface.py       # LLM integration and speech processing
├── export_manager.py      # Export functionality
├── main.py                # Main application logic
├── ui.py                  # Gradio user interface
├── run.py                 # Application launcher script
├── test_setup.py          # Setup verification and testing
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── LICENSE               # License information
```

### Key Components

1. **Document Processor**: Handles PDF extraction, OCR, and document segmentation
2. **RAG Engine**: Manages chunking, vector storage, and retrieval
3. **LLM Interface**: Integrates with Google Gemini for text generation
4. **Export Manager**: Handles conversation and document exports
5. **Performance Metrics**: Tracks system performance and user interactions

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Tesseract OCR (for scanned document processing)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DocuQuery-AI-Assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Get Google Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Keep it secure for use in the application

5. **Verify Setup**:
   ```bash
   python test_setup.py
   ```
   This will test all module imports and verify the system is ready to run.

## Running the Application

### Option 1: Using the Launcher Script (Recommended)
```bash
python run.py
```
The `run.py` script is a simple launcher that imports and calls the `launch()` function from `ui.py`. This is the recommended way to start the application as it provides a clean entry point.

### Option 2: Direct Execution
```bash
python ui.py
```
This directly runs the Gradio interface by executing the `ui.py` file.

### Option 3: Using Gradio Directly
```bash
python -c "from ui import launch; launch()"
```
This method directly calls the launch function from the command line.

### What Happens When You Run It

1. **Gradio UI Initialization**: The application starts the Gradio web interface
2. **Local Server**: Typically runs on `http://127.0.0.1:7860` or the next available port
3. **Browser Launch**: The interface should automatically open in your default browser
4. **Ready State**: The application is ready to:
   - Accept PDF document uploads
   - Process documents with OCR and text extraction
   - Answer questions using the RAG pipeline
   - Handle voice input and output
   - Export conversations and summaries

## Testing and Verification

### Test Setup Script (`test_setup.py`)

The `test_setup.py` script is a comprehensive verification tool that ensures your environment is properly configured:

#### What It Tests:
- **Module Imports**: Verifies all Python modules can be imported successfully
- **Class Instantiation**: Tests that the main `EnhancedRAGChatbot` class can be created
- **Dependency Check**: Ensures all required packages are installed
- **Configuration Validation**: Checks that configuration files are accessible

#### Running the Tests:
```bash
python test_setup.py
```

#### Expected Output:
```
✓ Testing module imports...
✓ Testing main class instantiation...
✓ All tests passed! The setup is working correctly.
You can now run the application with: python run.py
```

#### If Tests Fail:
- **Import Errors**: Run `pip install -r requirements.txt` to install missing dependencies
- **Configuration Issues**: Check that all files are in the correct locations
- **Permission Errors**: Ensure you have write permissions in the project directory

### Manual Testing

You can also perform manual testing:

```bash
# Test individual components
python -c "from config import *; print('Config loaded successfully')"
python -c "from utils import *; print('Utils loaded successfully')"
python -c "from main import EnhancedRAGChatbot; rag = EnhancedRAGChatbot(); print('System initialized')"
```

## Usage

### Getting Started

1. **Launch the Application**: Run `python run.py` to start the web interface
2. **Configure API Key**: Enter your Google Gemini API key in the left panel
3. **Upload Documents**: Upload PDF files (supports multiple files)
4. **Process Documents**: Click "Process Documents" to extract and index content
5. **Ask Questions**: Use the chat interface to ask questions about your documents

### Advanced Features

#### Document Section Selection
- Choose specific document sections for targeted queries
- Automatic section detection for mortgage documents
- Support for custom document types

#### Language Support
- Automatic language detection
- Multi-language Q&A with translation
- Support for 12+ languages including:
  - English, Spanish, French, German
  - Italian, Portuguese, Russian, Japanese
  - Korean, Chinese, Arabic, Hindi

#### Export Options
- Export conversations to PDF, Word, or text formats
- Customizable filenames and formats
- Professional formatting with timestamps

#### Performance Tuning
- Adjustable confidence thresholds
- Query expansion controls
- Result reranking options
- Debug information display

## Configuration

### Key Settings

The application can be customized through `config.py`:

```python
# Model Configuration
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Performance Settings
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_USE_EXPANSION = True
DEFAULT_ENABLE_RERANKING = True

# OCR Settings
DEFAULT_DPI = 300
OCR_CONFIG = "--oem 3 --psm 6 ..."

# Chunking Settings
TARGET_TOKENS = 300
OVERLAP_SENTENCES = 2
BOUNDARY_DROP = 0.15
```

### Domain Customization

The system is optimized for mortgage documents but can be adapted for other domains:

1. **Update Keywords**: Modify `MORTGAGE_KEYWORDS` in `config.py`
2. **Adjust Heuristics**: Customize domain scoring in `utils.py`
3. **Add Parsers**: Extend `MortgageParser` class for new document types

## Performance Metrics

The system provides comprehensive performance tracking:

- **Retrieval Performance**: Hit rates, relevance scores, latency
- **Answer Quality**: Confidence scores, answer completeness
- **System Performance**: Response times, throughput, error rates
- **User Analytics**: Query patterns, document usage statistics

## Troubleshooting

### Common Issues and Solutions

#### 1. Module Import Errors
**Problem**: `ModuleNotFoundError` when running the application
**Solution**: 
```bash
pip install -r requirements.txt
python test_setup.py
```

#### 2. Tesseract OCR Not Found
**Problem**: OCR functionality not working
**Solution**: 
- **Windows**: Download and install Tesseract from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

#### 3. API Key Issues
**Problem**: "Invalid API key" errors
**Solution**: 
- Verify your Google Gemini API key is correct
- Ensure you have sufficient API quota
- Check that the API key is active and not expired

#### 4. Port Already in Use
**Problem**: Gradio can't start due to port conflicts
**Solution**: The application will automatically find an available port, or you can specify a custom port in the code

#### 5. Memory Issues
**Problem**: Application crashes with large documents
**Solution**: 
- Process documents one at a time
- Reduce `DEFAULT_TOP_K` in config.py
- Close other applications to free memory

### Debug Mode

Enable debug information by modifying the configuration:
```python
# In config.py or through the UI
DEBUG_MODE = True
```

This will show detailed information about:
- Document processing steps
- Retrieval performance
- Answer generation process
- System resource usage

## Security & Privacy

- **Local Processing**: All document processing happens locally
- **API Key Security**: API keys are stored in session memory only
- **No Data Storage**: Documents are processed in memory, not stored
- **Export Control**: Users control what gets exported

## Development

### Project Structure

```
├── config.py              # Configuration management
├── utils.py               # Core utilities and metrics
├── document_processor.py  # Document processing pipeline
├── rag_engine.py          # RAG implementation
├── llm_interface.py       # LLM and speech integration
├── export_manager.py      # Export functionality
├── main.py                # Main application class
├── ui.py                  # User interface
├── run.py                 # Application launcher
├── test_setup.py          # Setup verification
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── LICENSE               # License information
```

### Adding New Features

1. **New Document Types**: Extend `EnhancedDocumentProcessor`
2. **Additional Languages**: Update `SUPPORTED_LANGUAGES` in config
3. **Export Formats**: Add new formats to `ExportManager`
4. **Search Methods**: Implement new retrieval methods in `AdvancedVectorStore`

### Testing

Run the included smoke tests:
```bash
python test_setup.py
```

For more comprehensive testing:
```bash
python -c "from main import EnhancedRAGChatbot; rag = EnhancedRAGChatbot(); print('System initialized successfully')"
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes**: Follow the existing code style
4. **Test thoroughly**: Ensure all functionality works
5. **Submit a pull request**: Include detailed description of changes

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Keep functions focused and modular

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google Gemini**: For providing the LLM capabilities
- **Gradio**: For the excellent web interface framework
- **Sentence Transformers**: For semantic search capabilities
- **Open Source Community**: For the various libraries and tools used

## Support

For support and questions:

- **Issues**: Use the GitHub issues page
- **Documentation**: Check this README and inline code comments
- **Community**: Join our discussion forum

## Version History

### v1.0.0 (Current)
- Initial release with full RAG pipeline
- Multi-language support
- Export functionality
- Performance metrics
- Speech interface
- Modular architecture
- Comprehensive testing suite

### Planned Features
- Real-time collaboration
- Advanced document analytics
- Custom model fine-tuning
- API endpoints for integration
- Mobile application

---

**Note**: This application requires a Google Gemini API key for full functionality. The API key is used only for text generation and is not stored permanently.
