# Gradio UI for Enhanced RAG Chatbot
import gradio as gr
from main import EnhancedRAGChatbot
from config import DEFAULT_GEMINI_MODEL, DEFAULT_TOP_K, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_USE_EXPANSION, DEFAULT_ENABLE_RERANKING, DEFAULT_MAX_TOKENS, DEFAULT_SHOW_DEBUG

def launch():
    rag = EnhancedRAGChatbot()

    def ui_clear():
        cleared_status, cleared_chat, cleared_stats = rag.clear()
        return (cleared_status, cleared_chat, cleared_stats, gr.update(value=None), gr.update(value=""),
                gr.update(value=DEFAULT_TOP_K), gr.update(value=DEFAULT_CONFIDENCE_THRESHOLD), gr.update(value=DEFAULT_USE_EXPANSION), gr.update(value=DEFAULT_ENABLE_RERANKING),
                gr.update(value=DEFAULT_MAX_TOKENS), gr.update(value=DEFAULT_SHOW_DEBUG), gr.update(value=""), gr.update(value="en"), 
                gr.update(value="*Enter your Gemini API key above*"), gr.update(choices=["all"], value="all"))

    def reset_topk(): return DEFAULT_TOP_K
    def reset_confidence(): return DEFAULT_CONFIDENCE_THRESHOLD
    def reset_expansion(): return DEFAULT_USE_EXPANSION
    def reset_reranking(): return DEFAULT_ENABLE_RERANKING
    def reset_max_tokens(): return DEFAULT_MAX_TOKENS
    def reset_debug(): return DEFAULT_SHOW_DEBUG

    with gr.Blocks(title="Enhanced RAG Chatbot", theme=gr.themes.Soft()) as demo:
        # Header with title
        gr.HTML("<h1>DocuQuery AI Assistant</h1>")
        gr.HTML("<p>Your smart AI assistant for text-based PDFs — save hours by getting instant answers from your documents.</p>")

        with gr.Row():
            with gr.Column(scale=1):
                # Gemini API Key Configuration
                api_key = gr.Textbox(label="Google API Key (kept local to your session)", type="password", placeholder="Paste your key...")
                gr.Markdown(f"*Using {DEFAULT_GEMINI_MODEL} for optimal performance*")
                set_key_btn = gr.Button("Set Gemini Key", variant="primary")
                gemini_status = gr.Markdown("*Enter your Gemini API key above*")

                files = gr.File(label="Upload PDF Documents", file_count="multiple", file_types=[".pdf"], type="filepath")
                process_btn = gr.Button("Process Documents", variant="primary")
                status = gr.Textbox(label="Processing Status", lines=15, interactive=False, value="Ready to process documents...")

                with gr.Accordion("Smart Search Settings", open=False):
                    gr.Markdown("### Search Configuration")
                    
                    with gr.Row():
                        with gr.Column(scale=5):
                            topk = gr.Slider(minimum=3, maximum=8, value=DEFAULT_TOP_K, step=1, label="Top-K Results", info="Number of chunks to retrieve")
                        with gr.Column(scale=1, min_width=70):
                            topk_reset = gr.Button("Reset", variant="secondary")
                    
                    with gr.Row():
                        with gr.Column(scale=5):
                            confidence_threshold = gr.Slider(minimum=0.0, maximum=0.8, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05, label="Confidence Threshold", info="Minimum confidence for extractive answers")
                        with gr.Column(scale=1, min_width=70):
                            conf_reset = gr.Button("Reset", variant="secondary")
                    
                    gr.Markdown("### Query Enhancement")
                    with gr.Row():
                        with gr.Column(scale=5):
                            use_expansion = gr.Checkbox(value=DEFAULT_USE_EXPANSION, label="Enable Query Expansion", info="Automatically expand queries with synonyms")
                        with gr.Column(scale=1, min_width=70):
                            exp_reset = gr.Button("Reset", variant="secondary")
                    
                    with gr.Row():
                        with gr.Column(scale=5):
                            enable_reranking = gr.Checkbox(value=DEFAULT_ENABLE_RERANKING, label="Enable Result Reranking", info="Improve result relevance with semantic reranking")
                        with gr.Column(scale=1, min_width=70):
                            rerank_reset = gr.Button("Reset", variant="secondary")
                    
                    gr.Markdown("### Answer Generation")
                    with gr.Row():
                        with gr.Column(scale=5):
                            max_tokens = gr.Slider(minimum=0, maximum=600, value=DEFAULT_MAX_TOKENS, step=50, label="Max Answer Length", info="0 = Auto-determine optimal length")
                        with gr.Column(scale=1, min_width=70):
                            max_reset = gr.Button("Reset", variant="secondary")
                    
                    gr.Markdown("### Debug & Performance")
                    with gr.Row():
                        with gr.Column(scale=5):
                            show_debug = gr.Checkbox(value=DEFAULT_SHOW_DEBUG, label="Show Debug Information", info="Display retrieval details and performance metrics")
                        with gr.Column(scale=1, min_width=70):
                            debug_reset = gr.Button("Reset", variant="secondary")

                    with gr.Row():
                        stats_btn = gr.Button("System Information", variant="secondary")
                        metrics_btn = gr.Button("Performance Metrics", variant="secondary")
                        reset_metrics_btn = gr.Button("Reset Metrics", variant="secondary")
                    stats_box = gr.Textbox(label="System Information", lines=20, interactive=False)
                
                # Export Functionality
                with gr.Accordion("Export & Share", open=False):
                    gr.Markdown("### Export Options")
                    with gr.Row():
                        export_format = gr.Dropdown(choices=["pdf", "docx", "txt"], value="pdf", label="Export Format")
                        export_filename = gr.Textbox(label="Filename (optional)", placeholder="Leave empty for auto-generated name")
                    
                    with gr.Row():
                        export_conversation_btn = gr.Button("Export Conversation", variant="secondary")
                    
                    export_status = gr.Textbox(label="Export Status", lines=3, interactive=False, placeholder="Export status will appear here...")
                    export_file_output = gr.File(label="Download Exported File", visible=False)

                clear_btn = gr.Button("Clear All", variant="secondary")

            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Enhanced RAG Assistant", height=600, show_label=True)
                
                # Language and Speech Controls
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Language Settings:**")
                        target_language = gr.Dropdown(
                            choices=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"],
                            value="en",
                            label="Target Language",
                            info="Language for answers"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("**Speech Features:**")
                        speech_btn = gr.Button("Speech Input", variant="secondary")
                
                # Document Section Selection
                with gr.Row():
                    gr.Markdown("**Select Document Section:**")
                    document_selector = gr.Dropdown(
                        choices=["all"],
                        value="all",
                        label="Document Section",
                        info="Choose specific section or 'all' for entire document"
                    )
                
                with gr.Row():
                    question_input = gr.Textbox(placeholder="Ask any question about your documents...", lines=2, scale=4, label="Your Question")
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)

        # Bind actions
        set_key_btn.click(fn=lambda k: rag.set_gemini(k), inputs=[api_key], outputs=[status, gemini_status], show_progress="hidden")
        
        # Update document selector when documents are processed
        def update_document_selector():
            print(f"update_document_selector called")
            print(f"rag.processed: {rag.processed}")
            print(f"rag.document_sections: {rag.document_sections}")
            
            if rag.processed:
                choices = ["all"]
                
                # Add document sections if available
                if rag.document_sections:
                    for filename, sections in rag.document_sections.items():
                        for section_name in sections.keys():
                            # Format: "filename - section_name"
                            section_option = f"{filename} - {section_name.replace('_', ' ').title()}"
                            print(f"Adding section: {section_option}")
                            choices.append(section_option)
                
                # Also add individual documents if no sections
                if not rag.document_sections:
                    for doc_info in rag.processed:
                        filename = doc_info["file"]
                        print(f"Adding document: {filename}")
                        choices.append(filename)
                
                print(f"Updated document selector with choices: {choices}")
                return gr.update(choices=choices, value="all")
            
            print("No documents processed yet, keeping default choices")
            return gr.update(choices=["all"], value="all")
        
        # Process documents and then update document selector
        def process_and_update(files):
            # First process documents
            result = rag.process_documents(files)
            # Then update document selector with sections
            selector_update = update_document_selector()
            print(f"Processing complete. Document sections available: {rag.document_sections}")
            return result, selector_update
        
        process_btn.click(fn=process_and_update, inputs=[files], outputs=[status, document_selector], show_progress="full")
        
        # Handle queries with document and language support
        def handle_query(question, history, selected_doc, k, conf, exp, rerank, mt, debug, target_lang):
            if target_lang != "en":
                # Use bilingual answer method
                return rag.answer_with_translation(question, history, target_lang, k, conf, exp, rerank, mt, debug)
            else:
                # Use regular answer method for all documents
                return rag.answer(question, history, k, conf, exp, mt, rerank, debug)
        
        ask_btn.click(fn=handle_query,
                      inputs=[question_input, chat, document_selector, topk, confidence_threshold, use_expansion, enable_reranking, max_tokens, show_debug, target_language],
                      outputs=[question_input, chat], show_progress="full")
        question_input.submit(fn=handle_query,
                              inputs=[question_input, chat, document_selector, topk, confidence_threshold, use_expansion, enable_reranking, max_tokens, show_debug, target_language],
                              outputs=[question_input, chat], show_progress="full")
        
        # Speech and Language Features
        speech_btn.click(fn=rag.speech_to_text, outputs=[question_input], show_progress="full")

        stats_btn.click(fn=rag.stats, outputs=[stats_box], show_progress="hidden")
        metrics_btn.click(fn=rag.get_performance_metrics, outputs=[stats_box], show_progress="hidden")
        reset_metrics_btn.click(fn=rag.reset_metrics, outputs=[stats_box], show_progress="hidden")

        topk_reset.click(fn=reset_topk, outputs=[topk], show_progress="hidden")
        conf_reset.click(fn=reset_confidence, outputs=[confidence_threshold], show_progress="hidden")
        exp_reset.click(fn=reset_expansion, outputs=[use_expansion], show_progress="hidden")
        rerank_reset.click(fn=reset_reranking, outputs=[enable_reranking], show_progress="hidden")
        max_reset.click(fn=reset_max_tokens, outputs=[max_tokens], show_progress="hidden")
        debug_reset.click(fn=reset_debug, outputs=[show_debug], show_progress="hidden")

        clear_btn.click(fn=ui_clear, outputs=[status, chat, stats_box, files, question_input, topk, confidence_threshold,
                                              use_expansion, enable_reranking, max_tokens, show_debug, api_key, target_language, gemini_status, document_selector],
                        show_progress="hidden")
        
        # Export functionality
        def handle_conversation_export(fmt, filename, chat_history):
            result = rag.export_conversation(chat_history, fmt, filename)
            if result[0]:  # If file was created
                return result[1], gr.update(value=result[0], visible=True)
            else:
                return result[1], gr.update(visible=False)
        
        export_conversation_btn.click(
            fn=handle_conversation_export,
            inputs=[export_format, export_filename, chat],
            outputs=[export_status, export_file_output],
            show_progress="full"
        )

    # Simple launch - just open in browser
    print("Launching Enhanced RAG Chatbot...")
    print(f"Using Gemini model: {DEFAULT_GEMINI_MODEL}")
    print("SINGLE MODEL CONFIGURATION - No model selection needed!")
    print("Simple browser launch - ready for Vercel deployment later!")
    
    # Simple launch - just open in browser
    print("Opening in browser...")
    demo.launch(
        debug=True,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    launch()
