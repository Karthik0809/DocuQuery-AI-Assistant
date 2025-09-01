# Main application file for Enhanced RAG Chatbot
import os
import re
import time
import torch
from sentence_transformers import SentenceTransformer

from config import DEFAULT_GEMINI_MODEL
from utils import PerformanceMetrics, ResponseDebugInfo, ErrorRecovery, safe_sent_tokenize
from document_processor import EnhancedDocumentProcessor, MortgageParser
from rag_engine import HierarchicalChunker, AdvancedVectorStore, EnhancedQAModel
from llm_interface import GeminiGenerator, SpeechProcessor, BilingualProcessor
from export_manager import ExportManager

class EnhancedRAGChatbot:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {device.upper()}")
        self.embed = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        self.doc = EnhancedDocumentProcessor()
        self.chunker = HierarchicalChunker(self.embed)
        self.vs = AdvancedVectorStore(self.embed)
        self.qa = EnhancedQAModel()
        self.gen = GeminiGenerator()
        self.processed = []
        self.total_chunks = 0
        self.document_stats = {}
        self.raw_texts = {}  # filename -> raw extracted text
        self.document_sections = {}  # filename -> sections
        self.parser = MortgageParser()
        self.components_table = {}
        
        # Speech and Language Processing
        self.speech_processor = SpeechProcessor()
        self.bilingual_processor = BilingualProcessor()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.debug_info = ResponseDebugInfo()
        
        # New enhanced features
        self.export_manager = ExportManager()
        self.error_recovery = ErrorRecovery()

    def _set_components_table(self):
        """Set up components table for tracking document components"""
        self.components_table = {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "chunker": "HierarchicalChunker",
            "vector_store": "AdvancedVectorStore",
            "qa_model": "EnhancedQAModel",
            "generator": "GeminiGenerator",
            "total_chunks": self.total_chunks,
            "document_stats": self.document_stats
        }

    # --- Configure API key + model from UI
    def set_gemini(self, api_key: str):
        print(f"set_gemini called with API key: {'Yes' if api_key else 'No'}")
        print(f"API key length: {len(api_key) if api_key else 0}")
        print(f"Using model: {DEFAULT_GEMINI_MODEL}")
        
        ok = self.gen.configure(api_key)
        
        if ok:
            status_msg = f"Gemini configured successfully with model: {DEFAULT_GEMINI_MODEL}"
            gemini_status = f"**Gemini Ready!**\nModel: {DEFAULT_GEMINI_MODEL}\nStatus: Active"
            print(f"Configuration successful: {status_msg}")
            return status_msg, gemini_status
        else:
            status_msg = f"Failed to configure Gemini. Check your API key."
            gemini_status = f"**Gemini Configuration Failed**\nPlease check your API key and try again."
            print(f"Configuration failed: {status_msg}")
            return status_msg, gemini_status

    def process_documents(self, file_paths):
        if not file_paths: 
            return "No files provided. Upload PDF(s) to begin."
        msgs = ["Starting Enhanced Document Processing Pipeline"]
        all_chunks = []
        stats = {"total_pages":0, "total_text_length":0, "extraction_methods": {}}

        for path in file_paths:
            try:
                extraction_start = time.time()
                fname = os.path.basename(path)
                msgs.append(f"Processing {fname}")
                text, meta = self.doc.extract_text(path)
                extraction_time = time.time() - extraction_start

                if meta.get("extraction_method")=="failed":
                    msgs.append(f"Failed: {text}")
                    continue
                
                stats["total_pages"] += meta.get("total_pages",0)
                stats["total_text_length"] += len(text)
                if meta["extraction_method"] not in stats["extraction_methods"]:
                    stats["extraction_methods"][meta["extraction_method"]] = 0
                stats["extraction_methods"][meta["extraction_method"]] += 1
                
                # Store document sections
                if "sections" in meta:
                    self.document_sections[fname] = meta["sections"]
                    section_summary = self.doc.segmenter.get_section_summary(meta["sections"])
                    msgs.append(f"Document sections detected: {len(meta['sections'])} sections")
                
                chunks = self.chunker.split(text, meta)
                all_chunks.extend(chunks)
                dist = {}
                for c in chunks:
                    chunk_type = c["metadata"].get("chunk_type","medium")
                    dist[chunk_type] = dist.get(chunk_type, 0) + 1
                
                self.processed.append({"file": meta["file"], "pages": meta.get("total_pages",0), "chunks": len(chunks),
                                       "chunk_distribution": dist, "extraction_method": meta["extraction_method"],
                                       "structure": meta.get("structure", {}), "extraction_time": extraction_time,
                                       "sections": meta.get("sections", {})})
                self.raw_texts[fname] = text
                msgs.append(f"Created {len(chunks)} chunks (extraction: {extraction_time:.2f}s)")
                
            except Exception as e:
                # Use error recovery system
                error_result = self.error_recovery.handle_error(e, "document_processing")
                msgs.append(f"Error processing {os.path.basename(path)}: {error_result['message']}")
                if 'suggestion' in error_result:
                    msgs.append(f"Suggestion: {error_result['suggestion']}")
        
        if not all_chunks:
            msgs.append("No searchable content created from uploaded documents")
            return "\n".join(msgs)
        
        msgs.append(f"Building Advanced Search Index ({len(all_chunks)} chunks)")
        self.vs.build(all_chunks)
        self.total_chunks = len(all_chunks)
        self.document_stats = stats
        # capture components for the table
        self._set_components_table()
        msgs.append("System Ready")
        return "\n".join(msgs)

    def get_document_sections(self):
        """Get available document sections for UI display"""
        if not self.document_sections:
            return "No documents processed yet."
        
        summary = "**Available Document Sections**\n\n"
        for filename, sections in self.document_sections.items():
            summary += f"**File: {filename}**\n"
            for section_name, section_info in sections.items():
                pages = section_info["pages"]
                page_range = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
                summary += f"  {section_name.replace('_', ' ').title()} ({page_range})\n"
            summary += "\n"
        
        return summary

    def answer(self, question, history, k=5, confidence_threshold=0.3, use_expansion=True, max_answer_tokens=0, enable_reranking=True, show_debug=True):
        if not question.strip():
            return "", history
        if self.vs.index is None:
            return "", history + [(question, "Please upload and process documents first (left panel).")]

        # Start timing the entire response
        response_start = time.time()

        try:
            hits, retrieval_time = self.vs.search(
                question,
                top_k=int(k),
                use_expansion=bool(use_expansion),
                rerank=bool(enable_reranking)
            )

            if not hits:
                total_time = time.time() - response_start
                self.metrics.log_query(total_time, retrieval_time, 0.0, 0.0, 0.0, 0, "no_results", 0.0, False)
                return "", history + [(question, "No relevant information found in the uploaded documents.")]

            # Use higher confidence threshold for better quality
            threshold = max(0.3, float(confidence_threshold))  # Minimum 0.3
            max_relevance = max(hit.get("relevance", 0.0) for hit in hits)

            # 1) Try extractive QA first (precise answers)
            qa = self.qa.answer(question, hits)
            final = None
            method = "extractive_qa"
            conf = qa.get("confidence", 0.0)
            qa_time = qa.get("processing_time", 0.0)
            generation_time = 0.0

            if qa["found"] and conf >= threshold:
                final = qa["answer"].strip()
                # Clean any residual citations from QA
                final = self._clean_answer(final)
            else:
                # 2) Fall back to Gemini (generative)
                budget = int(max_answer_tokens) if max_answer_tokens and int(max_answer_tokens) > 0 else None
                gen, gen_time = self.gen.generate(question, hits, max_tokens=budget)
                generation_time = gen_time
                if gen and len(gen.split()) >= 3 and not gen.lower().startswith(("insufficient", "error", "not stated")):
                    final = self._clean_answer(gen.strip())
                    method = "generative"
                    conf = 0.75 if self.gen.api_key_set else 0.3
                else:
                    # 3) Last resort: extract relevant sentences
                    final = self._extract_relevant_sentences(question, hits)
                    method = "sentence_extraction"
                    conf = 0.5 if final != "Information not found in the provided documents." else 0.2

            if not final or final.lower().startswith("information not found"):
                final = "The requested information is not available in the uploaded documents."

            # Calculate total response time
            total_time = time.time() - response_start

            # Log metrics
            has_relevant = len(hits) > 0 and max_relevance > 0.3
            self.metrics.log_query(
                response_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                extraction_time=qa_time,
                max_relevance=max_relevance,
                answer_length=len(final),
                method=method,
                confidence=conf,
                has_relevant_results=has_relevant
            )

            # Format response with debug info if enabled
            if show_debug:
                debug_section = self.debug_info.format_debug_info(
                    hits, method, conf, total_time, retrieval_time, generation_time
                )
                final_response = final + debug_section
            else:
                final_response = final

            return "", history + [(question, final_response)]

        except Exception as e:
            total_time = time.time() - response_start
            self.metrics.log_query(total_time, 0.0, 0.0, 0.0, 0.0, 0, "error", 0.0, False)
            return "", history + [(question, f"Error processing question: {e}")]

    def _clean_answer(self, text):
        """Comprehensive answer cleaning"""
        if not text:
            return ""

        # Remove all types of citations and references
        text = re.sub(r'\[[^\]]*\]', '', text)  # [anything]
        text = re.sub(r'\([^)]*\.pdf[^)]*\)', '', text, flags=re.I)  # (anything.pdf)
        text = re.sub(r'[A-Za-z\s]*\.pdf[#\w]*', '', text, flags=re.I)  # filename.pdf references
        text = re.sub(r'#[a-f0-9]+', '', text)  # hash references
        text = re.sub(r'\(relevance\s*[0-9.]+\)', '', text)  # relevance scores
        text = re.sub(r'Method:\s*\w+\s*\|\s*Confidence:\s*[0-9.]+\s*\|\s*[0-9:]+', '', text)  # method info

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove empty parentheses and brackets
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)

        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?')):
            text += "."

        return text

    def _extract_relevant_sentences(self, question, hits):
        """Extract relevant sentences as fallback"""
        q_terms = [w.lower() for w in question.split() if len(w) > 2]
        if not q_terms:
            return "Information not found in the provided documents."

        candidates = []
        for h in hits[:2]:  # Only check top 2 hits
            sentences = safe_sent_tokenize(h["text"])
            for sent in sentences:
                sent_lower = sent.lower()
                # Check if sentence contains multiple query terms
                matches = sum(1 for term in q_terms if term in sent_lower)
                if matches >= min(2, len(q_terms)) and len(sent.split()) > 5:
                    candidates.append((sent.strip(), matches))

        if not candidates:
            return "Information not found in the provided documents."

        # Sort by number of matches and take the best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sentence = candidates[0][0]

        # Clean the sentence
        best_sentence = self._clean_answer(best_sentence)
        return best_sentence if best_sentence else "Information not found in the provided documents."

    def explain_document(self):
        if not self.processed:
            return "No documents have been processed yet. Upload PDF files first."
        combined = "\n\n".join(self.raw_texts.get(doc['file'], '') for doc in self.processed)
        parsed = self.parser.parse(combined)
        lines = ["Document Analysis and Explanation",""]
        pages = sum(d['pages'] for d in self.processed)
        chunks = sum(d['chunks'] for d in self.processed)
        lines.append(f"Overview: processed {len(self.processed)} document(s), {pages} page(s), split into {chunks} chunks.")
        facts = []
        if parsed.get("borrowers"): facts.append("Borrower(s): " + ", ".join(parsed["borrowers"]))
        if parsed.get("product"): facts.append("Product: " + parsed["product"])
        if parsed.get("term_years"): facts.append("Term: " + str(parsed["term_years"]) + " years")
        if parsed.get("rate_percent"): facts.append("Interest rate: " + parsed["rate_percent"] + "%")
        if parsed.get("loan_amount_usd"): facts.append("Loan amount (approx): $" + parsed["loan_amount_usd"])
        if facts:
            lines.append("Key Details:")
            for f in facts: lines.append("- " + f)
        fees = parsed.get("fees", [])
        if fees:
            lines.append("")
            lines.append("Selected Fees:")
            for f in fees[:10]:
                lines.append(f"- {f['name']}: ${f['amount']}")
        if not facts and not fees:
            lines.append("")
            lines.append("The system could not confidently extract structured mortgage details from the uploaded text. Try asking targeted questions such as 'What are the origination charges?' or 'What is the interest rate and term?'.")
        return "\n".join(lines)

    def stats(self):
        if not self.processed:
            return "No documents processed yet."

        # Build components table (plain text, clean)
        components = self.components_table or {}
        table_lines = []
        if components:
            table_lines.append("Components Table")
            widest = max(len(k) for k in components.keys())
            for k,v in components.items():
                table_lines.append(f"- {k:{widest}} : {v}")
            table_lines.append("")

        s = ["Enhanced RAG System Information","",
           f"Documents: {len(self.processed)}", f"Total Chunks: {self.total_chunks}",
           f"Total Pages: {sum(doc['pages'] for doc in self.processed)}",
           f"Text Processed: {self.document_stats.get('total_text_length',0):,} characters",""]
        s.extend(table_lines)

        # Add performance metrics
        if self.metrics.queries_processed > 0:
            s.append("")
            s.extend(self.metrics.get_stats().split('\n'))
            s.append("")

        s.append("Documents:")
        for d in self.processed:
            dist = ", ".join([f"{cnt} {k}" for k,cnt in d['chunk_distribution'].items()])
            ext_time = d.get('extraction_time', 0.0)
            s.append(f"- {d['file']} — {d['chunks']} chunks ({dist}) from {d['pages']} pages [{d.get('extraction_method','unknown')}] ({ext_time:.2f}s)")
        
        # Add section information
        if self.document_sections:
            s.append("")
            s.append("Document Sections:")
            for filename, sections in self.document_sections.items():
                s.append(f"- {filename}: {len(sections)} sections")
                for section_name in sections.keys():
                    s.append(f"  • {section_name.replace('_', ' ').title()}")
        
        return "\n".join(s)

    def get_performance_metrics(self):
        """Return just the performance metrics"""
        return self.metrics.get_stats()

    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset_metrics()
        return "Performance metrics reset."

    # Speech and Language Processing Methods
    def speech_to_text(self):
        """Convert speech to text using microphone"""
        if not self.speech_processor.available:
            return "Speech recognition not available. Please install required packages."
        return self.speech_processor.speech_to_text()
    
    def text_to_speech(self, text, language='en'):
        """Convert text to speech and play it"""
        if not self.speech_processor.available:
            return "Text-to-speech not available. Please install required packages."
        return self.speech_processor.text_to_speech(text, language)
    
    def detect_language(self, text):
        """Detect the language of input text"""
        if not self.bilingual_processor.available:
            return "Language detection not available. Please install required packages."
        lang_code = self.bilingual_processor.detect_language(text)
        lang_name = self.bilingual_processor.get_language_name(lang_code)
        return f"Detected language: {lang_name} ({lang_code})"
    
    def translate_text(self, text, target_lang='en', source_lang='auto'):
        """Translate text to target language"""
        if not self.bilingual_processor.available:
            return "Translation not available. Please install required packages."
        translated = self.bilingual_processor.translate_text(text, target_lang, source_lang)
        target_name = self.bilingual_processor.get_language_name(target_lang)
        return f"Translated to {target_name}: {translated}"
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        if not self.bilingual_processor.available:
            return "Language support not available. Please install required packages."
        languages = []
        for code, name in self.bilingual_processor.supported_languages.items():
            languages.append(f"{code}: {name}")
        return "Supported Languages:\n" + "\n".join(languages)
    
    def answer_with_translation(self, question, history, target_lang='en', k=5, confidence_threshold=0.3, use_expansion=True, max_answer_tokens=0, enable_reranking=True, show_debug=True):
        """Answer question and provide translation"""
        # First, detect the language of the question
        if self.bilingual_processor.available:
            source_lang = self.bilingual_processor.detect_language(question)
            if source_lang != 'en':
                # Translate question to English for processing
                question_en = self.bilingual_processor.translate_text(question, 'en', source_lang)
                print(f"Translated question: {question_en}")
            else:
                question_en = question
        else:
            question_en = question
        
        # Get answer in English
        if target_lang == 'en':
            # Use regular answer method
            return self.answer(question_en, history, k, confidence_threshold, use_expansion, max_answer_tokens, enable_reranking, show_debug)
        else:
            # Get answer and then translate
            empty, new_history = self.answer(question_en, history, k, confidence_threshold, use_expansion, max_answer_tokens, enable_reranking, show_debug)
            
            if new_history and len(new_history) > 0:
                last_question, last_answer = new_history[-1]
                
                # Translate the answer to target language
                if self.bilingual_processor.available:
                    translated_answer = self.bilingual_processor.translate_text(last_answer, target_lang, 'en')
                    target_name = self.bilingual_processor.get_language_name(target_lang)
                    
                    # Create bilingual response
                    bilingual_response = f"[{target_name} Translation]\n{translated_answer}\n\n[Original English]\n{last_answer}"
                    
                    # Update the last answer with bilingual response
                    new_history[-1] = (last_question, bilingual_response)
            
            return "", new_history

    def clear(self):
        self.__init__()
        return ("System cleared. Upload PDFs to start again.", [], "Ready.")

    # Export Methods
    def export_conversation(self, conversation_history, format_type="pdf", filename=None):
        """Export conversation to specified format"""
        return self.export_manager.export_conversation(conversation_history, format_type, filename)

    def get_export_formats(self):
        """Get available export formats"""
        return self.export_manager.get_export_formats()

    # Error Recovery Methods
    def handle_error_with_recovery(self, error, context="", retry_count=0):
        """Handle errors with recovery strategies"""
        return self.error_recovery.handle_error(error, context, retry_count)
    
    def get_error_statistics(self):
        """Get error statistics"""
        return self.error_recovery.get_error_stats()
    
    def reset_error_counts(self):
        """Reset error counts"""
        self.error_recovery.reset_error_counts()
