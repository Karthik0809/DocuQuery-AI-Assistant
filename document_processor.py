# Document processing module for Enhanced RAG Chatbot
import os
import re
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from config import DEFAULT_DPI, OCR_CONFIG

class PDFSectionSegmenter:
    """Segments PDF documents into logical sections for better organization"""
    
    def __init__(self):
        self.section_keywords = {
            "lender_sheet": ["lender", "loan estimate", "loan terms", "interest rate", "apr", "loan amount"],
            "closing_costs": ["closing costs", "closing disclosure", "settlement costs", "total closing costs"],
            "fees_worksheet": ["fees worksheet", "origination charges", "services borrower did not shop for", "services borrower did shop for"],
            "bills": ["bill", "invoice", "statement", "amount due", "payment due"],
            "contract": ["contract", "agreement", "terms and conditions", "witness whereof"],
            "disclosures": ["disclosure", "notice", "important information", "federal law"]
        }
    
    def segment_document(self, text, filename):
        """Segment document into logical sections"""
        sections = {}
        text_lower = text.lower()
        
        # Split by page boundaries first
        pages = text.split("--- Page")
        
        for section_name, keywords in self.section_keywords.items():
            section_content = []
            section_pages = []
            
            for i, page in enumerate(pages):
                if i == 0:  # Skip the first empty part
                    continue
                    
                page_lower = page.lower()
                # Check if this page contains section keywords
                keyword_matches = sum(1 for keyword in keywords if keyword in page_lower)
                
                if keyword_matches >= 2:  # At least 2 keywords must match
                    section_content.append(page)
                    section_pages.append(i)
            
            if section_content:
                sections[section_name] = {
                    "content": "\n".join(section_content),
                    "pages": section_pages,
                    "filename": filename,
                    "keyword_matches": len(keywords)
                }
        
        # If no specific sections found, create general sections by page ranges
        if not sections and len(pages) > 1:
            sections["general"] = {
                "content": text,
                "pages": list(range(1, len(pages))),
                "filename": filename,
                "keyword_matches": 0
            }
        
        return sections
    
    def get_section_summary(self, sections):
        """Generate a summary of available sections"""
        if not sections:
            return "No sections detected in the document."
        
        summary = "Document Sections Available:\n\n"
        for section_name, section_info in sections.items():
            pages = section_info["pages"]
            page_range = f"Pages {min(pages)}-{max(pages)}" if len(pages) > 1 else f"Page {pages[0]}"
            summary += f"**{section_name.replace('_', ' ').title()}** ({page_range})\n"
            summary += f"   Contains: {section_info['keyword_matches']} relevant keywords\n\n"
        
        return summary

class EnhancedDocumentProcessor:
    def __init__(self, dpi=DEFAULT_DPI): 
        self.dpi = dpi
        self.segmenter = PDFSectionSegmenter()
    
    def extract_text(self, path: str):
        try:
            text, pages, structure = self._extract_digital_enhanced(path)
            if len(text.strip()) > 200 and not self._looks_scanned(text):
                # Segment the document
                sections = self.segmenter.segment_document(text, os.path.basename(path))
                return text, {"file": os.path.basename(path), "path": path, "total_pages": pages,
                              "extraction_method": "digital_enhanced", "structure": structure, "sections": sections}
        except Exception as e:
            print("Enhanced digital extract failed:", e)
        
        try:
            text, pages = self._extract_ocr_enhanced(path)
            # Segment the OCR text as well
            sections = self.segmenter.segment_document(text, os.path.basename(path))
            return text, {"file": os.path.basename(path), "path": path, "total_pages": pages,
                          "extraction_method": "ocr_enhanced", "structure": {"sections":[],"tables":[],"lists":[]}, "sections": sections}
        except Exception as e:
            return f"Enhanced OCR failed: {e}", {"extraction_method":"failed","path":path}
    
    def _extract_digital_enhanced(self, path):
        reader = PdfReader(path); parts=[]; structure={"sections":[],"tables":[],"lists":[]}
        for i,p in enumerate(reader.pages):
            t = p.extract_text() or ""
            self._detect_structure(t, i+1, structure)
            parts.append(f"\n--- Page {i+1} ---\n{self._clean_text(t)}")
        return "\n".join(parts), len(reader.pages), structure
    
    def _extract_ocr_enhanced(self, path):
        images = convert_from_path(path, dpi=self.dpi); parts=[]
        # Enhanced OCR configuration for better accuracy
        
        for i,im in enumerate(images):
            # Preprocess image for better OCR
            im_enhanced = self._enhance_image_for_ocr(im)
            txt = pytesseract.image_to_string(im_enhanced, lang="eng", config=OCR_CONFIG)
            # Post-process OCR text
            txt = self._post_process_ocr_text(txt)
            parts.append(f"\n--- Page {i+1} (OCR) ---\n{self._clean_text(txt)}")
        return "\n".join(parts), len(images)
    
    def _enhance_image_for_ocr(self, image):
        """Enhance image for better OCR accuracy"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.3)
        
        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return image
    
    def _post_process_ocr_text(self, text):
        """Post-process OCR text to improve accuracy"""
        # Fix common OCR mistakes
        replacements = {
            '0': 'O', 'O': '0',  # Common confusion
            '1': 'l', 'l': '1',  # Common confusion
            'rn': 'm', 'm': 'rn',  # Common confusion
            'cl': 'd', 'd': 'cl',  # Common confusion
        }
        
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        # Fix spacing issues
        text = re.sub(r'(\d)\s*([A-Za-z])', r'\1 \2', text)  # Add space between numbers and letters
        text = re.sub(r'([A-Za-z])\s*(\d)', r'\1 \2', text)  # Add space between letters and numbers
        
        return text
    
    def _clean_text(self, text):
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text, flags=re.I)
        return text.strip()
    
    def _detect_structure(self, text, page, structure):
        for i,line in enumerate((text or "").split('\n')):
            s = line.strip()
            if not s: continue
            if len(s) < 120 and (s.isupper() or re.match(r'^\d+\.\s+[A-Z]', s)):
                structure["sections"].append({"title": s, "page": page, "line": i})
            if re.match(r'^\s*[•\-*]|^\s*\d+\.', s):
                structure["lists"].append({"content": s, "page": page, "line": i})
    
    def _looks_scanned(self, text):
        if len(text.split()) < 100: return True
        indicators = ['|','_','~','`',' rn ',' l1 ','0O']
        return sum(text.count(i) for i in indicators) > len(text) * 0.05

class MortgageParser:
    FEE_PAT = re.compile(r"([A-Za-z][A-Za-z ]{2,}?Fee)[^$\n]*\$\s*([\d,]+(?:\.\d{2})?)")
    RATE_PAT = re.compile(r"(\d{1,2}\.\d{2,3})\s*%")
    TERM_PAT = re.compile(r"(\d{2,3})\s*YEAR\s*(FIXED|ARM)", re.I)
    LOAN_AMT_PAT = re.compile(r"\$\s*([\d,]{3,})(?:\.\d{2})?")
    PRODUCT_PAT = re.compile(r"(\d{2,3}\s*YEAR\s*FIXED|ARM|ADJUSTABLE)\b", re.I)
    
    def parse(self, text: str):
        out={}
        prod = self.PRODUCT_PAT.search(text)
        if prod: out["product"] = prod.group(0).upper().replace("  "," ")
        rate = self.RATE_PAT.search(text)
        if rate: out["rate_percent"] = rate.group(1)
        term = self.TERM_PAT.search(text)
        if term: out["term_years"] = term.group(1)
        amts = [int(a.replace(',','')) for a in self.LOAN_AMT_PAT.findall(text)]
        if amts:
            guess = max(a for a in amts if 50_000 <= a <= 5_000_000) if any(50_000<=a<=5_000_000 for a in amts) else max(amts)
            out["loan_amount_usd"] = f"{guess:,}"
        name_line = next((ln for ln in text.split('\n') if '/' in ln and any(x.strip() for x in ln.split('/'))), "")
        if name_line:
            out["borrowers"] = [n.strip() for n in name_line.split('/') if n.strip() and len(n.strip())>2][:3]
        fees=[{"name": m.group(1).strip(), "amount": m.group(2)} for m in self.FEE_PAT.finditer(text)]
        if fees: out["fees"] = fees
        return out
