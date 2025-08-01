import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        print(f"Extracting text from {len(reader.pages)} pages...")
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1} pages...")
        
        print(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def preprocess_medical_text(text: str) -> str:
    """Advanced preprocessing for medical text"""
    print("Preprocessing medical text...")
    
    # Remove headers, footers, and page numbers
    text = re.sub(r'Page \d+.*?\n', '', text)
    text = re.sub(r'Chapter \d+.*?\n', '', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Normalize medical abbreviations
    medical_abbrevs = {
        'mg/dl': 'milligrams per deciliter',
        'mmHg': 'millimeters of mercury',
        'CBC': 'complete blood count',
        'BP': 'blood pressure',
        'HR': 'heart rate',
        'RBC': 'red blood cells',
        'WBC': 'white blood cells',
        'ECG': 'electrocardiogram',
        'EKG': 'electrocardiogram',
        'MRI': 'magnetic resonance imaging',
        'CT': 'computed tomography',
        'ICU': 'intensive care unit',
        'ER': 'emergency room',
        'IV': 'intravenous',
        'IM': 'intramuscular',
        'PO': 'by mouth',
        'PRN': 'as needed',
        'BID': 'twice daily',
        'TID': 'three times daily',
        'QID': 'four times daily'
    }
    
    for abbrev, full_form in medical_abbrevs.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full_form, text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace and clean formatting
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove special characters but preserve medical symbols
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\%\°\+\-\/]', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    print(f"Preprocessing complete. Cleaned text length: {len(text)}")
    return text.strip()

def create_smart_chunks(text: str) -> List[Dict]:
    """Create intelligent chunks optimized for medical content"""
    print("Creating intelligent chunks...")
    
    # Create text splitter optimized for medical content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # Smaller chunks for better precision
        chunk_overlap=150,        # Good overlap to maintain context
        length_function=len,
        separators=[
            "\n\n",               # Paragraph breaks (highest priority)
            "\n",                 # Line breaks
            ". ",                 # Sentence endings
            "? ",                 # Question endings
            "! ",                 # Exclamation endings
            "; ",                 # Semicolon breaks
            ", ",                 # Comma breaks
            " ",                  # Word breaks
            ""                    # Character breaks (last resort)
        ]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Process and enhance chunks
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        # Clean chunk
        chunk = chunk.strip()
        
        # Skip very short chunks
        if len(chunk) < 50:
            continue
        
        # Create enhanced chunk with metadata
        enhanced_chunk = {
            'text': chunk,
            'metadata': {
                'chunk_id': i,
                'source': 'medical_encyclopedia',
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'has_medical_terms': detect_medical_terms(chunk)
            }
        }
        enhanced_chunks.append(enhanced_chunk)
    
    print(f"Created {len(enhanced_chunks)} intelligent chunks")
    return enhanced_chunks

def detect_medical_terms(text: str) -> bool:
    """Detect if chunk contains medical terminology"""
    medical_keywords = [
        'symptom', 'disease', 'treatment', 'medicine', 'diagnosis',
        'patient', 'doctor', 'hospital', 'pain', 'fever', 'infection',
        'medical', 'health', 'illness', 'condition', 'therapy',
        'prescription', 'medication', 'surgery', 'procedure', 'test',
        'blood', 'heart', 'lung', 'brain', 'liver', 'kidney'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in medical_keywords)

def process_pdf_complete(pdf_path: str) -> List[Dict]:
    """Complete PDF processing pipeline"""
    print(f"Starting complete PDF processing for: {pdf_path}")
    
    # Step 1: Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return []
    
    # Step 2: Preprocess text
    cleaned_text = preprocess_medical_text(raw_text)
    if not cleaned_text:
        return []
    
    # Step 3: Create chunks
    chunks = create_smart_chunks(cleaned_text)
    
    print(f"PDF processing complete! Generated {len(chunks)} chunks")
    return chunks
