import os
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
import pandas as pd
import numpy as np


class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = {'.pdf', '.docx', '.xlsx', '.xls', '.txt'}
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        if extension == '.pdf':
            return self._process_pdf(file_path)
        elif extension == '.docx':
            return self._process_docx(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._process_excel(file_path)
        elif extension == '.txt':
            return self._process_txt(file_path)
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        text_chunks = []
        metadata = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text and text.strip():
                            chunks = self._chunk_text(text, chunk_size=500, overlap=50)
                            for i, chunk in enumerate(chunks):
                                if chunk.strip():
                                    text_chunks.append(chunk)
                                    metadata.append({
                                        'source': os.path.basename(file_path),
                                        'page': page_num + 1,
                                        'chunk': i,
                                        'type': 'pdf'
                                    })
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                        continue
            
            if not text_chunks:
                raise ValueError(f"No text could be extracted from PDF: {file_path}")
            
            return {
                'chunks': text_chunks,
                'metadata': metadata,
                'num_pages': num_pages,
                'source': os.path.basename(file_path)
            }
        except Exception as e:
            raise ValueError(f"Error processing PDF {file_path}: {str(e)}")
    
    def _process_docx(self, file_path: str) -> Dict[str, Any]:
        text_chunks = []
        metadata = []
        
        try:
            doc = Document(file_path)
            full_text = []
            
            for para in doc.paragraphs:
                if para.text and para.text.strip():
                    full_text.append(para.text)
            
            if not full_text:
                raise ValueError(f"No text content found in DOCX: {file_path}")
            
            combined_text = '\n'.join(full_text)
            chunks = self._chunk_text(combined_text, chunk_size=500, overlap=50)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    text_chunks.append(chunk)
                    metadata.append({
                        'source': os.path.basename(file_path),
                        'chunk': i,
                        'type': 'docx'
                    })
            
            return {
                'chunks': text_chunks,
                'metadata': metadata,
                'num_paragraphs': len(full_text),
                'source': os.path.basename(file_path)
            }
        except Exception as e:
            raise ValueError(f"Error processing DOCX {file_path}: {str(e)}")
    
    def _process_excel(self, file_path: str) -> Dict[str, Any]:
        text_chunks = []
        metadata = []
        
        df = pd.read_excel(file_path, sheet_name=None)
        
        for sheet_name, sheet_df in df.items():
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += sheet_df.to_string(index=False)
            
            chunks = self._chunk_text(sheet_text, chunk_size=500, overlap=50)
            
            for i, chunk in enumerate(chunks):
                text_chunks.append(chunk)
                metadata.append({
                    'source': os.path.basename(file_path),
                    'sheet': sheet_name,
                    'chunk': i,
                    'type': 'excel'
                })
        
        return {
            'chunks': text_chunks,
            'metadata': metadata,
            'num_sheets': len(df),
            'source': os.path.basename(file_path)
        }
    
    def _process_txt(self, file_path: str) -> Dict[str, Any]:
        text_chunks = []
        metadata = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        chunks = self._chunk_text(text, chunk_size=500, overlap=50)
        
        for i, chunk in enumerate(chunks):
            text_chunks.append(chunk)
            metadata.append({
                'source': os.path.basename(file_path),
                'chunk': i,
                'type': 'txt'
            })
        
        return {
            'chunks': text_chunks,
            'metadata': metadata,
            'source': os.path.basename(file_path)
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks if chunks else [text]
