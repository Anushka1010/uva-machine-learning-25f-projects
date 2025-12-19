"""
File processing utilities for different document formats.

Handles extraction of text from:
- PDF files
- Plain text files
- Markdown files
- DOCX files (optional)
"""

import PyPDF2
import markdown
from pathlib import Path
from typing import Optional

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class FileProcessor:
    """Extract text from various document formats."""
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file format is unsupported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return FileProcessor._extract_pdf(file_path)
        elif extension == '.txt':
            return FileProcessor._extract_txt(file_path)
        elif extension in ['.md', '.markdown']:
            return FileProcessor._extract_markdown(file_path)
        elif extension == '.docx':
            if not DOCX_AVAILABLE:
                raise ValueError("DOCX support requires python-docx library")
            return FileProcessor._extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    
    @staticmethod
    def _extract_txt(file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    @staticmethod
    def _extract_markdown(file_path: str) -> str:
        """
        Extract text from Markdown file.
        
        Converts markdown to plain text while preserving structure.
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            md_content = file.read()
            
        # Convert markdown to HTML, then strip HTML tags for plain text
        # For RAG purposes, we want the content without markdown syntax
        html = markdown.markdown(md_content)
        
        # Simple HTML tag removal (good enough for this project)
        import re
        text = re.sub('<[^<]+?>', '', html)
        
        return text
    
    @staticmethod
    def _extract_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        return '\n\n'.join(text_parts)
