from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from typing import Dict, List, Optional

class CVParserAgent:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize CV Parser with configurable chunk settings
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def update_chunk_settings(self, chunk_size: int, chunk_overlap: int):
        """Update chunking parameters dynamically"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def extract_name_from_text(self, text: str, email: str = None) -> Optional[str]:
        """Extract candidate name using multiple strategies"""

        # Strategy 1: Look for name patterns at the beginning of text
        # Common pattern: Name appears in first 200 characters, often as Title Case words
        first_part = text[:500]

        # Strategy 2: If we have email, extract name from email prefix
        if email:
            email_prefix = email.split('@')[0]
            # Common patterns: john.doe, johndoe, john_doe, john-doe
            name_from_email = email_prefix.replace('.', ' ').replace('_', ' ').replace('-', ' ')
            # Filter out common non-name prefixes
            non_name_prefixes = ['info', 'contact', 'admin', 'hr', 'jobs', 'career', 'resume', 'cv', 'apply']
            if name_from_email.lower() not in non_name_prefixes and len(name_from_email) > 2:
                # Title case the name
                name_parts = name_from_email.split()
                if len(name_parts) >= 1 and all(len(p) > 1 for p in name_parts):
                    # Check if it could be a name (not just numbers)
                    if any(c.isalpha() for c in name_from_email):
                        return ' '.join(word.capitalize() for word in name_parts)

        # Strategy 3: Look for capitalized name pattern (2-4 words, each starting with capital)
        # Pattern: First Last or First Middle Last
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        name_matches = re.findall(name_pattern, first_part)

        if name_matches:
            # Filter out common non-name matches
            exclude_words = ['resume', 'curriculum', 'vitae', 'profile', 'summary', 'objective',
                           'experience', 'education', 'skills', 'contact', 'phone', 'email',
                           'address', 'professional', 'personal', 'information', 'details',
                           'work', 'history', 'career', 'technical', 'software', 'engineer',
                           'developer', 'manager', 'senior', 'junior', 'lead', 'project',
                           'data', 'analyst', 'scientist', 'designer', 'full', 'stack']

            for match in name_matches:
                match_lower = match.lower()
                # Skip if it contains excluded words
                if not any(ex in match_lower for ex in exclude_words):
                    # Likely a name if it's 2-4 words, each 2-15 chars
                    words = match.split()
                    if 2 <= len(words) <= 4 and all(2 <= len(w) <= 15 for w in words):
                        return match

        # Strategy 4: Look for ALL CAPS name at the beginning
        caps_pattern = r'^[A-Z\s]{4,40}(?=\s|$)'
        caps_match = re.search(caps_pattern, first_part.strip())
        if caps_match:
            potential_name = caps_match.group().strip()
            # Check if it's not a header
            if potential_name.lower() not in ['resume', 'cv', 'curriculum vitae']:
                # Title case it
                return potential_name.title()

        return None

    def extract_structured_info(self, text: str, raw_text: str = None) -> Dict:
        """Extract structured information from CV text"""
        info = {
            'name': None,
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None,
            'skills': [],
            'education': [],
            'experience_years': None
        }

        # Extract email first (needed for name extraction)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            info['email'] = emails[0]

        # Extract name using the dedicated method
        # Use raw_text if available (preserves original formatting), otherwise use cleaned text
        name_source = raw_text if raw_text else text
        info['name'] = self.extract_name_from_text(name_source, info['email'])

        # Extract phone
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        phones = re.findall(phone_pattern, text)
        if phones:
            info['phone'] = phones[0]
        
        # Extract LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_matches:
            info['linkedin'] = linkedin_matches[0]
        
        # Extract GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_matches = re.findall(github_pattern, text, re.IGNORECASE)
        if github_matches:
            info['github'] = github_matches[0]
        
        # Extract years of experience (simple heuristic)
        year_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?experience'
        year_matches = re.findall(year_pattern, text, re.IGNORECASE)
        if year_matches:
            info['experience_years'] = int(year_matches[0])
        
        # Extract skills (common programming languages and tools)
        skill_keywords = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'SQL', 'R',
            'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring',
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'Jenkins',
            'Machine Learning', 'Deep Learning', 'Data Science', 'AI', 'NLP',
            'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
            'Agile', 'Scrum', 'DevOps', 'CI/CD', 'REST', 'GraphQL', 'MongoDB'
        ]
        
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                info['skills'].append(skill)
        
        return info

    def create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks based on CV sections"""
        sections = {
            'summary': [],
            'experience': [],
            'education': [],
            'skills': [],
            'projects': [],
            'other': []
        }
        
        # Common section headers
        section_patterns = {
            'summary': r'(?:professional\s*)?(?:summary|objective|profile|about)',
            'experience': r'(?:work\s*)?(?:experience|employment|history|career)',
            'education': r'(?:education|academic|qualification|degree)',
            'skills': r'(?:technical\s*)?(?:skills|competenc|expertise|technologies)',
            'projects': r'(?:projects|portfolio|achievements)'
        }
        
        lines = text.split('\n')
        current_section = 'other'
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            section_found = False
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    # Save previous section content
                    if current_content:
                        sections[current_section].append('\n'.join(current_content))
                    # Start new section
                    current_section = section
                    current_content = [line]
                    section_found = True
                    break
            
            if not section_found and line.strip():
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section].append('\n'.join(current_content))
        
        # Combine all sections into semantic chunks
        semantic_chunks = []
        for section_name, content_list in sections.items():
            if content_list:
                section_text = '\n'.join(content_list)
                if len(section_text) > self.chunk_size:
                    # Further split large sections
                    sub_chunks = self.splitter.split_text(section_text)
                    for chunk in sub_chunks:
                        semantic_chunks.append(f"[{section_name.upper()}]\n{chunk}")
                else:
                    semantic_chunks.append(f"[{section_name.upper()}]\n{section_text}")
        
        return semantic_chunks if semantic_chunks else self.splitter.split_text(text)

    def parse_cv(self, pdf_path: str, use_semantic_chunking: bool = True) -> Dict:
        """
        Parse CV from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            use_semantic_chunking: Whether to use semantic chunking based on CV sections
        
        Returns:
            Dictionary containing parsed text, chunks, and structured info
        """
        try:
            reader = PdfReader(pdf_path)
            raw_text = " ".join([page.extract_text() for page in reader.pages])

            # Keep raw text for name extraction (preserves some formatting)
            raw_text_for_name = raw_text

            # Clean up text
            text = re.sub(r'\s+', ' ', raw_text)  # Normalize whitespace
            text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters

            # Extract structured information (pass raw text for better name extraction)
            structured_info = self.extract_structured_info(text, raw_text_for_name)
            
            # Create chunks
            if use_semantic_chunking:
                chunks = self.create_semantic_chunks(text)
            else:
                chunks = self.splitter.split_text(text)
            
            return {
                "text": text,
                "chunks": chunks,
                "structured_info": structured_info,
                "num_pages": len(reader.pages),
                "chunk_method": "semantic" if use_semantic_chunking else "standard"
            }
        except Exception as e:
            raise Exception(f"Failed to parse PDF: {str(e)}")
