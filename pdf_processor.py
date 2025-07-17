import os
import logging
import PyPDF2
import re
import fitz  # PyMuPDF
import difflib
import requests
from bs4 import BeautifulSoup
from pathlib import Path

logger = logging.getLogger("PdfProcessor")

class PdfProcessor:
    def __init__(self, grobid_url="http://localhost:8070/api/processFulltextDocument"):
        self.max_content_length = 100000
        self.grobid_url = grobid_url
        
        self.section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background'],
            'methods': ['method', 'methods', 'materials and methods', 'methodology', 'experimental', 'procedure'],
            'results': ['result', 'results', 'findings', 'observations'],
            'discussion': ['discussion', 'general discussion', 'interpretation'],
            'conclusion': ['conclusion', 'conclusions', 'summary', 'final remarks', 'future work', 'future direction']
        }
        
        self.section_mapping = {
            'abstract': 'abstract',
            'summary': 'abstract',
            'introduction': 'introduction',
            'background': 'introduction',
            'methods': 'methods',
            'materials and methods': 'methods',
            'methodology': 'methods',
            'experimental': 'methods',
            'materials': 'methods',
            'procedure': 'methods',
            'results': 'results',
            'findings': 'results',
            'observations': 'results',
            'discussion': 'discussion',
            'general discussion': 'discussion',
            'interpretation': 'discussion',
            'conclusion': 'conclusion',
            'conclusions': 'conclusion',
            'concluding remarks': 'conclusion',
            'final remarks': 'conclusion',
            'future work': 'conclusion',
            'future directions': 'conclusion',
        }
    
    def extract_full_text(self, pdf_path):
        try:
            logger.info(f"Starting PDF extraction from {pdf_path}")
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                total_pages = len(reader.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                max_pages = min(30, total_pages)
                for page_num in range(max_pages):
                    logger.info(f"Extracting page {page_num+1}/{max_pages}")
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
                    
                    if len(text) > self.max_content_length:
                        text += "\n...[Content truncated due to length]..."
                        break
            
            logger.info(f"PDF extraction complete. Total text: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def extract_sections(self, pdf_path):
        try:
            logger.info(f"Extracting sections from {pdf_path} using GROBID")
            sections = self._extract_sections_grobid(pdf_path)
            
            if sections:
                for section_name, content in sections.items():
                    logger.info(f"Found {section_name} (GROBID): {len(content)} characters")
                return sections
            
            logger.info("GROBID extraction failed, falling back to PyMuPDF")
            sections = self._extract_sections_pymupdf(pdf_path)
            
            if sections:
                for section_name, content in sections.items():
                    logger.info(f"Found {section_name} (PyMuPDF): {len(content)} characters")
                return sections
            
            logger.info("PyMuPDF extraction failed, falling back to regex")
            sections = self._extract_sections_regex(pdf_path)
            
            if sections:
                for section_name, content in sections.items():
                    logger.info(f"Found {section_name} (regex): {len(content)} characters")
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in primary section extraction: {e}")
            import traceback
            traceback.print_exc()
            
            logger.info("Falling back to regex-based section extraction")
            return self._extract_sections_regex(pdf_path)
            
    def _extract_sections_grobid(self, pdf_path):
        """Extract sections using GROBID service."""
        try:
            pdf_path = Path(pdf_path)
            
            # Check if file exists and is accessible
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return {}
            
            # Open the PDF and send to GROBID
            with pdf_path.open("rb") as f:
                files = {"input": (pdf_path.name, f, "application/pdf")}
                params = {"teiCoordinates": "ref,biblStruct", "includePdfalto": "true"}
                
                # Send request to GROBID with a timeout
                logger.info(f"Sending PDF to GROBID: {pdf_path.name}")
                response = requests.post(
                    self.grobid_url, 
                    files=files, 
                    params=params, 
                    timeout=20000
                )
                
                # Check if request was successful
                if response.status_code != 200:
                    logger.error(f"GROBID request failed with status {response.status_code}")
                    return {}
                
                # Get XML content
                tei_xml = response.text
            
            # Parse XML with BeautifulSoup - use 'xml' parser and handle namespaces
            soup = BeautifulSoup(tei_xml, "xml")
            
            # Dictionary to hold all extracted sections
            sections = {}
            
            # Get abstract separately (it's in a different location in TEI)
            abstract_tag = soup.find("abstract")
            if abstract_tag:
                abstract = " ".join(p.get_text(" ", strip=True) for p in abstract_tag.find_all("p"))
                sections['abstract'] = abstract
            
            # Extract all divs from body recursively
            body = soup.find("body")
            if body:
                # Process each div in the body, including nested divs
                self._process_body_divs(body, sections)
                
                # If no sections were found but we have body text, extract the full text
                if len(sections) <= 1:  # Only abstract or empty
                    full_text = body.get_text(" ", strip=True)
                    if full_text:
                        # Use simple section extraction - don't rename sections
                        text_sections = self._break_text_into_sections_raw(full_text)
                        sections.update(text_sections)
            
            logger.info(f"GROBID extracted {len(sections)} sections")
            for section_name in sections.keys():
                logger.info(f"  Found section: {section_name}")
            
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections with GROBID: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _process_body_divs(self, element, sections, parent_path=""):
        """
        Recursively process divs in the document body to extract all sections and subsections.
        
        Args:
            element: Current XML element being processed
            sections: Dictionary to store extracted sections
            parent_path: Path of parent sections (for subsections)
        """
        # Process div elements (sections)
        for div in element.find_all("div", recursive=False):
            # Get section title
            head_tag = div.find("head")
            title = head_tag.get_text(" ", strip=True) if head_tag else "UNTITLED"
            
            # Clean up title if needed (keep exact wording but clean formatting)
            title = title.strip()
            
            # Get section content (all text from paragraphs)
            paragraphs = []
            
            # Extract all paragraphs directly in this div
            for p in div.find_all("p", recursive=False):
                paragraphs.append(" ".join(p.stripped_strings))
            
            # Get section content
            section_text = " ".join(paragraphs)
            
            # Add this section if it has content
            if section_text.strip():
                # Keep original section name - don't normalize
                section_name = title.lower() if title else "untitled"
                
                # If this section name already exists, append a suffix
                base_name = section_name
                suffix = 1
                while section_name in sections:
                    section_name = f"{base_name}_{suffix}"
                    suffix += 1
                
                # Add to sections dictionary
                sections[section_name] = section_text
                logger.debug(f"Extracted section: {section_name} ({len(section_text)} chars)")
            
            # Recursively process nested divs (subsections)
            self._process_body_divs(div, sections, title)
    
    def _break_text_into_sections_raw(self, text):
        """
        Attempt to break a continuous text into sections heuristically,
        keeping original section names as they appear in the text.
        """
        sections = {}
        
        # Common section pattern: title followed by content
        section_pattern = r'(?i)(?:^|\n)([A-Z][A-Za-z\s]+)(?:\n|:)(.*?)(?=(?:^|\n)[A-Z][A-Za-z\s]+(?:\n|:)|\Z)'
        matches = re.finditer(section_pattern, text, re.DOTALL | re.MULTILINE)
        
        for match in matches:
            title = match.group(1).strip()
            content = match.group(2).strip()
            
            # Keep original title, just convert to lowercase for dictionary key consistency
            section_name = title.lower()
            
            # Skip very short content (likely false matches)
            if len(content) < 50:
                continue
            
            # Handle duplicate section names
            base_name = section_name
            suffix = 1
            while section_name in sections:
                section_name = f"{base_name}_{suffix}"
                suffix += 1
            
            sections[section_name] = content
        
        # If we failed to find any sections, just return the whole text as one section
        if not sections and len(text) > 100:
            sections['text'] = text
        
        return sections
    
    def _normalize_section_name(self, section_title):
        """Normalize section name to standard categories."""
        if not section_title:
            return None
            
        # Convert to lowercase for matching
        title_lower = section_title.lower()
        
        # Direct mapping if exists
        if title_lower in self.section_mapping:
            return self.section_mapping[title_lower]
        
        # Check if title contains any of our keywords
        for section_type, keywords in self.section_keywords.items():
            # Exact match with any keyword
            for keyword in keywords:
                if keyword == title_lower:
                    return section_type
                    
            # Title contains keyword
            for keyword in keywords:
                if keyword in title_lower:
                    return section_type
        
        # Check numeric section titles (like "1. Introduction")
        section_pattern = r'^\d+\.?\s+(.+)$'
        match = re.match(section_pattern, title_lower)
        if match:
            actual_title = match.group(1).strip()
            # Recursive call with extracted title
            return self._normalize_section_name(actual_title)
        
        # If no match, return None
        return None
    
    def _break_text_into_sections(self, text):
        """
        Attempt to break a continuous text into sections heuristically.
        Used when GROBID returns full text but fails to identify sections.
        """
        sections = {}
        
        # Try to find abstract
        abstract_match = re.search(r'(?i)abstract[\s\n:]*(.+?)(?=introduction|keywords|methods|\d+\.|\n\n\n)', 
                                 text, re.DOTALL)
        if abstract_match:
            sections['abstract'] = abstract_match.group(1).strip()
            # Remove abstract from text for further processing
            text = text[abstract_match.end():]
        
        # Try to find other sections using regex
        section_patterns = {
            'introduction': r'(?i)(introduction|background)[\s\n:]*',
            'methods': r'(?i)(methods|materials|methodology|experimental)[\s\n:]*',
            'results': r'(?i)(results|findings)[\s\n:]*',
            'discussion': r'(?i)(discussion|general\s+discussion)[\s\n:]*',
            'conclusion': r'(?i)(conclusion|conclusions|summary|final\s+remarks)[\s\n:]*'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text)
            if match:
                start_pos = match.end()
                
                # Find next section start
                next_section_match = None
                for next_pattern in section_patterns.values():
                    next_match = re.search(next_pattern, text[start_pos:])
                    if next_match and (not next_section_match or next_match.start() < next_section_match.start()):
                        next_section_match = next_match
                
                # Extract section content
                if next_section_match:
                    end_pos = start_pos + next_section_match.start()
                    sections[section_name] = text[start_pos:end_pos].strip()
                else:
                    # Take the rest of the text if no next section
                    sections[section_name] = text[start_pos:].strip()
        
        return sections
    
    def _extract_sections_pymupdf(self, pdf_path):
        """Extract document sections using PyMuPDF's structural analysis capabilities."""
        try:
            logger.info(f"Extracting sections from {pdf_path} using PyMuPDF")
            
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # First detect all potential section headers based on font size and formatting
            potential_headers = self._detect_potential_headers(doc)
            
            # Match detected headers to standard section types
            identified_sections = self._match_headers_to_sections(potential_headers)
            
            # Extract text between identified section headers
            sections = self._extract_section_contents(doc, identified_sections)
            
            # Keep only non-empty sections
            sections = {k: v for k, v in sections.items() if v}
            
            return sections
        
        except Exception as e:
            logger.error(f"Error extracting sections with PyMuPDF: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def _detect_potential_headers(self, doc):
        """Detect potential section headers based on font properties."""
        headers = []
        
        # Calculate the most common font size (body text)
        font_sizes = []
        for page_num in range(min(10, doc.page_count)):  # Check first 10 pages or less
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
        
        # Get the most common font size (likely the body text size)
        if font_sizes:
            body_font_size = self._most_common(font_sizes)
            logger.info(f"Detected body font size: {body_font_size}")
        else:
            body_font_size = 10  # Fallback if we can't detect
        
        # Now find headers (larger font, often bold, fewer words)
        for page_num in range(min(30, doc.page_count)):  # Check first 30 pages or all if fewer
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        is_potential_header = False
                        
                        # Check spans in this line
                        for span in line["spans"]:
                            # If font is significantly larger than body text or is bold
                            if (span["size"] > body_font_size * 1.1 or 
                                "bold" in span["font"].lower() or 
                                span["text"].isupper()):
                                
                                # And number of words is less than 12 (typical for headers)
                                words = span["text"].split()
                                if len(words) < 12 and len(span["text"]) > 3:
                                    is_potential_header = True
                            
                            line_text += span["text"] + " "
                        
                        line_text = line_text.strip()
                        
                        # Additional checks to filter out false positives
                        if is_potential_header and line_text and not line_text.endswith(":"):
                            # Check if text is all lowercase (unlikely to be a header)
                            if not line_text.islower() and len(line_text.split()) < 12:
                                headers.append({
                                    "text": line_text,
                                    "page": page_num,
                                    "y": line["bbox"][1]  # Y-coordinate for ordering
                                })
        
        # Sort headers by page and y-coordinate
        headers.sort(key=lambda h: (h["page"], h["y"]))
        
        logger.info(f"Detected {len(headers)} potential section headers")
        return headers
        
    def _most_common(self, lst):
        """Find the most common element in a list."""
        return max(set(lst), key=lst.count)
        
    def _match_headers_to_sections(self, potential_headers):
        """Match detected headers to standard section types."""
        identified_sections = []
        
        for header in potential_headers:
            header_text = header["text"].lower()
            
            # Skip headers with numbers only or very short text
            if header_text.isdigit() or len(header_text) < 3:
                continue
                
            # Check for each section type
            matched_section = None
            highest_similarity = 0
            
            for section_type, keywords in self.section_keywords.items():
                # Check for exact matches first
                if any(keyword == header_text for keyword in keywords):
                    matched_section = section_type
                    highest_similarity = 1.0
                    break
                
                # Then check for keywords contained in the header
                if any(keyword in header_text for keyword in keywords):
                    similarity = 0.8
                    if similarity > highest_similarity:
                        matched_section = section_type
                        highest_similarity = similarity
                
                # Finally check for close similarity
                for keyword in keywords:
                    similarity = difflib.SequenceMatcher(None, keyword, header_text).ratio()
                    if similarity > 0.6 and similarity > highest_similarity:
                        matched_section = section_type
                        highest_similarity = similarity
            
            if matched_section:
                identified_sections.append({
                    "type": matched_section,
                    "text": header["text"],
                    "page": header["page"],
                    "y": header["y"]
                })
        
        # Filter to keep only the most likely header for each section type
        # (some papers might have multiple headers that match the same section)
        best_sections = {}
        for section in identified_sections:
            section_type = section["type"]
            if section_type not in best_sections:
                best_sections[section_type] = section
        
        # Convert to a list and sort by page/y-coordinate
        result = list(best_sections.values())
        result.sort(key=lambda s: (s["page"], s["y"]))
        
        logger.info(f"Identified {len(result)} standard section headers")
        return result
        
    def _extract_section_contents(self, doc, identified_sections):
        """Extract text content for each identified section."""
        sections = {}
        
        # If no sections were identified, return empty dict
        if not identified_sections:
            return sections
            
        # Add each section's content
        for i in range(len(identified_sections)):
            section = identified_sections[i]
            section_type = section["type"]
            start_page = section["page"]
            start_y = section["y"]
            
            # Determine end point (next section or end of document)
            if i < len(identified_sections) - 1:
                end_page = identified_sections[i + 1]["page"]
                end_y = identified_sections[i + 1]["y"]
            else:
                end_page = min(start_page + 10, doc.page_count - 1)  # Limit to 10 pages after
                end_y = float('inf')
            
            # Extract text for this section
            content = self._extract_text_between_points(doc, start_page, start_y, end_page, end_y)
            
            if content:
                sections[section_type] = content
                
        # Special case for abstract, which might be missing from identified sections
        if "abstract" not in sections and doc.page_count > 0:
            # Try to extract abstract from the first page
            abstract = self._extract_abstract_heuristic(doc)
            if abstract:
                sections["abstract"] = abstract
                
        return sections
        
    def _extract_text_between_points(self, doc, start_page, start_y, end_page, end_y):
        """Extract text between two points in the document."""
        text = ""
        
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num]
            
            # Get all blocks on this page
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                block_y = block["bbox"][1]  # Top y-coordinate of this block
                
                # Skip blocks before start point (on start page)
                if page_num == start_page and block_y < start_y:
                    continue
                    
                # Stop at end point (on end page)
                if page_num == end_page and block_y >= end_y:
                    break
                
                # Extract text from this block
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                    block_text += "\n"
                
                text += block_text + "\n"
        
        return text.strip()
        
    def _extract_abstract_heuristic(self, doc):
        """Extract abstract using heuristics when not identified by structure."""
        if doc.page_count == 0:
            return ""
            
        # Get first page content
        page = doc[0]
        text = page.get_text()
        
        # Try different approaches to find abstract
        abstract = ""
        
        # Method 1: Find text between "Abstract" and the next section
        abstract_match = re.search(r'(?i)abstract[\s\n:]*(.+?)(?=introduction|keywords|methods|\d+\.|\n\n\n)', 
                                 text, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        
        # Method 2: If no explicit abstract, try to get the first paragraph after title
        if not abstract:
            # Skip title and author info (usually first few lines)
            lines = text.split('\n')
            start_line = min(4, len(lines) - 1)
            
            # Find first significant paragraph
            paragraph = ""
            for i in range(start_line, min(15, len(lines))):
                line = lines[i].strip()
                if len(line) > 100:  # Likely a paragraph, not a title or author
                    paragraph = line
                    break
            
            if paragraph:
                abstract = paragraph
        
        return abstract
        
    def _extract_sections_regex(self, pdf_path):
        """Legacy method: Extract sections using regex patterns (fallback)."""
        try:
            full_text = self.extract_full_text(pdf_path)
            if not full_text:
                logger.warning(f"No text extracted from {pdf_path}")
                return {}
                
            # Define target sections
            target_sections = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion']
            
            # Initialize sections dictionary
            sections = {section: "" for section in target_sections}
            
            # Find section boundaries using improved regex patterns
            section_patterns = {
                'abstract': r'(?i)(abstract|summary)[\s\n:]*',
                'introduction': r'(?i)(introduction|background)[\s\n:]*',
                'methods': r'(?i)(methods|materials|methodology|experimental)[\s\n:]*',
                'results': r'(?i)(results|findings)[\s\n:]*',
                'discussion': r'(?i)(discussion|general\s+discussion)[\s\n:]*',
                'conclusion': r'(?i)(conclusion|conclusions|summary|final\s+remarks)[\s\n:]*'
            }
            
            end_patterns = {
                'abstract': r'(?i)(introduction|keywords|key\s+words)',
                'introduction': r'(?i)(methods|materials|experimental)',
                'methods': r'(?i)(results|findings)',
                'results': r'(?i)(discussion)',
                'discussion': r'(?i)(conclusion|acknowledgements|references)',
                'conclusion': r'(?i)(acknowledgements|references|bibliography)'
            }
            
            # For each section, find its content
            for section_name, pattern in section_patterns.items():
                # Find the section start
                match = re.search(pattern, full_text)
                if not match:
                    logger.warning(f"Section '{section_name}' not found")
                    continue
                    
                start_pos = match.end()
                
                # Find the end of this section (start of next section)
                end_match = re.search(end_patterns[section_name], full_text[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
                    sections[section_name] = full_text[start_pos:end_pos].strip()
                else:
                    # If no clear end marker, use a reasonable chunk of text
                    sections[section_name] = full_text[start_pos:start_pos + 5000].strip()
                
                logger.info(f"Found {section_name}: {len(sections[section_name])} characters")

            # Keep only non-empty sections
            sections = {k: v for k, v in sections.items() if v}
            return sections

        except Exception as e:
            logger.error(f"Error extracting sections: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    def find_pdf_by_metadata(self, pdf_dir, title, authors=None, year=None):
        """Find a PDF file in a directory based on metadata matching."""
        if not os.path.exists(pdf_dir):
            logger.error(f"Directory not found: {pdf_dir}")
            return None
            
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        # Clean the title for comparison
        clean_title = re.sub(r'[^\w\s]', ' ', title.lower())
        
        best_match = None
        best_match_score = 0
        
        for pdf_file in pdf_files:
            # Remove extension and clean filename
            pdf_name = os.path.splitext(pdf_file)[0]
            pdf_name_clean = re.sub(r'[^\w\s]', ' ', pdf_name.lower())
            
            # Calculate matching score
            score = self._calculate_title_match_score(clean_title, pdf_name_clean)
            
            # Check for year match if provided
            if year and str(year) in pdf_file:
                score += 0.2
                
            # Check for author match if provided
            if authors and any(author.lower() in pdf_file.lower() for author in authors):
                score += 0.3
                
            if score > best_match_score:
                best_match_score = score
                best_match = pdf_file
        
        # Return the best match if score exceeds threshold
        if best_match and best_match_score > 0.3:
            logger.info(f"Found PDF match: {best_match} (score: {best_match_score:.2f})")
            return os.path.join(pdf_dir, best_match)
        else:
            logger.warning(f"No matching PDF found for title: {title}")
            return None
    
    def _calculate_title_match_score(self, csv_title, pdf_name):
        """Calculate a matching score between a paper title and a PDF filename."""
        # Split into words and filter out short words
        csv_words = [w for w in csv_title.split() if len(w) > 3]
        pdf_words = [w for w in pdf_name.split() if len(w) > 3]
        
        # Method 1: Count matching significant words
        matching_words = set(csv_words).intersection(set(pdf_words))
        word_match_ratio = len(matching_words) / max(len(csv_words), 1)
        
        # Method 2: Check for consecutive words match
        consecutive_match = 0.0
        for i in range(len(csv_words) - 1):
            if i < len(csv_words) - 1:
                bigram = f"{csv_words[i]} {csv_words[i+1]}"
                if bigram in pdf_name:
                    consecutive_match = 0.4
                    break
        
        # Calculate final score
        final_score = (word_match_ratio * 0.6) + consecutive_match
        
        return min(final_score, 1.0)  # Cap at 1.0 