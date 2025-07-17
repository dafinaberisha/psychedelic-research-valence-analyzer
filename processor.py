import os
import csv
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
import openai
import PyPDF2

from .pdf_processor import PdfProcessor
from .claim_extractor import ClaimExtractor
from .db_connector import Neo4jConnector
from . import config

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("PaperProcessor")

class PaperProcessor:
    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None, openai_api_key=None):
        neo4j_uri = neo4j_uri or config.NEOJ4_URI
        neo4j_user = neo4j_user or config.NEOJ4_USER
        neo4j_password = neo4j_password or config.NEOJ4_PASSWORD
        
        if not openai_api_key:
            openai_api_key = os.environ.get('OPENAI_API_KEY', config.OPENAI_API_KEY)
        
        self.db_connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.pdf_processor = PdfProcessor()
        self.claim_extractor = ClaimExtractor(openai_api_key=openai_api_key)
        
        # Initialize zero-shot classifier
        self.zero_shot_loaded = False
        if TRANSFORMERS_AVAILABLE:
            self._init_zero_shot_classifier()
        
        self.results = {
            "processed_papers": 0,
            "successful_papers": 0,
            "failed_papers": 0,
            "sections_found": 0,
            "claims_extracted": 0,
            "papers": {}
        }
    
    def process_csv(self, csv_file: str, pdf_dir: str, drug_name: str, limit=None):
        try:
            drug_id = self.db_connector.create_drug_node(drug_name)
            
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                processed_count = 0
                
                for row in reader:
                    paper_data = self._extract_paper_metadata(row)
                    logger.info(f"Processing paper: {paper_data['title']}")
                    
                    self.results["processed_papers"] += 1
                    self.results["papers"][paper_data['title']] = {
                        "status": "processing",
                        "sections": {},
                        "claims": 0
                    }
                    
                    pdf_file = self._find_pdf_file(row, pdf_dir, paper_data)
                    
                    if pdf_file:
                        paper_data['pdf_path'] = pdf_file
                    
                    paper_id = self.db_connector.create_paper_node(paper_data)
                    
                    self.db_connector.create_relationship(
                        paper_id, "Paper", 
                        drug_id, "Drug", 
                        "DISCUSSES"
                    )
                    
                    self._process_authors(row, paper_id)
                    self._process_keywords(row, paper_id, paper_data['abstract'])
                    
                    try:
                        if pdf_file and os.path.exists(pdf_file):
                            logger.info(f"Found PDF: {pdf_file}")
                            
                            sections = self.pdf_processor.extract_sections(pdf_file)
                            
                            if sections:
                                self._process_sections(sections, paper_id, drug_id, drug_name)
                            else:
                                logger.warning(f"No sections found in PDF, using abstract from CSV")
                                self._process_abstract_fallback(paper_data, paper_id, drug_id, drug_name)
                        else:
                            logger.warning(f"No PDF found, using abstract from CSV")
                            self._process_abstract_fallback(paper_data, paper_id, drug_id, drug_name)
                        
                        self.results["successful_papers"] += 1
                        self.results["papers"][paper_data['title']]["status"] = "success"
                        
                    except Exception as e:
                        logger.error(f"Error processing paper {paper_data['title']}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        self.results["failed_papers"] += 1
                        self.results["papers"][paper_data['title']]["status"] = "error"
                        self.results["papers"][paper_data['title']]["error"] = str(e)
                    
                    processed_count += 1
                    
                    if limit is not None and processed_count >= limit:
                        logger.info(f"Reached processing limit of {limit} papers")
                        break

            
            logger.info(f"Finished processing CSV file: {csv_file}")
            logger.info(f"Processed {processed_count} papers, {self.results['successful_papers']} successful, {self.results['failed_papers']} failed")
            logger.info(f"Found {self.results['sections_found']} sections, extracted {self.results['claims_extracted']} claims")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            import traceback
            traceback.print_exc()
            return self.results
    
    def process_pdf_directory(self, pdf_dir: str, drug_name: str, limit=None):
        """Process all PDFs in a directory, extract sections and claims, and store in Neo4j."""
        try:
            # Create drug node
            drug_id = self.db_connector.create_drug_node(drug_name)
            
            # Get all PDF files in the directory
            pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
            logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
            
            # Process each PDF file
            processed_count = 0
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                logger.info(f"Processing PDF: {pdf_file}")
                
                # Update tracking
                self.results["processed_papers"] += 1
                self.results["papers"][pdf_file] = {
                    "status": "processing",
                    "sections": {},
                    "claims": 0
                }
                
                try:
                    # Try to find corresponding paper in database
                    paper_id = self.db_connector.find_paper_by_pdf_filename(pdf_file)
                    
                    if not paper_id:
                        # Create a minimal paper node if no matching paper was found
                        paper_data = {
                            'title': os.path.splitext(pdf_file)[0],
                            'paper_id': f"PDF-{os.path.splitext(pdf_file)[0]}",
                            'year': '',
                            'journal': '',
                            'doi': '',
                            'abstract': '',
                            'database': 'PDF_Directory',
                            'pdf_path': pdf_path  # Add the PDF path
                        }
                        paper_id = self.db_connector.create_paper_node(paper_data)
                        logger.info(f"Created new paper node for PDF: {pdf_file}")
                    else:
                        # Update existing paper with PDF path if needed
                        self._update_paper_pdf_path(paper_id, pdf_path)
                    
                    # Create relationship between paper and drug
                    self.db_connector.create_relationship(
                        paper_id, "Paper", 
                        drug_id, "Drug", 
                        "DISCUSSES"
                    )
                    
                    # Extract sections from PDF
                    sections = self.pdf_processor.extract_sections(pdf_path)
                    
                    # If we have sections, process them
                    if sections:
                        self._process_sections(sections, paper_id, drug_id, drug_name)
                        
                        # Update results
                        self.results["successful_papers"] += 1
                        self.results["papers"][pdf_file]["status"] = "success"
                    else:
                        logger.warning(f"No sections found in PDF: {pdf_file}")
                        self.results["failed_papers"] += 1
                        self.results["papers"][pdf_file]["status"] = "error"
                        self.results["papers"][pdf_file]["error"] = "No sections found"
                    
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Update results
                    self.results["failed_papers"] += 1
                    self.results["papers"][pdf_file]["status"] = "error"
                    self.results["papers"][pdf_file]["error"] = str(e)
                
                # Increment counter
                processed_count += 1
                
                # Check if we've reached the limit
                if limit is not None and processed_count >= limit:
                    logger.info(f"Reached processing limit of {limit} PDFs")
                    break
            
            # Print summary
            logger.info(f"Finished processing PDF directory: {pdf_dir}")
            logger.info(f"Processed {processed_count} PDFs, {self.results['successful_papers']} successful, {self.results['failed_papers']} failed")
            logger.info(f"Found {self.results['sections_found']} sections, extracted {self.results['claims_extracted']} claims")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error processing PDF directory: {e}")
            import traceback
            traceback.print_exc()
            return self.results
    
    def save_results(self, filename=None):
        """Save processing results to a JSON file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def _init_zero_shot_classifier(self):
        """Initialize zero-shot classification model."""
        try:
            logger.info("Loading zero-shot classification model...")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            self.zero_shot_loaded = True
            logger.info("Zero-shot classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading zero-shot model: {e}")
            import traceback
            traceback.print_exc()
            self.zero_shot_loaded = False
    
    def _calculate_zero_shot_valence(self, text: str) -> float:
        """Calculate valence using zero-shot classification."""
        try:
            if not text or len(text) < 10:
                logger.warning("Text too short, using default valence of 0.5")
                return 0.5
                
            if not self.zero_shot_loaded:
                logger.warning("Zero-shot model not loaded, using default valence of 0.5")
                return 0.5
                
            result = self.zero_shot_classifier(
                text,
                candidate_labels=["therapeutic", "neutral", "abusive"],
                multi_label=False
            )
            
            scores = dict(zip(result["labels"], result["scores"]))
            
            valence = scores.get("therapeutic", 0.0) * 1.0 + \
                     scores.get("neutral", 0.0) * 0.5 + \
                     scores.get("abusive", 0.0) * 0.0
                     
            valence = round(valence, 3)
            
            logger.info(f"Zero-shot valence: {valence:.3f}")
            logger.info(f"  Scores: therapeutic={scores.get('therapeutic', 0.0):.3f}, " + 
                       f"neutral={scores.get('neutral', 0.0):.3f}, " + 
                       f"abusive={scores.get('abusive', 0.0):.3f}")
            
            return valence
            
        except Exception as e:
            logger.error(f"Error calculating zero-shot valence: {e}")
            import traceback
            traceback.print_exc()
            return 0.5
    
    def close(self):
        """Close resources."""
        self.db_connector.close()
    
    def _extract_paper_metadata(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Extract basic paper metadata from CSV row."""
        return {
            'paper_id': row.get('accession_number', ''),
            'title': row.get('title', ''),
            'year': row.get('year', ''),
            'journal': row.get('secondary_title', ''),
            'doi': row.get('doi', ''),
            'abstract': row.get('abstract', ''),
            'database': row.get('database_provider', '')
        }
    
    def _process_authors(self, row: Dict[str, str], paper_id: str):
        """Process authors from CSV row and create nodes and relationships."""
        # Extract authors
        authors_raw = row.get('authors', '')
        if not authors_raw:
            logger.warning("No authors found in CSV row")
            return
        
        try:
            # Determine the format of authors_raw
            if isinstance(authors_raw, str):
                if authors_raw.startswith('[') and authors_raw.endswith(']'):
                    # It's a string representation of a list
                    try:
                        import ast
                        authors_list = ast.literal_eval(authors_raw)
                    except (SyntaxError, ValueError):
                        # If literal_eval fails, fall back to simple parsing
                        authors_list = [a.strip() for a in authors_raw.strip('[]').split(',') if a.strip()]
                elif ',' in authors_raw:
                    # Multiple authors separated by commas
                    authors_list = [a.strip() for a in authors_raw.split(',') if a.strip()]
                else:
                    # Single author
                    authors_list = [authors_raw.strip()]
            else:
                # Already a list or other iterable
                authors_list = authors_raw
            
            # Process each author
            for author_entry in authors_list:
                if not author_entry or not isinstance(author_entry, str):
                    continue
                    
                # Handle "LastName, FirstName" format
                if ',' in author_entry:
                    parts = author_entry.split(',', 1)
                    if len(parts) == 2:
                        last_name, first_name = parts[0].strip(), parts[1].strip()
                        author_name = f"{first_name} {last_name}"
                    else:
                        author_name = author_entry.strip()
                else:
                    author_name = author_entry.strip()
                
                # Skip empty names
                if not author_name:
                    continue
                    
                logger.info(f"Processing author: {author_name}")
                
                # Create author node
                author_id = self.db_connector.create_author_node(author_name)
                
                # Create relationship
                self.db_connector.create_relationship(
                    paper_id, "Paper", 
                    author_id, "Author", 
                    "WRITTEN_BY"
                )
        except Exception as e:
            logger.error(f"Error processing authors: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_keywords(self, row: Dict[str, str], paper_id: str, abstract: str):
        """Process keywords from CSV row and create nodes and relationships."""
        # Extract keywords - simplified version
        keywords_raw = row.get('keywords', '')
        keywords = []
        
        if keywords_raw:
            if isinstance(keywords_raw, str):
                if ',' in keywords_raw or ';' in keywords_raw:
                    keywords = [k.strip() for k in keywords_raw.replace(';', ',').split(',') if k.strip()]
                else:
                    keywords = [keywords_raw.strip()]
        
        # If no keywords or empty, extract from abstract
        if not keywords and abstract:
            try:
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                from nltk import pos_tag
                
                # Tokenize and remove stopwords
                stop_words = set(stopwords.words('english'))
                word_tokens = word_tokenize(abstract.lower())
                filtered_tokens = [w for w in word_tokens if w.isalpha() and w not in stop_words and len(w) > 3]
                
                # Get nouns and adjectives
                tagged = pos_tag(filtered_tokens)
                keywords = list(set([word for word, tag in tagged if tag.startswith('NN') or tag.startswith('JJ')]))
                
                # Limit to top 10
                keywords = keywords[:10]
                
            except Exception as e:
                logger.warning(f"Error extracting keywords from abstract: {e}")
                keywords = []
        
        # Create nodes and relationships
        for keyword in keywords:
            if keyword:
                # Create keyword node
                keyword_id = self.db_connector.create_keyword_node(keyword.lower())
                
                # Create relationship
                self.db_connector.create_relationship(
                    paper_id, "Paper", 
                    keyword_id, "Keyword", 
                    "HAS_KEYWORD"
                )
    
    def _find_pdf_file(self, row: Dict[str, str], pdf_dir: str, paper_data: Dict[str, Any]) -> Optional[str]:
        """Find the PDF file corresponding to this paper."""
        if not pdf_dir or not os.path.exists(pdf_dir):
            logger.warning(f"PDF directory does not exist: {pdf_dir}")
            return None

        try:
            pdf_path = None
            title = paper_data.get('title', '').strip()
            
            # Determine drug name from pdf_dir path
            if "ketamine" in pdf_dir.lower():
                drug_name = "ketamine"
            elif "psilocybin" in pdf_dir.lower():
                drug_name = "psilocybin"
            else:
                drug_name = None
            
            # For Ketamine papers
            if drug_name and drug_name.lower() == "ketamine":
                # First try the specific folder from file_attachments1
                file_attachment = row.get('file_attachments1', '')
                
                # Try method 1: Extract folder ID from file_attachments1
                if 'internal-pdf://' in file_attachment:
                    match = re.search(r'internal-pdf://(\d+)/', file_attachment)
                    if match:
                        folder_id = match.group(1)
                        folder_path = os.path.join(pdf_dir, folder_id)
                        if os.path.exists(folder_path):
                            for file in os.listdir(folder_path):
                                if file.lower().endswith('.pdf'):
                                    pdf_path = os.path.join(folder_path, file)
                                    logger.info(f"Found PDF using folder ID: {pdf_path}")
                                    break
                
                # If method 1 failed, try method 2: Search all folders
                if not pdf_path or not os.path.exists(pdf_path):
                    logger.info(f"Searching all Ketamine folders for paper: {title}")
                    # Get all folders in the Ketamine PDFs directory
                    if os.path.exists(pdf_dir):
                        best_match_score = 0
                        best_match_path = None
                        
                        for folder_name in os.listdir(pdf_dir):
                            folder_path = os.path.join(pdf_dir, folder_name)
                            if os.path.isdir(folder_path):
                                # Look for PDF files in this folder
                                for file in os.listdir(folder_path):
                                    if file.lower().endswith('.pdf'):
                                        # Calculate match score between file and paper title
                                        score = self._calculate_title_match_score(file_attachment, file)
                                        # logger.info(f"Match score for {file}: {score:.2f}")
                                        
                                        # Keep track of the best match
                                        if score > best_match_score:
                                            best_match_score = score
                                            best_match_path = os.path.join(folder_path, file)
                        
                        # Use the best match if any was found
                        if best_match_path:
                            pdf_path = best_match_path
                            logger.info(f"Found best PDF match: {pdf_path} with score {best_match_score:.2f}")
            
            # For Psilocybin papers
            elif drug_name and drug_name.lower() == "psilocybin":
                if title and os.path.exists(pdf_dir):
                    # Improved matching algorithm
                    best_match = None
                    best_match_score = 0
                    
                    # Get all PDF files in the directory
                    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
                    
                    # For each PDF file, calculate a matching score with the title
                    for pdf_file in pdf_files:
                        # Remove the file extension for comparison
                        pdf_name = os.path.splitext(pdf_file)[0]
                        
                        # Calculate matching score based on several methods
                        score = self._calculate_title_match_score(title, pdf_name)
                        
                        if score > best_match_score:
                            best_match_score = score
                            best_match = pdf_file
                    
                    # Use the best match if any was found
                    if best_match:
                        pdf_path = os.path.join(pdf_dir, best_match)
                        logger.info(f"Found best PDF match: '{best_match}' with score {best_match_score:.2f} for title: '{title}'")
            
            # Check file size before returning - skip extremely large files
            if pdf_path and os.path.exists(pdf_path):
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
                if file_size_mb > 20:  # Skip PDFs larger than 20MB
                    logger.warning(f"PDF is too large ({file_size_mb:.1f}MB) - skipping")
                    return None
                return pdf_path
            
            logger.warning(f"No matching PDF found for paper: {title}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_title_match_score(self, csv_title: str, pdf_name: str) -> float:
        """Calculate a matching score between a paper title and a PDF filename."""
        # Normalize both strings: lowercase and remove punctuation
        csv_title_norm = re.sub(r'[^\w\s]', ' ', csv_title.lower())
        pdf_name_norm = re.sub(r'[^\w\s]', ' ', pdf_name.lower())
        
        # Split into words and filter out short words
        csv_words = [w for w in csv_title_norm.split() if len(w) > 3]
        pdf_words = [w for w in pdf_name_norm.split() if len(w) > 3]
        
        # Method 1: Count matching significant words
        matching_words = set(csv_words).intersection(set(pdf_words))
        word_match_ratio = len(matching_words) / max(len(csv_words), 1)
        
        # Method 2: Check for author names (typically at the beginning of PDF filename)
        author_match = 0.0
        if " - " in pdf_name:
            pdf_authors = pdf_name.split(" - ")[0].lower()
            # Check if any author name appears in the CSV title (less common)
            if any(author.lower() in csv_title_norm for author in pdf_authors.split()):
                author_match = 0.3
        
        # Method 3: Check for year match
        year_match = 0.0
        csv_year_match = re.search(r'\b(19|20)\d{2}\b', csv_title_norm)
        pdf_year_match = re.search(r'\b(19|20)\d{2}\b', pdf_name_norm)
        if csv_year_match and pdf_year_match and csv_year_match.group() == pdf_year_match.group():
            year_match = 0.2
        
        # Method 4: Check for consecutive words match
        consecutive_match = 0.0
        for i in range(len(csv_words) - 1):
            if i < len(csv_words) - 1:  # Ensure we don't go out of bounds
                bigram = f"{csv_words[i]} {csv_words[i+1]}"
                if bigram in pdf_name_norm:
                    consecutive_match = 0.4
                    break
        
        # Calculate final score (weighted combination)
        final_score = (word_match_ratio * 0.5) + author_match + year_match + consecutive_match
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _process_sections(self, sections: Dict[str, str], paper_id: str, drug_id: str, drug_name: str):
        """Process sections from a paper and extract claims."""
        logger.info(f"Processing {len(sections)} sections")
        
        # Update counter
        self.results["sections_found"] += len(sections)
        
        # Extract claims from sections
        all_section_claims = self.claim_extractor.extract_claims_from_sections(sections, drug_name)
        
        # Process each section - make sure all sections are processed, even if they have no claims
        for section_name, section_text in sections.items():
            logger.info(f"Creating section node: {section_name}")
            
            # Get claims for this section (default to empty list if none found)
            section_claims = all_section_claims.get(section_name, [])
            
            # Calculate section-level metrics based on claims
            openai_valence = 0.5
            zero_shot_valence = 0.5
            sentiment = 0.5
            
            if section_claims:
                # Average the claim metrics
                openai_valence = sum(c.get('openai_valence', 0.5) for c in section_claims) / len(section_claims)
                zero_shot_valence = sum(c.get('zero_shot_valence', 0.5) for c in section_claims) / len(section_claims)
                sentiment = sum(c.get('sentiment', 0.5) for c in section_claims) / len(section_claims)
            else:
                # If no claims, calculate section valence directly
                logger.info(f"No claims found for {section_name}, calculating section valence directly")
                
                # For very short sections, we might want to skip valence calculation or use defaults
                if len(section_text) < 200:
                    logger.info(f"Section {section_name} is too short ({len(section_text)} chars), using default valence values")
                    openai_valence = 0.5
                    zero_shot_valence = 0.5
                    sentiment = 0.5
                else:
                    # Calculate valence for sections with sufficient length
                    valence_scores = self._calculate_section_valence(section_text, section_name, drug_name)
                    openai_valence = valence_scores.get('openai_valence', 0.5)
                    zero_shot_valence = valence_scores.get('zero_shot_valence', 0.5)
                    sentiment = valence_scores.get('sentiment', 0.5)
            
            # Create section node
            section_data = {
                'name': section_name,
                'text': section_text,
                'openai_valence': openai_valence,
                'zero_shot_valence': zero_shot_valence,
                'sentiment': sentiment,
                'key_claims': [c.get('text', '') for c in section_claims],  # Store claims
                'length': len(section_text)  # Add section length as additional metadata
            }
            
            section_id = self.db_connector.create_section_node(section_data)
            
            # Create relationship between paper and section
            self.db_connector.create_relationship(
                paper_id, "Paper", 
                section_id, "Section", 
                "HAS_SECTION"
            )
            
            # Process claims for this section if any exist
            if section_claims:
                self._process_claims(section_claims, section_id, paper_id, drug_id)
            else:
                logger.info(f"No claims to process for section: {section_name}")
            
            # Update results
            key = list(self.results["papers"].keys())[-1]  # Get key of the current paper
            self.results["papers"][key]["sections"][section_name] = {
                "claims": len(section_claims),
                "openai_valence": openai_valence,
                "zero_shot_valence": zero_shot_valence,
                "sentiment": sentiment,
                "length": len(section_text)
            }
    
    def _calculate_section_valence(self, section_text: str, section_name: str, drug_name: str) -> Dict[str, float]:
        """Calculate valence scores directly for a section when no claims are found."""
        try:
            # Truncate text if too long for processing
            if len(section_text) > 4000:
                logger.info(f"Truncating section text from {len(section_text)} to 4000 chars for valence calculation")
                section_text = section_text[:4000]
                
            # Calculate OpenAI valence
            openai_valence = self._calculate_openai_section_valence(section_text, section_name, drug_name)
            
            # Calculate zero-shot valence
            zero_shot_valence = self._calculate_zero_shot_valence(section_text)
            
            # Calculate sentiment
            sentiment = 0.5  # Default
            try:
                from textblob import TextBlob
                blob = TextBlob(section_text)
                polarity = blob.sentiment.polarity
                sentiment = (polarity + 1) / 2  # Normalize to 0-1
            except ImportError:
                logger.warning("TextBlob not available for sentiment analysis. Using default.")
            
            logger.info(f"Section valence: OpenAI={openai_valence:.2f}, Zero-shot={zero_shot_valence:.2f}, Sentiment={sentiment:.2f}")
            
            return {
                'openai_valence': openai_valence,
                'zero_shot_valence': zero_shot_valence,
                'sentiment': sentiment
            }
            
        except Exception as e:
            logger.error(f"Error calculating section valence: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default values on error
            return {
                'openai_valence': 0.5,
                'zero_shot_valence': 0.5,
                'sentiment': 0.5
            }
    
    def _calculate_openai_section_valence(self, section_text: str, section_name: str, drug_name: str) -> float:
        """Calculate section valence using OpenAI."""
        try:
            # Create a prompt for valence calculation
            prompt = f"""
You are an expert scientific analyzer focusing on psychedelic research.
Analyze the following text from the {section_name} section of a scientific paper about {drug_name}.

Provide a single valence score between 0 and 1 that represents the overall therapeutic vs. abusive 
effects described in THIS section regarding {drug_name}:

- Values close to 1: Strong therapeutic effects with minimal side effects
- Values close to 0: Strong abusive effects, serious side effects, or negative outcomes
- Values around 0.5: Neutral or balanced effects, or purely descriptive content

IMPORTANT: 
1. Your response MUST be a single number between 0 and 1 (e.g., 0.7, 0.25, 0.9).
2. Any mention of hallucinations, dissociation, psychosis, cognitive impairment, or other side effects should lower the score.
3. Strong therapeutic benefits should increase the score.
4. If the section is purely methodological or descriptive without clear valence, use a score of 0.5.

Here is the text:
{section_text}
"""

            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a scientific analyzer specializing in psychedelic research."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract valence value from response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse a floating point number from the response
            valence_match = re.search(r'(\d+\.\d+|\d+)', response_text)
            if valence_match:
                valence = float(valence_match.group(1))
                # Ensure the value is between 0 and 1
                valence = max(0.0, min(1.0, valence))
                return valence
            else:
                logger.warning(f"Could not extract valence value from OpenAI response: {response_text}")
                return 0.5  # Default to neutral
                
        except Exception as e:
            logger.error(f"Error calculating OpenAI section valence: {e}")
            return 0.5  # Default to neutral on error
    
    def _process_abstract_fallback(self, paper_data: Dict[str, Any], paper_id: str, drug_id: str, drug_name: str):
        """Process the abstract as a fallback when no PDF is available."""
        abstract = paper_data.get('abstract', '')
        
        if not abstract or len(abstract) < 100:
            logger.warning(f"Abstract too short or not available")
            return
        
        # Create a section with the abstract
        sections = {'abstract': abstract}
        
        # Update counter
        self.results["sections_found"] += 1
        
        # Process the section
        self._process_sections(sections, paper_id, drug_id, drug_name)
    
    def _process_claims(self, claims: List[Dict[str, Any]], section_id: str, paper_id: str, drug_id: str):
        """Process claims from a section and create nodes and relationships."""
        if not claims:
            return
        
        logger.info(f"Processing {len(claims)} claims")
        
        # Update counter
        self.results["claims_extracted"] += len(claims)
        
        # Create nodes and relationships for each claim
        for claim in claims:
            # Ensure all required fields are present
            claim_data = {
                'text': claim.get('text', ''),
                'openai_valence': claim.get('openai_valence', 0.5),
                'zero_shot_valence': claim.get('zero_shot_valence', 0.5) if 'zero_shot_valence' in claim else claim.get('valence', 0.5),
                'sentiment': claim.get('sentiment', 0.5),
                'evidence': claim.get('evidence', []),
                'rationale': claim.get('rationale', ''),
                'section': claim.get('section', '')
            }
            
            # Create claim node
            claim_id = self.db_connector.create_claim_node(claim_data)
            
            # Create relationship between section and claim
            self.db_connector.create_relationship(
                section_id, "Section", 
                claim_id, "Claim", 
                "CONTAINS"
            )
            
            # Create relationship between paper and claim
            self.db_connector.create_relationship(
                paper_id, "Paper", 
                claim_id, "Claim", 
                "CLAIMS"
            )
            
            # Create relationship between claim and drug
            self.db_connector.create_relationship(
                claim_id, "Claim", 
                drug_id, "Drug", 
                "ABOUT"
            )



    def _sort_sections(self, sections):
        """Sort sections in the logical order they appear in papers."""
        # Common section order in papers
        known_section_order = {
            'abstract': 0,
            'introduction': 10,
            'background': 20,
            'materials and methods': 30,
            'methods': 30,
            'results': 40,
            'discussion': 50,
            'conclusion': 60,
            'conclusions': 60,
            'references': 70,
            'bibliography': 70
        }
        
        # Helper function to determine section position
        def get_section_position(section):
            name = section.get('name', '').lower()
            
            # Check for exact matches first
            if name in known_section_order:
                return known_section_order[name]
            
            # Check for partial matches
            for key, value in known_section_order.items():
                if key in name:
                    return value
                
            # For unknown sections, assume they belong somewhere in the middle
            return 35
        
        # Sort sections by their position
        return sorted(sections, key=get_section_position)

    def _count_drug_mentions(self, text, drug):
        """Count mentions of a drug in text."""
        if not text:
            return 0
        
        # Normalize text for searching
        text_lower = text.lower()
        
        # Define variations of drug names to look for
        drug_variations = {
            'ketamine': ['ketamine', 'ketalar', 'special k', 'nmda antagonist', 'dissociative anesthetic'],
            'psilocybin': ['psilocybin', 'psilocin', 'mushroom', 'magic mushroom', 'psychedelic', 'hallucinogen']
        }
        
        # Count occurrences of each variation
        count = 0
        for variation in drug_variations.get(drug.lower(), [drug.lower()]):
            # Use word boundaries for more accurate matching
            pattern = r'\b' + re.escape(variation) + r'\b'
            count += len(re.findall(pattern, text_lower))
        
        return count 

    def _extract_sections(self, pdf_path):
        """Extract sections from a PDF file using the best available method."""
        # Try GROBID first (best quality)
        sections = self.pdf_processor._extract_sections_grobid(pdf_path)
        if sections and len(sections) > 1:
            logger.info(f"Successfully extracted {len(sections)} sections using GROBID")
            return sections
        
        # Fall back to PyMuPDF if GROBID fails
        logger.info("GROBID extraction failed or returned limited results, trying PyMuPDF...")
        sections = self.pdf_processor._extract_sections_pymupdf(pdf_path)
        if sections and len(sections) > 1:
            logger.info(f"Successfully extracted {len(sections)} sections using PyMuPDF")
            return sections
        
        # Last resort: regex-based extraction
        logger.info("PyMuPDF extraction failed, trying regex-based extraction...")
        sections = self.pdf_processor._extract_sections_regex(pdf_path)
        if sections:
            logger.info(f"Successfully extracted {len(sections)} sections using regex")
            return sections
        
        logger.warning(f"All section extraction methods failed for {pdf_path}")
        return {}

    def _extract_metadata(self, pdf_path):
        """Extract metadata from a PDF file."""
        try:
            filename = os.path.basename(pdf_path)
            
            # Try to get metadata from PDF
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                info = reader.metadata
                
                title = info.get('/Title', '')
                if not title:
                    # Try to get title from first page
                    first_page = reader.pages[0]
                    text = first_page.extract_text()
                    
                    # Use first line as title if not too long
                    lines = text.split('\n')
                    if lines and len(lines[0]) < 200:
                        title = lines[0].strip()
                    else:
                        # Use filename as fallback
                        title = os.path.splitext(filename)[0]
                        
                # Extract other metadata if available
                authors = info.get('/Author', '')
                date = info.get('/CreationDate', '')
                
                # Clean up date if present (format: D:YYYYMMDDhhmmss)
                if date and date.startswith('D:'):
                    date_str = date[2:10]  # Extract YYYYMMDD
                    try:
                        year = date_str[:4]
                        month = date_str[4:6]
                        day = date_str[6:8]
                        date = f"{year}-{month}-{day}"
                    except:
                        date = ''
                        
                return {
                    'title': title,
                    'authors': authors,
                    'date': date,
                    'filename': filename
                }
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return basic metadata from filename
            return {
                'title': os.path.splitext(os.path.basename(pdf_path))[0],
                'authors': '',
                'date': '',
                'filename': os.path.basename(pdf_path)
            } 

    def _update_paper_pdf_path(self, paper_id: str, pdf_path: str):
        """Update an existing paper with the PDF path if it's not already set."""
        try:
            with self.db_connector.driver.session() as session:
                result = session.run("""
                MATCH (p:Paper) WHERE elementId(p) = $paper_id AND (p.pdf_path IS NULL OR p.pdf_path = '')
                SET p.pdf_path = $pdf_path
                RETURN p.title as title
                """, 
                    paper_id=paper_id,
                    pdf_path=pdf_path
                )
                record = result.single()
                if record:
                    logger.info(f"Updated paper '{record['title']}' with PDF path: {pdf_path}")
                else:
                    logger.info(f"Paper already has PDF path or not found")
        except Exception as e:
            logger.warning(f"Error updating paper PDF path: {e}")
            # Continue processing even if update fails 