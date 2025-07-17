import logging
from db_connector import Neo4jConnector
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SectionValenceUpdater")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library is available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Please install with: pip install transformers")

class SectionValenceUpdater:
    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        neo4j_uri = neo4j_uri or config.NEOJ4_URI
        neo4j_user = neo4j_user or config.NEOJ4_USER
        neo4j_password = neo4j_password or config.NEOJ4_PASSWORD
        
        self.db_connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        
        self.model_loaded = False
        if TRANSFORMERS_AVAILABLE:
            self._init_model()
        
        self.stats = {
            'sections_processed': 0,
            'sections_updated': 0,
            'errors': 0
        }
    
    def _init_model(self):
        try:
            logger.info("Loading zero-shot classification model...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            self.model_loaded = True
            logger.info("Zero-shot classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def update_sections(self, batch_size=20, limit=None):
        logger.info("Starting section valence update process...")
        
        if not self.model_loaded:
            logger.error("Model is not loaded. Cannot update valence values.")
            return self.stats
        
        try:
            sections = self._get_sections_needing_update(batch_size, limit)
            logger.info(f"Found {len(sections)} sections needing valence update")
            
            for section in sections:
                try:
                    section_text = section['text']
                    section_id = section['id']
                    section_name = section['name']
                    
                    logger.info(f"Processing section: {section_name} ({len(section_text)} chars)")
                    
                    if len(section_text) > 1024:
                        logger.info(f"Section text is long ({len(section_text)} chars), truncating to 1024 chars")
                        section_text = section_text[:1024]
                    
                    valence = self.calculate_valence(section_text)
                    
                    self._update_section_valence(section_id, valence)
                    
                    self.stats['sections_updated'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing section {section.get('id')}: {e}")
                    self.stats['errors'] += 1
                
                self.stats['sections_processed'] += 1
                
                if self.stats['sections_processed'] % 5 == 0:
                    logger.info(f"Progress: {self.stats['sections_processed']}/{len(sections)} sections processed")
            
            logger.info(f"Update complete. {self.stats['sections_updated']} sections updated, {self.stats['errors']} errors")
            return self.stats
            
        except Exception as e:
            logger.error(f"Error updating section valence values: {e}")
            import traceback
            traceback.print_exc()
            return self.stats
    
    def calculate_valence(self, text: str) -> float:
        try:
            if not text or len(text) < 10:
                logger.warning("Text too short, using default valence of 0.5")
                return 0.5
                
            result = self.classifier(
                text,
                candidate_labels=["therapeutic", "neutral", "abusive"],
                multi_label=False
            )
            
            scores = dict(zip(result["labels"], result["scores"]))
            
            valence = scores.get("therapeutic", 0.0) * 1.0 + \
                     scores.get("neutral", 0.0) * 0.5 + \
                     scores.get("abusive", 0.0) * 0.0
                     
            valence = round(valence, 3)
            
            logger.info(f"Valence: {valence:.3f}")
            logger.info(f"  Scores: therapeutic={scores.get('therapeutic', 0.0):.3f}, " + 
                       f"neutral={scores.get('neutral', 0.0):.3f}, " + 
                       f"abusive={scores.get('abusive', 0.0):.3f}")
            
            return valence
            
        except Exception as e:
            logger.error(f"Error calculating valence: {e}")
            import traceback
            traceback.print_exc()
            return 0.5
    
    def _get_sections_needing_update(self, batch_size=20, limit=None):
        with self.db_connector.driver.session() as session:
            query = """
            MATCH (s:Section)
            RETURN elementId(s) as id, s.name as name, s.text as text
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = session.run(query)
            return [dict(record) for record in result]
    
    def _update_section_valence(self, section_id, valence):
        with self.db_connector.driver.session() as session:
            result = session.run("""
            MATCH (s:Section)
            WHERE elementId(s) = $section_id
            SET s.zero_shot_valence = $valence
            RETURN s.name as name
            """, 
                section_id=section_id,
                valence=valence
            )
            
            record = result.single()
            if record:
                name = record['name']
                logger.info(f"Updated section: {name} with valence: {valence:.3f}")
                return True
            return False
    
    def run_diagnostics(self, limit=3):
        """Run diagnostics on a few sections to see the model outputs."""
        if not self.model_loaded:
            logger.error("Model is not loaded. Cannot run diagnostics.")
            return
        
        logger.info("=== RUNNING SECTION DIAGNOSTICS ===")
        
        try:
            # Get a sample of sections
            with self.db_connector.driver.session() as session:
                query = """
                MATCH (s:Section)
                RETURN elementId(s) as id, s.name as name, s.text as text, 
                       s.openai_valence as openai_valence, s.zero_shot_valence as zero_shot_valence
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                sections = [dict(record) for record in result]
            
            if not sections:
                logger.warning("No sections found for diagnostics")
                return
            
            logger.info(f"Running diagnostics on {len(sections)} sections")
            
            for section in sections:
                section_name = section['name']
                section_text = section['text']
                openai_valence = section.get('openai_valence', 'N/A')
                existing_zero_shot = section.get('zero_shot_valence', 'N/A')
                
                # Truncate text for logging
                text_excerpt = section_text[:200] + "..." if len(section_text) > 200 else section_text
                
                logger.info("\n" + "="*80)
                logger.info(f"SECTION NAME: {section_name}")
                logger.info(f"TEXT EXCERPT: {text_excerpt}")
                logger.info(f"EXISTING VALENCES: OpenAI={openai_valence}, Zero-shot={existing_zero_shot}")
                
                # Calculate valence
                valence = self.calculate_valence(section_text[:1024])  # Use first 1024 chars
                logger.info(f"NEW VALENCE: {valence}")
            
            logger.info("=== DIAGNOSTICS COMPLETE ===")
        
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Close database connection."""
        self.db_connector.close()

if __name__ == "__main__":
    # Initialize updater
    updater = SectionValenceUpdater()
    
    try:
        # Run diagnostics first
        updater.run_diagnostics(limit=3)
        
        # Ask if user wants to continue
        response = input("\nContinue with updating all sections? (y/n): ")
        if response.lower() == 'y':
            # Ask for a limit
            limit_input = input("Enter limit (or leave empty for all sections): ")
            limit = int(limit_input) if limit_input.strip() else None
            
            # Run the update process
            stats = updater.update_sections(limit=limit)
            
            # Print summary
            print("\nUpdate Summary:")
            print(f"Total sections processed: {stats['sections_processed']}")
            print(f"Sections updated with valence: {stats['sections_updated']}")
            print(f"Errors encountered: {stats['errors']}")
        else:
            print("Update cancelled.")
        
    finally:
        # Ensure resources are closed
        updater.close()