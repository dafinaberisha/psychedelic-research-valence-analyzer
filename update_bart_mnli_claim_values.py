import logging
from db_connector import Neo4jConnector
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValenceUpdater")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library is available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Please install with: pip install transformers")

class ValenceUpdater:
    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        neo4j_uri = neo4j_uri or config.NEOJ4_URI
        neo4j_user = neo4j_user or config.NEOJ4_USER
        neo4j_password = neo4j_password or config.NEOJ4_PASSWORD
        
        self.db_connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        
        self.model_loaded = False
        if TRANSFORMERS_AVAILABLE:
            self._init_model()
        
        self.stats = {
            'claims_processed': 0,
            'claims_updated': 0,
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
    
    def update_claims(self, batch_size=100, limit=None):
        logger.info("Starting valence update process...")
        
        if not self.model_loaded:
            logger.error("Model is not loaded. Cannot update valence values.")
            return self.stats
        
        try:
            claims = self._get_claims_needing_update(batch_size, limit)
            logger.info(f"Found {len(claims)} claims needing valence update")
            
            for claim in claims:
                try:
                    claim_text = claim['text']
                    claim_id = claim['id']
                    
                    logger.info(f"Processing claim: {claim_text[:50]}...")
                    
                    valence = self.calculate_valence(claim_text)
                    
                    self._update_claim_valence(claim_id, valence)
                    
                    self.stats['claims_updated'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing claim {claim.get('id')}: {e}")
                    self.stats['errors'] += 1
                
                self.stats['claims_processed'] += 1
                
                if self.stats['claims_processed'] % 10 == 0:
                    logger.info(f"Progress: {self.stats['claims_processed']}/{len(claims)} claims processed")
            
            logger.info(f"Update complete. {self.stats['claims_updated']} claims updated, {self.stats['errors']} errors")
            return self.stats
            
        except Exception as e:
            logger.error(f"Error updating valence values: {e}")
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
    
    def _get_claims_needing_update(self, batch_size=100, limit=None):
        with self.db_connector.driver.session() as session:
            query = """
            MATCH (c:Claim)
            RETURN elementId(c) as id, c.text as text
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = session.run(query)
            return [dict(record) for record in result]
    
    def _update_claim_valence(self, claim_id, valence):
        with self.db_connector.driver.session() as session:
            result = session.run("""
            MATCH (c:Claim)
            WHERE elementId(c) = $claim_id
            SET c.zero_shot_valence = $valence
            RETURN c.text as text
            """, 
                claim_id=claim_id,
                valence=valence
            )
            
            record = result.single()
            if record:
                text = record['text']
                logger.info(f"Updated claim: {text[:30]}... with valence: {valence:.3f}")
                return True
            return False
    
    def run_diagnostics(self, limit=5):
        """Run diagnostics on a few claims to see the raw model outputs."""
        if not self.model_loaded:
            logger.error("Model is not loaded. Cannot run diagnostics.")
            return
        
        logger.info("=== RUNNING DIAGNOSTICS ===")
        
        try:
            # Get a sample of claims
            with self.db_connector.driver.session() as session:
                query = """
                MATCH (c:Claim)
                RETURN elementId(c) as id, c.text as text, c.openai_valence as openai_valence
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                claims = [dict(record) for record in result]
            
            if not claims:
                logger.warning("No claims found for diagnostics")
                return
            
            logger.info(f"Running diagnostics on {len(claims)} claims")
            
            for claim in claims:
                claim_text = claim['text']
                openai_valence = claim.get('openai_valence', 'N/A')
                
                logger.info("\n" + "="*80)
                logger.info(f"CLAIM TEXT: {claim_text}")
                logger.info(f"OPENAI VALENCE: {openai_valence}")
                
                # Calculate valence
                valence = self.calculate_valence(claim_text)
                logger.info(f"CALCULATED VALENCE: {valence}")
            
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
    updater = ValenceUpdater()
    
    try:
        # Run diagnostics first
        updater.run_diagnostics(limit=5)
        
        # Ask if user wants to continue
        response = input("\nContinue with updating all claims? (y/n): ")
        if response.lower() == 'y':
            # Run the update process
            stats = updater.update_claims(limit=None)  # set limit=None to process all claims
            
            # Print summary
            print("\nUpdate Summary:")
            print(f"Total claims processed: {stats['claims_processed']}")
            print(f"Claims updated with valence: {stats['claims_updated']}")
            print(f"Errors encountered: {stats['errors']}")
        else:
            print("Update cancelled.")
        
    finally:
        # Ensure resources are closed
        updater.close()