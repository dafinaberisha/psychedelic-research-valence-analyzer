import logging
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional

logger = logging.getLogger("DbConnector")

class Neo4jConnector:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
        logger.info("Database connection closed")
    
    def create_drug_node(self, drug_name: str) -> str:
        with self.driver.session() as session:
            result = session.run("""
            MERGE (d:Drug {name: $name})
            RETURN elementId(d) as drug_id
            """, name=drug_name)
            record = result.single()
            drug_id = record["drug_id"]
            logger.info(f"Created or retrieved drug node: {drug_name} (ID: {drug_id})")
            return drug_id
    
    def create_paper_node(self, paper_data: Dict[str, Any]) -> str:
        with self.driver.session() as session:
            result = session.run("""
            MERGE (p:Paper {
                paper_id: $paper_id,
                title: $title,
                year: $year,
                journal: $journal,
                doi: $doi,
                abstract: $abstract,
                database: $database,
                pdf_path: $pdf_path
            })
            RETURN elementId(p) as paper_id
            """, 
                paper_id=paper_data.get('paper_id', ''),
                title=paper_data.get('title', ''),
                year=paper_data.get('year', ''),
                journal=paper_data.get('journal', ''),
                doi=paper_data.get('doi', ''),
                abstract=paper_data.get('abstract', ''),
                database=paper_data.get('database', ''),
                pdf_path=paper_data.get('pdf_path', '')
            )
            record = result.single()
            node_id = record["paper_id"]
            logger.info(f"Created or retrieved paper node: {paper_data.get('title', 'Unknown')} (ID: {node_id})")
            return node_id
    
    def create_author_node(self, author_name: str) -> str:
        with self.driver.session() as session:
            result = session.run("""
            MERGE (a:Author {name: $name})
            RETURN elementId(a) as author_id
            """, name=author_name)
            record = result.single()
            author_id = record["author_id"]
            logger.info(f"Created or retrieved author node: {author_name} (ID: {author_id})")
            return author_id
    
    def create_keyword_node(self, keyword: str) -> str:
        with self.driver.session() as session:
            result = session.run("""
            MERGE (k:Keyword {name: $name})
            RETURN elementId(k) as keyword_id
            """, name=keyword)
            record = result.single()
            keyword_id = record["keyword_id"]
            logger.info(f"Created or retrieved keyword node: {keyword} (ID: {keyword_id})")
            return keyword_id
    
    def create_claim_node(self, claim_data: Dict[str, Any]) -> str:
        with self.driver.session() as session:
            result = session.run("""
            CREATE (c:Claim {
                text: $text,
                openai_valence: $openai_valence,
                zero_shot_valence: $zero_shot_valence,
                evidence: $evidence,
                sentiment: $sentiment,
                rationale: $rationale,
                section: $section
            })
            RETURN elementId(c) as claim_id
            """, 
                text=claim_data.get('text', ''),
                openai_valence=claim_data.get('openai_valence', 0.5),
                zero_shot_valence=claim_data.get('zero_shot_valence', 0.5),
                evidence=claim_data.get('evidence', []),
                sentiment=claim_data.get('sentiment', 0.5),
                rationale=claim_data.get('rationale', ''),
                section=claim_data.get('section', 'Unknown')
            )
            record = result.single()
            claim_id = record["claim_id"]
            logger.info(f"Created claim node: {claim_data.get('text', '')[:50]}... (ID: {claim_id})")
            return claim_id
    
    def create_section_node(self, section_data: Dict[str, Any]) -> str:
        with self.driver.session() as session:
            result = session.run("""
            CREATE (s:Section {
                name: $name,
                text: $text,
                openai_valence: $openai_valence,
                zero_shot_valence: $zero_shot_valence,
                sentiment: $sentiment,
                key_claims: $key_claims
            })
            RETURN elementId(s) as section_id
            """, 
                name=section_data.get('name', ''),
                text=section_data.get('text', ''),
                openai_valence=section_data.get('openai_valence', 0.5),
                zero_shot_valence=section_data.get('zero_shot_valence', 0.5),
                sentiment=section_data.get('sentiment', 0.5),
                key_claims=section_data.get('key_claims', [])
            )
            record = result.single()
            section_id = record["section_id"]
            logger.info(f"Created section node: {section_data.get('name', '')} (ID: {section_id})")
            return section_id
    
    def create_relationship(self, source_id: str, source_label: str, 
                           target_id: str, target_label: str, 
                           relationship_type: str) -> bool:
        with self.driver.session() as session:
            result = session.run("""
            MATCH (source) WHERE elementId(source) = $source_id
            MATCH (target) WHERE elementId(target) = $target_id
            MERGE (source)-[r:""" + relationship_type + """]->(target)
            RETURN COUNT(r) as relationship_count
            """, 
                source_id=source_id,
                target_id=target_id
            )
            record = result.single()
            relationship_count = record["relationship_count"]
            logger.info(f"Created or retrieved relationship: ({source_label})-[{relationship_type}]->({target_label})")
            return relationship_count > 0
    
    def find_paper_by_title(self, title: str) -> Optional[str]:
        with self.driver.session() as session:
            result = session.run("""
            MATCH (p:Paper)
            WHERE toLower(p.title) CONTAINS toLower($title)
            RETURN elementId(p) as paper_id, p.title as title
            LIMIT 1
            """, title=title)
            record = result.single()
            if record:
                logger.info(f"Found paper: {record['title']}")
                return record["paper_id"]
            else:
                logger.warning(f"No paper found with title containing: {title}")
                return None
    
    def find_paper_by_pdf_filename(self, pdf_filename: str) -> Optional[str]:
        import re
        clean_filename = pdf_filename.replace('.pdf', '')
        
        year_match = re.search(r'(?:19|20)\d{2}', clean_filename)
        year = year_match.group(0) if year_match else None
        
        with self.driver.session() as session:
            query = """
            MATCH (p:Paper)
            WHERE 
                toLower(p.title) CONTAINS toLower($clean_filename)
                OR (p.year = $year AND $year IS NOT NULL)
            RETURN elementId(p) as paper_id, p.title as title, p.year as year
            LIMIT 1
            """
            
            result = session.run(
                query,
                clean_filename=clean_filename,
                year=year
            )
            
            record = result.single()
            if record:
                logger.info(f"Found matching paper: {record['title']} ({record['year']})")
                return record["paper_id"]
            else:
                logger.warning(f"No match found for PDF: {pdf_filename}")
                return None 