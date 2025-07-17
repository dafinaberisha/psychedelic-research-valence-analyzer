import logging
import re
import openai
import os
from typing import List, Dict, Any

logger = logging.getLogger("ClaimExtractor")

class ClaimExtractor:
    def __init__(self, openai_api_key=None):
        if openai_api_key:
            openai.api_key = openai_api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            logger.warning("No OpenAI API key provided. Claim extraction may fail.")
            
        self.model_token_limits = {
            "gpt-4": 8000,
            "gpt-4o": 8000,
            "gpt-4-turbo": 8000,
            "gpt-4o-mini": 4000,
            "gpt-3.5-turbo": 4000,
        }
        
        self.default_model = "gpt-4o-mini"
        self.max_text_length = 4000
    
    def extract_claims(self, text: str, drug: str, section_context: str = "") -> List[Dict[str, Any]]:
        if not text or len(text) < 50:
            logger.warning("Text too short for claim extraction")
            return []
            
        chunks = self._split_text_into_chunks(text)
        
        all_claims = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} for claim extraction")
            chunk_claims = self._extract_claims_from_chunk(chunk, drug, section_context)
            all_claims.extend(chunk_claims)
            
        return all_claims
        
    def _split_text_into_chunks(self, text: str) -> List[str]:
        if len(text) <= self.max_text_length:
            return [text]
            
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.max_text_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    def _extract_claims_from_chunk(self, text: str, drug: str, section_context: str) -> List[Dict[str, Any]]:
        try:
            prompt = self._create_extraction_prompt(text, drug, section_context)
            
            response = openai.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a scientific analyst specializing in extracting factual claims from scientific papers about psychedelics and other drugs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content.strip()
            return self._parse_claims_from_response(result, text)
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []
            
    def _create_extraction_prompt(self, text: str, drug: str, section_context: str) -> str:
        prompt = f"""
You are an expert scientific claim analyzer focusing on psychedelic research.
Read the following text from a scientific paper about {drug} {section_context}.

ONLY extract claims that represent ACTUAL FINDINGS, RESULTS, or CONCLUSIONS from THIS specific research paper.
DO NOT extract background information, literature review statements, or established facts from previous research.

For example:
- "Chronic stress and GABA deficits are implicated in depression" is background information, NOT a claim to extract.
- "Our results showed that {drug} reduced depression scores by 30%" IS a claim to extract.

Extract up to 3 key scientific claims about {drug} from THIS text only.
Focus on what THIS paper discovered or concluded, not what was already known.

For each claim:
- Claim: A concise statement of the NEW finding or conclusion from THIS research
- Valence: A value between 0 and 1 (with up to two decimal places) that represents:
  * Values close to 1: Strong therapeutic effects
  * Values close to 0: Strong abusive effects
  * Values around 0.5: Neutral or balanced effects
- Evidence: Supporting details from THIS text showing this is a finding FROM THIS STUDY
- Rationale: Brief explanation for why you assigned this valence value

IMPORTANT: 
1. Valence scores MUST be between 0 and 1 (e.g., 0.7, 0.25, 0.9).
2. Any mention of hallucinations, dissociation, psychosis, cognitive impairment, or other side effects should receive lower valence values (closer to 0).
3. Strong therapeutic benefits should receive higher valence values (closer to 1).
4. Only extract claims that represent NEW findings or conclusions from the current research.

Format each claim exactly as:
- Claim: [text]
- Valence: [0-1]
- Evidence: [text]
- Rationale: [text]

Text:
{text}
"""
        return prompt
        
    def _parse_claims_from_response(self, response_text: str, original_text: str) -> List[Dict[str, Any]]:
        claims = []

        claim_blocks = re.split(r'(?:\n|^)-\s*Claim:', response_text)
        
        blocks_to_process = claim_blocks[1:] if len(claim_blocks) > 1 else claim_blocks
        
        for block in blocks_to_process:
            if not block.strip():
                continue
            
            claim_data = {
                'text': '',
                'openai_valence': 0.5,
                'evidence': '',
                'rationale': '',
                'start': -1,
                'end': -1
            }
            
            claim_text = block.strip()
            valence_match = re.search(r'(?:\n|^)-?\s*Valence:\s*([\d.]+)', block)
            if valence_match:
                claim_text = block[:valence_match.start()].strip()
                claim_data['text'] = claim_text
                
                try:
                    valence = float(valence_match.group(1))
                    claim_data['openai_valence'] = max(0.0, min(1.0, valence))
                except ValueError:
                    logger.warning(f"Could not parse valence value: {valence_match.group(1)}")
            
            evidence_match = re.search(r'(?:\n|^)-?\s*Evidence:\s*(.*?)(?=(?:\n|^)-?\s*(?:Rationale|$))', block, re.DOTALL)
            if evidence_match:
                claim_data['evidence'] = evidence_match.group(1).strip()
            
            rationale_match = re.search(r'(?:\n|^)-?\s*Rationale:\s*(.*?)(?:\Z)', block, re.DOTALL)
            if rationale_match:
                claim_data['rationale'] = rationale_match.group(1).strip()
            
            if claim_data['text']:
                claim_words = claim_data['text'].split()
                if len(claim_words) >= 5:
                    search_pattern = ' '.join(claim_words[:min(10, len(claim_words))])
                    
                    match_pos = original_text.lower().find(search_pattern.lower())
                    if match_pos >= 0:
                        claim_data['start'] = match_pos
                        claim_data['end'] = match_pos + len(claim_data['text'])
            
            # Add to results if we have a valid claim
            if claim_data['text']:
                claims.append(claim_data)
        
        return claims
        
    def _deduplicate_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not claims:
            return []
            
        sorted_claims = sorted(claims, key=lambda x: len(x.get('text', '')), reverse=True)
        
        unique_claims = []
        claim_texts = set()
        
        for claim in sorted_claims:
            claim_text = claim.get('text', '').strip()
            
            if not claim_text or len(claim_text) < 10:
                continue
                
            if claim_text.lower() in claim_texts:
                continue
                
            unique_claims.append(claim)
            claim_texts.add(claim_text.lower())
            
        return unique_claims
        
    def extract_claims_from_sections(self, sections: Dict[str, str], drug: str) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        
        for section_name, section_text in sections.items():
            logger.info(f"Extracting claims from section: {section_name}")
            
            if not section_text or len(section_text) < 50:
                continue
                
            section_context = f"from the {section_name} section"
                
            section_claims = self.extract_claims(section_text, drug, section_context)
            
            if section_claims:
                results[section_name] = section_claims
                
        return results
    
 