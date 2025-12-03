"""
Qwen Reranker
Qwen2.5 model for reranking documents using LLM inference
"""
from typing import List, Tuple, Dict, Any, Union
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenReranker:
    """
    Qwen2.5 based reranker using LLM inference.
    Uses chat template to select the most relevant document.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = None,
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        threshold: float = 0.8
    ):
        """
        Initialize Qwen reranker with threshold-based selection.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.threshold = threshold
        self.max_content_length = 2000  
        
        # Load tokenizer and model 
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load model only when needed (for LLM inference)"""
        if self._model_loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self._model_loaded = True
    
    @staticmethod
    def build_documents_dict(documents: List[Dict[str, Any]], id_key: str = "aid") -> Dict[str, Dict[str, Any]]:
        """
        Build documents dictionary from list for efficient lookup.
        
        Example:
            Input: [{"aid": "A1", "content": "..."}, {"aid": "A2", "content": "..."}]
            Output: {"A1": {"aid": "A1", "content": "..."}, "A2": {"aid": "A2", "content": "..."}}
        """
        documents_dict = {}
        for idx, doc in enumerate(documents):
            # Get ID from document
            doc_id = doc.get(id_key, doc.get("id", f"doc_{idx}"))
            if doc_id:
                documents_dict[doc_id] = doc
        return documents_dict
    
    def _truncate_content(self, content: str, max_length: int = None) -> str:
        """
        Truncate content to prevent context window overflow.
        """
        if max_length is None:
            max_length = self.max_content_length
        
        if len(content) <= max_length:
            return content
        
        words = content.split()
        if len(words) > max_length: 
            return " ".join(words[:max_length]) + "..."

        return content
    
    def rerank(
        self, 
        query: str, 
        reranker_results: List[Tuple[str, float]], 
        documents_dict: Dict[str, Dict[str, Any]],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Select relevant article using threshold-based logic with LLM fallback.
        
        Logic:
        1. If any article has score > threshold: select article with highest score
        2. If no article > threshold: use LLM to select most relevant article
        """
        if not reranker_results:
            return []
        
        # Sort by score (descending)
        sorted_results = sorted(reranker_results, key=lambda x: x[1], reverse=True)
        
        # Check if any article has score > threshold
        max_score_article = sorted_results[0] if sorted_results else None
        
        if max_score_article and max_score_article[1] > self.threshold:
            # Score > threshold: return article with highest score
            return [max_score_article]
        
        # Score <= threshold: use LLM to select
        return self._rerank_with_llm(query, sorted_results, documents_dict, top_k)
    
    def _rerank_with_llm(
        self, 
        query: str, 
        reranker_results: List[Tuple[str, float]], 
        documents_dict: Dict[str, Dict[str, Any]],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Use LLM to select most relevant article from candidates.
        """
        if not reranker_results:
            return []
        
        # Load model if not loaded
        self._load_model()
        
        # Format documents into context string (use top candidates)
        context_str = ""
        candidate_aids = []
        for aid, score in reranker_results[:10]:  # Limit to top 10 for LLM
            if aid not in documents_dict:
                continue
            doc = documents_dict[aid]
            content = doc.get('content', doc.get('text', ''))
            if content:
                # Truncate content 
                truncated_content = self._truncate_content(content)
                context_str += f"Article ID: {aid}\nScore: {score:.4f}\nContent: {truncated_content}\n---\n"
                candidate_aids.append(aid)
        
        if not context_str or not candidate_aids:
            # Fallback: return highest score article if no valid candidates
            return [reranker_results[0]] if reranker_results else []
        
        # System prompt
        system_prompt = """You are a legal retrieval assistant. Your task is to analyze the user's query and the provided candidate articles.
                    Identify  articles that is most relevant and directly answers the query.
                    CRITICAL INSTRUCTION: Output ONLY the Article ID (aid) in JSON format like this: {"aid": "..."}. Do not provide explanations."""
        
        user_prompt = f"""Query: {query}

Candidate Articles:
{context_str}

Which article is the most relevant to the query?"""
        
        # Create messages list
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and inference
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract result
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Parse JSON output
        selected_aid = None
        try:
            # Try to parse JSON directly first
            result = json.loads(generated_text)
            selected_aid = result.get('aid', '')
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    result = json.loads(json_str)
                    selected_aid = result.get('aid', '')
                else:
                    # Fallback: try to extract aid from text (e.g., "aid": "art_01")
                    import re
                    match = re.search(r'"aid"\s*:\s*"([^"]+)"', generated_text)
                    if match:
                        selected_aid = match.group(1)
                    else:
                        selected_aid = generated_text.strip()
            except Exception:
                # Last fallback: use raw text
                selected_aid = generated_text.strip()
        
        # Return result with proper fallback logic
        if not reranker_results:
            return []
        
        if selected_aid and selected_aid in candidate_aids:
            # LLM successfully selected a valid candidate
            return [(selected_aid, 1.0)]
        else:
            # Fallback: return highest score article (already sorted by score descending)
            return [reranker_results[0]]
    
    def select_article(
        self,
        query: str,
        reranker_results: List[Tuple[str, float]],
        documents: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
    ) -> Tuple[str, float]:
        """
        Select the most relevant article using threshold-based logic.
        """
        # Convert List to Dict if needed
        if isinstance(documents, list):
            documents_dict = self.build_documents_dict(documents)
        else:
            documents_dict = documents
        
        results = self.rerank(query, reranker_results, documents_dict, top_k=1)
        if results:
            return results[0]
        return (None, 0.0)