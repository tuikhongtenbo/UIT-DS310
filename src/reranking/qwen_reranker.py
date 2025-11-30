"""
Qwen Reranker
Qwen2.5 model for reranking documents using LLM inference
"""
from typing import List, Tuple, Dict, Any
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
        temperature: float = 0.1
    ):
        """
        Initialize Qwen reranker.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            self.model = self.model.to(device)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Rerank documents for a given query using LLM inference.
        
        Args:
            query: Search query
            documents: List of document dicts with keys like 'aid' (article id) and 'content'
            top_k: Number of top results to return (currently only top-1 is supported)
        
        Returns:
            List of (document_id, score) tuples (score is always 1.0 for selected document)
        """
        if not documents:
            return []
        
        # Format documents into context string
        context_str = ""
        for doc in documents:
            aid = doc.get('aid', doc.get('id', ''))
            content = doc.get('content', doc.get('text', ''))
            context_str += f"Article ID: {aid}\nContent: {content}\n---\n"
        
        # System prompt
        system_prompt = """You are a legal retrieval assistant. Your task is to analyze the user's query and the provided candidate articles.
                    Identify the ONE article that is most relevant and directly answers the query.
                    CRITICAL INSTRUCTION: Output ONLY the Article ID (aid) in JSON format like this: {"aid": "..."}. Do not provide explanations."""
        
        user_prompt = f"""Query: {query}

Candidate Articles:
{context_str}

    Which article is the most relevant?"""
        
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
        try:
            # Try to extract JSON from the output
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                result = json.loads(json_str)
                selected_aid = result.get('aid', '')
            else:
                # Fallback: try to find aid in text
                selected_aid = generated_text
        except:
            selected_aid = generated_text
        
        # Return result with score 1.0
        if selected_aid:
            return [(selected_aid, 1.0)]
        return []
    
    def score(self, query: str, document: Dict[str, Any]) -> float:
        """
        Calculate relevance score between query and document.
        Returns 1.0 if document is selected, 0.0 otherwise.
        
        Args:
            query: Search query
            document: Document dict with 'aid' and 'content'
        
        Returns:
            Relevance score (1.0 if selected, 0.0 otherwise)
        """
        results = self.rerank(query, [document], top_k=1)
        if results and results[0][0] == document.get('aid', document.get('id', '')):
            return 1.0
        return 0.0