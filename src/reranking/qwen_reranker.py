"""
Qwen Reranker
Qwen2.5 model for reranking documents using LLM inference
"""
from typing import List, Tuple, Dict, Any, Union, Optional
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

# Try to import BitsAndBytesConfig for 4-bit quantization
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None
    logger.warning("bitsandbytes not available. 4-bit quantization will be disabled.")


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
        threshold: float = 0.8,
        use_4bit: bool = True,
        exp_num: Optional[int] = None
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
        self.use_4bit = use_4bit and BITSANDBYTES_AVAILABLE and device == "cuda"
        self.exp_num = exp_num
        
        # Load tokenizer and model 
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load model only when needed (for LLM inference)"""
        if self._model_loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Use 4-bit quantization if enabled and available
        if self.use_4bit:
            logger.info("Loading Qwen model with 4-bit quantization to save VRAM...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            # device_map="auto" handles device placement automatically
        else:
            logger.info("Loading Qwen model with full precision...")
            # Try to load with device_map if accelerate is available
            try:
                import accelerate
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            except ImportError:
                # Fallback: load without device_map
                logger.warning("accelerate not installed, loading model without device_map")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                # Manually move to device
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                elif self.device == "cpu":
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
        
        # Select prompts based on experiment number
        if self.exp_num in [10, 11, 12]:
            # System prompt for exp_10, 11, 12
            if self.exp_num == 12:
                # exp_12: select 1-3 articles
                system_prompt = """You are a legal retrieval assistant. Your task is to analyze the user's query and the provided candidate articles.
                            Select the most relevant articles (from 1 to 3 articles) based on relevance.
                            CRITICAL INSTRUCTION: Output ONLY the Article IDs (aids) in JSON format like this: {"aids": ["aid1", "aid2", ...]}. 
                            You can select 1, 2, or 3 articles depending on relevance. Do not provide explanations."""
            else:
                # exp_10, 11: select exact number
                system_prompt = """You are a legal retrieval assistant. Your task is to analyze the user's query and the provided candidate articles.
                            Select the most relevant articles based on the user's specific instruction (top-1 or top-2).
                            CRITICAL INSTRUCTION: Output ONLY the Article IDs (aids) in JSON format like this: {"aids": ["aid1", "aid2", ...]}. 
                            Follow the user's instruction exactly for the number of articles to select. Do not provide explanations."""
            
            # User prompts for exp_10, 11, 12
            if self.exp_num == 10:
                user_prompt = f"""Query: {query}

Candidate Articles:
{context_str}

Which article is the most relevant to the query? Select only the single most relevant article."""
            elif self.exp_num == 11:
                user_prompt = f"""Query: {query}

Candidate Articles:
{context_str}

Which articles are relevant to the query? Select exactly the top 2 most relevant articles."""
            elif self.exp_num == 12:
                user_prompt = f"""Query: {query}

Candidate Articles:
{context_str}

Which articles are relevant to the query? Select the most relevant articles (from 1 to 3 articles)."""
        else:
            # System prompt for exp_7 & exp_9 (default)
            system_prompt = """You are a legal retrieval assistant. Your task is to analyze the user's query and the provided candidate articles.
                        Identify ALL articles that are relevant and directly answer the query. You can select 1, 2, 3, or more articles depending on relevance.
                        CRITICAL INSTRUCTION: Output ONLY the Article IDs (aids) in JSON format like this: {"aids": ["aid1", "aid2", ...]}. 
                        If only one article is relevant, return: {"aids": ["aid1"]}. Do not provide explanations."""
            
            # User prompt for exp_7 & exp_9
            user_prompt = f"""Query: {query}

Candidate Articles:
{context_str}

Which articles are relevant to the query? Select all articles that are relevant (can be 1, 2, 3, or more)."""
        
        # Log the prompt being sent to Qwen (debug level)
        logger.debug(f"Qwen system prompt: {system_prompt}")
        logger.debug(f"Qwen user prompt (truncated): {user_prompt[:500]}...")
        
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
        
        # Log raw output from Qwen via logger
        logger.debug(f"Qwen raw output: {generated_text}")
        
        # Parse JSON output - LLM can return multiple articles
        selected_aids = []
        try:
            # Try to parse JSON directly first
            result = json.loads(generated_text)
            # Support both new format {"aids": [...]} and old format {"aid": "..."}
            if 'aids' in result:
                selected_aids = result.get('aids', [])
            elif 'aid' in result:
                # Backward compatibility with old format
                aid = result.get('aid', '')
                if aid:
                    selected_aids = [aid]
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from text
            try:
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = generated_text[json_start:json_end]
                    result = json.loads(json_str)
                    if 'aids' in result:
                        selected_aids = result.get('aids', [])
                    elif 'aid' in result:
                        aid = result.get('aid', '')
                        if aid:
                            selected_aids = [aid]
                else:
                    # Fallback: try to extract aids from text
                    import re
                    # Try to find array format: ["aid1", "aid2", "aid3"]
                    array_match = re.search(r'\["([^"]+)"(?:\s*,\s*"([^"]+)")*\]', generated_text)
                    if array_match:
                        # Extract all quoted strings in the array
                        selected_aids = re.findall(r'"([^"]+)"', generated_text[json_start:json_end] if json_start >= 0 else generated_text)
                    else:
                        # Try single aid format: {"aid": "..."}
                        match = re.search(r'"aid"\s*:\s*"([^"]+)"', generated_text)
                        if match:
                            selected_aids = [match.group(1)]
                        else:
                            # Try to find aids array in the JSON-like structure
                            aids_match = re.search(r'"aids"\s*:\s*\["([^"]+)"(?:\s*,\s*"([^"]+)")*\]', generated_text)
                            if aids_match:
                                selected_aids = re.findall(r'"([^"]+)"', aids_match.group(0))
                            else:
                                # Last resort: try to extract any quoted strings that look like aids
                                matches = re.findall(r'"([^"]+)"', generated_text)
                                if matches:
                                    selected_aids = matches[:10]  # Limit to 10 aids max
            except Exception:
                # Last fallback: try to extract any numbers or strings that might be aids
                import re
                # Try to find any quoted strings or numbers
                matches = re.findall(r'"([^"]+)"', generated_text)
                if matches:
                    selected_aids = matches[:5]
        
        # Filter to only include valid candidate aids and return results
        if not reranker_results:
            return []
        
        valid_aids = [aid for aid in selected_aids if aid in candidate_aids]
        
        if valid_aids:
            # LLM successfully selected valid candidates
            # Return in order of selection, with score 1.0
            return [(aid, 1.0) for aid in valid_aids]
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