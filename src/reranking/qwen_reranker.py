"""
Qwen Reranker

Qwen2.5 model for reranking documents using LLM inference.
Supports both transformers and vLLM backends.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    Qwen2.5 based reranker using Transformers (Standard Backend).
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 50,
        temperature: float = 0.1,
        threshold: float = 0.8,
        use_4bit: bool = True,
        exp_num: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize QwenReranker.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.threshold = threshold
        self.max_content_length = 1500
        self.use_4bit = use_4bit and BITSANDBYTES_AVAILABLE and device == "cuda"
        self.exp_num = exp_num
        self.max_candidates_for_llm = 20
        self.verbose = verbose
        
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
    
    def _load_model(self) -> None:
        """Lazy load the model and tokenizer."""
        if self._model_loaded:
            return
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.use_4bit:
            logger.info("Loading Qwen model with 4-bit quantization...")
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
        else:
            logger.info("Loading Qwen model with full precision...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        self._model_loaded = True
    
    @staticmethod
    def build_documents_dict(documents: List[Dict[str, Any]], id_key: str = "aid") -> Dict[str, Dict[str, Any]]:
        documents_dict = {}
        for idx, doc in enumerate(documents):
            doc_id = doc.get(id_key, doc.get("id", f"doc_{idx}"))
            if doc_id:
                documents_dict[str(doc_id)] = doc
        return documents_dict
    
    def _truncate_content(self, content: str, max_length: Optional[int] = None) -> str:
        """
        Truncate content to maximum length.
        """
        if max_length is None:
            max_length = self.max_content_length
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."
    
    def _build_context_string(
        self,
        reranker_results: List[Tuple[str, float]],
        documents_dict: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        Build context string from reranker results and documents.
        
        Args:
            reranker_results: List of (aid, score) tuples
            documents_dict: Dictionary mapping aid to document
            
        Returns:
            Tuple of (context_string, candidate_aids_list)
        """
        context_str = ""
        candidate_aids = []
        
        for aid, score in reranker_results:
            aid_str = str(aid)
            if aid_str not in documents_dict:
                continue
            
            doc = documents_dict[aid_str]
            content = doc.get('content', doc.get('text', ''))
            
            if content:
                truncated = self._truncate_content(content)
                context_str += (
                    f"ID điều luật: {aid_str}\n"
                    f"Điểm số: {score:.4f}\n"
                    f"Nội dung: {truncated}\n---\n"
                )
                candidate_aids.append(aid_str)
        
        return context_str, candidate_aids
    
    def _build_prompts(
        self,
        query: str,
        context_str: str,
        top_k: int
    ) -> tuple:
        """
        Build system and user prompts based on exp_num.
        
        Args:
            query: User query/question
            context_str: Context string with candidate articles
            top_k: Top-k value for selection
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        base_instruction = (
            "Bạn là trợ lý tra cứu pháp luật chuyên về Luật Việt Nam.\n"
            "Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và các điều luật ứng viên được cung cấp (bằng tiếng Việt)."
        )
        
        if self.exp_num == 7:
            sys_p = f"""{base_instruction}
Nhiệm vụ của bạn là LỌC BỎ các điều luật KHÔNG liên quan đến câu hỏi, chỉ giữ lại những điều luật ĐÚNG và liên quan.
HƯỚNG DẪN QUAN TRỌNG: 
- Phân tích từng điều luật ứng viên và xác định xem nó có thực sự liên quan đến câu hỏi hay không
- CHỈ giữ lại những điều luật mà bạn chắc chắn là liên quan và đúng
- Nếu TẤT CẢ các điều luật đều liên quan và đúng, hãy giữ lại TẤT CẢ
- CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp
- Chỉ xuất ra ID của các điều luật được giữ lại (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}
- Nếu không có điều luật nào liên quan, trả về: {{"aids": []}}
- Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Hãy phân tích từng điều luật ứng viên và LỌC BỎ những điều luật KHÔNG liên quan đến câu hỏi. 
CHỈ giữ lại những điều luật mà bạn chắc chắn là liên quan và đúng.
Nếu TẤT CẢ các điều luật đều liên quan, hãy giữ lại TẤT CẢ.
CHỈ được chọn từ danh sách các điều luật ứng viên ở trên."""
        
        elif self.exp_num == 10 or (self.exp_num is None and top_k == 1):
            sys_p = f"""{base_instruction}
Chọn điều luật liên quan nhất dựa trên yêu cầu cụ thể của người dùng (top-1).
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1"]}}. 
Bạn phải chọn chính xác 1 điều luật. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Điều luật nào liên quan nhất đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn chỉ một điều luật liên quan nhất."""
        
        elif self.exp_num == 11 or (self.exp_num is None and top_k == 2):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất dựa trên yêu cầu cụ thể của người dùng (top-2).
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2"]}}. 
Bạn phải chọn chính xác 2 điều luật. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn chính xác 2 điều luật liên quan nhất."""
        
        elif self.exp_num == 12 or (self.exp_num is None and top_k == 3):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất (từ 1 đến 3 điều luật) dựa trên mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Bạn có thể chọn 1, 2, hoặc 3 điều luật tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn các điều luật liên quan nhất (từ 1 đến 3 điều luật)."""
        
        elif self.exp_num == 13 or (self.exp_num is None and top_k == 4):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất (từ 1 đến 4 điều luật) dựa trên mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Bạn có thể chọn 1, 2, 3, hoặc 4 điều luật tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn các điều luật liên quan nhất (từ 1 đến 4 điều luật)."""
        
        elif self.exp_num == 14 or (self.exp_num is None and top_k == 5):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất (từ 1 đến 5 điều luật) dựa trên mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Bạn có thể chọn 1, 2, 3, 4, hoặc 5 điều luật tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn các điều luật liên quan nhất (từ 1 đến 5 điều luật)."""
        
        elif self.exp_num == 15 or (self.exp_num is None and top_k == 6):
            sys_p = f"""{base_instruction}
Chọn TẤT CẢ các điều luật liên quan đến câu hỏi mà bạn thấy có liên quan. Không có giới hạn số lượng - hãy chọn tất cả các điều luật mà bạn đánh giá là liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Chọn tất cả các điều luật liên quan, có thể là 1, 2, 3, 5, 10, hoặc nhiều hơn tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn TẤT CẢ các điều luật mà bạn thấy liên quan, không giới hạn số lượng."""
        
        else:
            sys_p = f"""{base_instruction}
Xác định TẤT CẢ các điều luật liên quan và trực tiếp trả lời câu hỏi. Bạn có thể chọn 1, 2, 3, hoặc nhiều điều luật tùy theo mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Nếu chỉ có một điều luật liên quan, trả về: {{"aids": ["aid1"]}}. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn tất cả các điều luật liên quan (có thể là 1, 2, 3, hoặc nhiều hơn)."""
        
        return sys_p, user_p
    
    def _parse_output(self, gen_text: str, candidate_aids: List[str]) -> List[str]:
        """
        Parse model output to extract selected aids.
        
        Args:
            gen_text: Generated text from model
            candidate_aids: List of candidate aid strings
            
        Returns:
            List of valid selected aid strings
        """
        selected_aids = []
        try:
            result = json.loads(gen_text)
            if 'aids' in result:
                selected_aids = result['aids']
            elif 'aid' in result:
                selected_aids = [result['aid']]
        except json.JSONDecodeError:
            matches = re.findall(r'"([^"]+)"', gen_text)
            if matches:
                selected_aids = matches[:10]
        
        return [aid for aid in selected_aids if aid in candidate_aids]
    
    def rerank(
        self,
        query: str,
        reranker_results: List[Tuple[str, float]],
        documents_dict: Dict[str, Dict[str, Any]],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents using Qwen LLM.
        
        Args:
            query: User query/question
            reranker_results: List of (aid, score) tuples from initial reranker
            documents_dict: Dictionary mapping aid to document
            top_k: Number of top results to return
            
        Returns:
            List of (aid, score) tuples for selected documents
        """
        if not reranker_results:
            return []
        
        sorted_results = sorted(reranker_results, key=lambda x: x[1], reverse=True)
        return self._rerank_with_llm(query, sorted_results, documents_dict, top_k)
    
    def _rerank_with_llm(
        self,
        query: str,
        reranker_results: List[Tuple[str, float]],
        documents_dict: Dict[str, Dict[str, Any]],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Internal method to perform LLM-based reranking.
        
        Args:
            query: User query/question
            reranker_results: List of (aid, score) tuples
            documents_dict: Dictionary mapping aid to document
            top_k: Number of top results to return
            
        Returns:
            List of (aid, score) tuples for selected documents
        """
        if not reranker_results:
            return []
        
        self._load_model()
        
        context_str, candidate_aids = self._build_context_string(reranker_results, documents_dict)
        
        if not candidate_aids:
            return [reranker_results[0]]
        
        sys_p, user_p = self._build_prompts(query, context_str, top_k)
        
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p}
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        gen_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if self.verbose:
            logger.info(f"Model output (raw): {gen_text}")
        else:
            logger.debug(f"Model output: {gen_text}")
        
        valid_aids = self._parse_output(gen_text, candidate_aids)
        
        if valid_aids:
            return [(aid, 1.0) for aid in valid_aids]
        return [reranker_results[0]]
    
    def select_article(
        self,
        query: str,
        reranker_results: List[Tuple[str, float]],
        documents: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
    ) -> Tuple[Optional[str], float]:
        """
        Select the most relevant article using threshold-based logic.
        
        Args:
            query: User query/question
            reranker_results: List of (aid, score) tuples
            documents: List of documents or dictionary mapping aid to document
            
        Returns:
            Tuple of (selected_aid, score) or (None, 0.0) if no result
        """
        if isinstance(documents, list):
            documents_dict = self.build_documents_dict(documents)
        else:
            documents_dict = documents
        
        results = self.rerank(query, reranker_results, documents_dict, top_k=1)
        if results:
            return results[0]
        return (None, 0.0)


class QwenRerankerVLLM:
    """
    Qwen Reranker using vLLM backend for faster inference.
    
    This class uses vLLM for high-throughput LLM serving with better
    memory management and performance compared to transformers.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        max_content_length: int = 1500,
        exp_num: Optional[int] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_candidates_for_llm: int = 20,
        verbose: bool = False
    ):
        """
        Initialize QwenRerankerVLLM.
        
        Args:
            model_name: HuggingFace model name or path
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            max_content_length: Maximum content length per document
            exp_num: Experiment number for prompt selection
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_candidates_for_llm: Maximum number of candidates to process
            verbose: Whether to print detailed logs
        """
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(f"vLLM not installed or incompatible. Error: {e}")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_content_length = max_content_length
        self.exp_num = exp_num
        self.max_candidates_for_llm = max_candidates_for_llm
        self.verbose = verbose
        
        logger.info(f"Initializing vLLM engine for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=16384,
            dtype="float16",
            enforce_eager=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        logger.info("✅ QwenReranker with vLLM initialized!")
    
    def build_documents_dict(
        self,
        documents: List[Dict[str, Any]],
        id_key: str = "aid"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a dictionary mapping document IDs to documents.
        
        Args:
            documents: List of document dictionaries
            id_key: Key to use for document ID
            
        Returns:
            Dictionary mapping document IDs to documents
        """
        return {str(d.get(id_key)): d for d in documents if d.get(id_key)}
    
    def _truncate_content(self, content: str, max_length: Optional[int] = None) -> str:
        """
        Truncate content to maximum length.
        
        Args:
            content: Content string to truncate
            max_length: Maximum length (uses self.max_content_length if None)
            
        Returns:
            Truncated content string
        """
        if max_length is None:
            max_length = self.max_content_length
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."
    
    def _parse_json_output_vllm(
        self,
        generated_text: str,
        candidate_aids: List[str]
    ) -> List[str]:
        """
        Parse JSON output from vLLM with improved fallback handling.
        
        Args:
            generated_text: Generated text from model
            candidate_aids: List of candidate aid strings
            
        Returns:
            List of valid selected aid strings
        """
        selected_aids = []
        try:
            json_match = re.search(r'\{[^{}]*"aids"[^{}]*\}', generated_text, re.DOTALL)
            if json_match:
                text_to_parse = json_match.group(0)
                result = json.loads(text_to_parse)
                if 'aids' in result:
                    selected_aids = result['aids']
                elif 'aid' in result:
                    selected_aids = [result['aid']]
            else:
                result = json.loads(generated_text)
                if 'aids' in result:
                    selected_aids = result['aids']
                elif 'aid' in result:
                    selected_aids = [result['aid']]
        except json.JSONDecodeError:
            matches = re.findall(r'"([^"]+)"', generated_text)
            selected_aids = [
                m for m in matches
                if any(c.isdigit() for c in m) or m.isdigit()
            ]
        except Exception as e:
            logger.debug(f"Error parsing JSON output: {e}. Trying regex fallback...")
            matches = re.findall(r'"([^"]+)"', generated_text)
            selected_aids = [
                m for m in matches
                if any(c.isdigit() for c in m) or m.isdigit()
            ]
        
        return [aid for aid in selected_aids if str(aid) in candidate_aids]
    
    def _build_context_string(
        self,
        reranker_results: List[Tuple[str, float]],
        documents_dict: Dict[str, Dict[str, Any]]
    ) -> tuple:
        """
        Build context string from reranker results and documents.
        
        Args:
            reranker_results: List of (aid, score) tuples
            documents_dict: Dictionary mapping aid to document
            
        Returns:
            Tuple of (context_string, candidate_aids_list)
        """
        context_str = ""
        candidate_aids = []
        
        for aid, score in reranker_results[:self.max_candidates_for_llm]:
            aid_str = str(aid)
            if aid_str not in documents_dict:
                continue
            
            doc = documents_dict[aid_str]
            content = self._truncate_content(
                doc.get('content', doc.get('text', '')),
                max_length=1500
            )
            title = doc.get('title', '')
            law_name = f" ({title})" if title else ""
            context_str += (
                f"ID điều luật: {aid_str}{law_name}\n"
                f"Điểm số: {score:.4f}\n"
                f"Nội dung: {content}\n---\n"
            )
            candidate_aids.append(aid_str)
        
        return context_str, candidate_aids
    
    def _build_prompts(
        self,
        query: str,
        context_str: str,
        top_k: int
    ) -> tuple:
        """
        Build system and user prompts based on exp_num.
        
        Args:
            query: User query/question
            context_str: Context string with candidate articles
            top_k: Top-k value for selection
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        base_instruction = (
            "Bạn là trợ lý tra cứu pháp luật chuyên về Luật Việt Nam.\n"
            "Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và các điều luật ứng viên được cung cấp (bằng tiếng Việt)."
        )
        
        if self.exp_num == 7:
            sys_p = f"""{base_instruction}
Nhiệm vụ của bạn là LỌC BỎ các điều luật KHÔNG liên quan đến câu hỏi, chỉ giữ lại những điều luật ĐÚNG và liên quan.
HƯỚNG DẪN QUAN TRỌNG: 
- Phân tích từng điều luật ứng viên và xác định xem nó có thực sự liên quan đến câu hỏi hay không
- CHỈ giữ lại những điều luật mà bạn chắc chắn là liên quan và đúng
- Nếu TẤT CẢ các điều luật đều liên quan và đúng, hãy giữ lại TẤT CẢ
- CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp
- Chỉ xuất ra ID của các điều luật được giữ lại (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}
- Nếu không có điều luật nào liên quan, trả về: {{"aids": []}}
- Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Hãy phân tích từng điều luật ứng viên và LỌC BỎ những điều luật KHÔNG liên quan đến câu hỏi. 
CHỈ giữ lại những điều luật mà bạn chắc chắn là liên quan và đúng.
Nếu TẤT CẢ các điều luật đều liên quan, hãy giữ lại TẤT CẢ.
CHỈ được chọn từ danh sách các điều luật ứng viên ở trên."""
        
        elif self.exp_num == 10 or (self.exp_num is None and top_k == 1):
            sys_p = f"""{base_instruction}
Chọn điều luật liên quan nhất dựa trên yêu cầu cụ thể của người dùng (top-1).
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1"]}}. 
Bạn phải chọn chính xác 1 điều luật. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Điều luật nào liên quan nhất đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn chỉ một điều luật liên quan nhất."""
        
        elif self.exp_num == 11 or (self.exp_num is None and top_k == 2):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất dựa trên yêu cầu cụ thể của người dùng (top-2).
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2"]}}. 
Bạn phải chọn chính xác 2 điều luật. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn chính xác 2 điều luật liên quan nhất."""
        
        elif self.exp_num == 12 or (self.exp_num is None and top_k == 3):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất (từ 1 đến 3 điều luật) dựa trên mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Bạn có thể chọn 1, 2, hoặc 3 điều luật tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn các điều luật liên quan nhất (từ 1 đến 3 điều luật)."""
        
        elif self.exp_num == 13 or (self.exp_num is None and top_k == 4):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất (từ 1 đến 4 điều luật) dựa trên mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Bạn có thể chọn 1, 2, 3, hoặc 4 điều luật tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn các điều luật liên quan nhất (từ 1 đến 4 điều luật)."""
        
        elif self.exp_num == 14 or (self.exp_num is None and top_k == 5):
            sys_p = f"""{base_instruction}
Chọn các điều luật liên quan nhất (từ 1 đến 5 điều luật) dựa trên mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Bạn có thể chọn 1, 2, 3, 4, hoặc 5 điều luật tùy theo mức độ liên quan. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn các điều luật liên quan nhất (từ 1 đến 5 điều luật)."""
        
        elif self.exp_num == 15 or (self.exp_num is None and top_k == 6):
            sys_p = f"""{base_instruction}
Chọn TẤT CẢ các điều luật liên quan đến câu hỏi mà bạn thấy có liên quan. Không có giới hạn số lượng - hãy chọn tất cả các điều luật mà bạn đánh giá là liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Chọn tất cả các điều luật liên quan, có thể là 1, 2, 3, 5, 10, hoặc số lượng khác tùy theo suy luận của bạn. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn TẤT CẢ các điều luật mà bạn thấy liên quan, không giới hạn số lượng."""
        
        else:
            sys_p = f"""{base_instruction}
Xác định TẤT CẢ các điều luật liên quan và trực tiếp trả lời câu hỏi. Bạn có thể chọn 1, 2, 3, hoặc nhiều điều luật tùy theo mức độ liên quan.
HƯỚNG DẪN QUAN TRỌNG: CHỈ được chọn từ danh sách các điều luật ứng viên đã cung cấp. Chỉ xuất ra ID của điều luật (aids) dưới dạng JSON như sau: {{"aids": ["aid1", "aid2", ...]}}. 
Nếu chỉ có một điều luật liên quan, trả về: {{"aids": ["aid1"]}}. Không được giải thích thêm."""
            
            user_p = f"""Câu hỏi: {query}

Các điều luật ứng viên:
{context_str}

Những điều luật nào liên quan đến câu hỏi? CHỈ được chọn từ danh sách các điều luật ứng viên ở trên. Chọn tất cả các điều luật liên quan (có thể là 1, 2, 3, hoặc nhiều hơn)."""
        
        return sys_p, user_p
    
    def _truncate_prompt_if_needed(
        self,
        text_input: str,
        query: str,
        reranker_results: List[Tuple[str, float]],
        documents_dict: Dict[str, Dict[str, Any]],
        top_k: int
    ) -> str:
        """
        Truncate prompt dynamically if it exceeds maximum length.
        
        Args:
            text_input: Initial prompt text
            query: User query/question
            reranker_results: List of (aid, score) tuples
            documents_dict: Dictionary mapping aid to document
            top_k: Top-k value for selection
            
        Returns:
            Truncated prompt text
        """
        input_ids = self.tokenizer.encode(text_input, return_tensors="pt")
        prompt_length = input_ids.shape[1]
        max_allowed_tokens = 16000
        
        if prompt_length <= max_allowed_tokens:
            return text_input
        
        logger.warning(
            f"Prompt length ({prompt_length}) exceeds limit ({max_allowed_tokens}). "
            "Truncating content dynamically..."
        )
        
        old_max_length = self.max_content_length
        max_iterations = 10
        
        for iteration in range(max_iterations):
            reduction = 300 * (iteration + 1)
            self.max_content_length = max(300, old_max_length - reduction)
            
            context_str, candidate_aids = self._build_context_string(
                reranker_results, documents_dict
            )
            
            sys_p, user_p = self._build_prompts(query, context_str, top_k)
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p}
            ]
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            input_ids = self.tokenizer.encode(text_input, return_tensors="pt")
            prompt_length = input_ids.shape[1]
            
            if prompt_length <= max_allowed_tokens:
                logger.info(
                    f"Prompt truncated successfully: {prompt_length} tokens "
                    f"(reduced max_content_length to {self.max_content_length})"
                )
                break
            elif iteration == max_iterations - 1:
                logger.warning(
                    f"Still too long after {max_iterations} iterations "
                    f"({prompt_length} tokens). Using truncated version anyway."
                )
        
        self.max_content_length = old_max_length
        return text_input
    
    def rerank(
        self,
        query: str,
        reranker_results: List[Tuple[str, float]],
        documents_dict: Dict[str, Dict[str, Any]],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents using Qwen LLM with vLLM backend.
        
        Args:
            query: User query/question
            reranker_results: List of (aid, score) tuples from initial reranker
            documents_dict: Dictionary mapping aid to document
            top_k: Number of top results to return
            
        Returns:
            List of (aid, score) tuples for selected documents
        """
        if not reranker_results:
            return []
        
        context_str, candidate_aids = self._build_context_string(
            reranker_results, documents_dict
        )
        
        if not candidate_aids:
            return [reranker_results[0]]
        
        sys_p, user_p = self._build_prompts(query, context_str, top_k)
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p}
        ]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        text_input = self._truncate_prompt_if_needed(
            text_input, query, reranker_results, documents_dict, top_k
        )
        
        outputs = self.llm.generate([text_input], self.sampling_params)
        gen_text = outputs[0].outputs[0].text.strip()
        
        if self.verbose:
            logger.info(f"Model output (vLLM, raw): {gen_text}")
        else:
            logger.debug(f"Model output (vLLM): {gen_text}")
        
        selected_aids = self._parse_json_output_vllm(gen_text, candidate_aids)
        valid_aids = [aid for aid in selected_aids if aid in candidate_aids]
        
        if valid_aids:
            return [(aid, 1.0) for aid in valid_aids]
        return [reranker_results[0]]
