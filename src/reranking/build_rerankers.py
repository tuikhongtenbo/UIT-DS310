"""
Build Rerankers from Config
Helper functions to initialize rerankers from config.yaml
"""
from typing import Dict, Any, List, Optional
import yaml
from .single_reranker import SingleReranker
from .ensemble_reranker import EnsembleReranker
from .qwen_reranker import QwenReranker


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_single_reranker(
    model_config: Dict[str, Any],
    device: Optional[str] = None
) -> SingleReranker:
    """
    Build a SingleReranker from config.
    """
    model_name = model_config.get("model_name")
    if not model_name:
        raise ValueError("model_name is required in model_config")
    
    # Use device from parameter, then config, then default
    device = device or model_config.get("device")
    
    return SingleReranker(
        model_name=model_name,
        device=device,
        trust_remote_code=True,
        max_length=model_config.get("max_length", 512)
    )


def build_ensemble_reranker(
    config: Dict[str, Any],
    device: Optional[str] = None,
    require_all_models: bool = True
) -> Optional[EnsembleReranker]:
    """
    Build EnsembleReranker from config.
    """
    reranker_config = config.get("reranker", {})
    ensemble_config = reranker_config.get("ensemble", {})
    
    # Check if ensemble is enabled
    if not ensemble_config.get("enabled", False):
        return None
    
    # Get enabled models
    models_config = reranker_config.get("models", {})
    enabled_models = []
    required_models = ["gte", "bge_v2", "jina"]
    
    for model_key in required_models:
        model_config = models_config.get(model_key, {})
        if model_config.get("enabled", False):
            model_name = model_config.get("model_name")
            if model_name:
                enabled_models.append(model_name)
            else:
                print(f"Warning: {model_key} is enabled but model_name is missing")
    
    # Validate: require all 3 models if require_all_models is True
    if require_all_models and len(enabled_models) < len(required_models):
        enabled_keys = [k for k in required_models if models_config.get(k, {}).get("enabled", False)]
        missing = [m for m in required_models if m not in enabled_keys]
        raise ValueError(
            f"Ensemble reranker requires all 3 models to be enabled. "
            f"Missing models: {missing}. "
            f"Currently enabled: {enabled_keys} ({len(enabled_models)}/{len(required_models)})"
        )
    
    if not enabled_models:
        return None
    
    # Get RRF k parameter
    method = ensemble_config.get("method", "rrf")
    if method == "rrf":
        rrf_k = ensemble_config.get("rrf", {}).get("k", 60)
    else:
        rrf_k = 60  # Default
    
    # Get device
    device = device or reranker_config.get("embedder", {}).get("device")
    
    # Build model configs dict for EnsembleReranker
    model_configs = {}
    for model_name in enabled_models:
        # Find config for this model
        for model_key in ["gte", "bge_v2", "jina"]:
            model_config = models_config.get(model_key, {})
            if model_config.get("model_name") == model_name:
                model_configs[model_name] = {
                    "device": device or model_config.get("device"),
                    "max_length": model_config.get("max_length", 512),
                    "batch_size": model_config.get("batch_size", 16)
                }
                break
    
    # Build ensemble reranker
    return EnsembleReranker(
        reranker_models=enabled_models,
        rrf_k=rrf_k,
        trust_remote_code=True,
        model_configs=model_configs
    )


def build_qwen_reranker(
    config: Dict[str, Any],
    device: Optional[str] = None
) -> Optional[QwenReranker]:
    """
    Build QwenReranker from config (if enabled).
    """
    reranker_config = config.get("reranker", {})
    qwen_config = reranker_config.get("qwen", {})
    
    # Check if Qwen is enabled
    if not qwen_config.get("enabled", False):
        return None
    
    # Get device: parameter > qwen config > embedder config
    device = device or qwen_config.get("device") or reranker_config.get("embedder", {}).get("device")
    
    # Build QwenReranker with config
    qwen = QwenReranker(
        model_name=qwen_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        device=device,
        threshold=qwen_config.get("threshold", 0.8),
        max_new_tokens=qwen_config.get("max_new_tokens", 50),
        temperature=qwen_config.get("temperature", 0.1)
    )
    
    # Set max_content_length if specified
    if "max_content_length" in qwen_config:
        qwen.max_content_length = qwen_config.get("max_content_length", 2000)
    
    return qwen


def build_all_rerankers(
    config_path: str = "config/config.yaml",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build all rerankers from config file.
    """
    config = load_config(config_path)
    reranker_config = config.get("reranker", {})
    models_config = reranker_config.get("models", {})
    
    result = {
        "ensemble": None,
        "qwen": None,
        "singles": {}
    }
    
    # Build ensemble reranker
    result["ensemble"] = build_ensemble_reranker(config, device)
    
    # Build QwenReranker if enabled
    result["qwen"] = build_qwen_reranker(config, device)
    
    # Build individual rerankers 
    for model_key in ["gte", "bge_v2", "jina"]:
        model_config = models_config.get(model_key, {})
        if model_config.get("enabled", False):
            try:
                result["singles"][model_key] = build_single_reranker(
                    model_config,
                    device
                )
            except Exception as e:
                print(f"Warning: Failed to build {model_key} reranker: {e}")
                result["singles"][model_key] = None
    
    return result


def get_reranker(
    config_path: str = "config/config.yaml",
    device: Optional[str] = None
) -> EnsembleReranker:
    """
    Get ensemble reranker (always 3 models: GTE, BGE-v2, Jina).
    """
    config = load_config(config_path)
    ensemble = build_ensemble_reranker(
        config, 
        device=device, 
        require_all_models=True  
    )
    
    if ensemble is None:
        raise ValueError(
            "Ensemble reranker is not available. "
            "Please ensure reranker.ensemble.enabled=true and "
            "all 3 models (gte, bge_v2, jina) are enabled in config.yaml"
        )
    
    return ensemble