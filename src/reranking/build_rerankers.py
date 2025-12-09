"""
Build Rerankers from Config
Helper functions to initialize rerankers from config.yaml
"""

import logging
from typing import Any, Dict, Optional

import yaml

from .ensemble_reranker import EnsembleReranker
from .qwen_reranker import QwenReranker
from .single_reranker import SingleReranker

logger = logging.getLogger(__name__)


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
    require_all_models: bool = False
) -> Optional[EnsembleReranker]:
    """
    Build EnsembleReranker from config.
    
    Args:
        config: Configuration dictionary
        device: Device to use (cuda/cpu)
        require_all_models: If True, requires all 3 models (gte, bge_v2, jina) to be enabled.
                          If False, allows ensemble with any number of enabled models (at least 1).
    
    Returns:
        Initialized EnsembleReranker instance or None if disabled
    """
    reranker_config = config.get("reranker", {})
    ensemble_config = reranker_config.get("ensemble", {})
    
    # Check if ensemble is enabled
    if not ensemble_config.get("enabled", False):
        return None
    
    # Get enabled models
    models_config = reranker_config.get("models", {})
    enabled_models = []
    enabled_keys = []
    required_models = ["gte", "bge_v2", "jina"]
    
    for model_key in required_models:
        model_config = models_config.get(model_key, {})
        if model_config.get("enabled", False):
            model_name = model_config.get("model_name")
            if model_name:
                enabled_models.append(model_name)
                enabled_keys.append(model_key)
            else:
                logger.warning(f"{model_key} is enabled but model_name is missing")
    
    # Validate: require all 3 models if require_all_models is True
    if require_all_models and len(enabled_models) < len(required_models):
        missing = [m for m in required_models if m not in enabled_keys]
        raise ValueError(
            f"Ensemble reranker requires all 3 models to be enabled. "
            f"Missing models: {missing}. "
            f"Currently enabled: {enabled_keys} ({len(enabled_models)}/{len(required_models)})"
        )
    
    # Require at least 1 model to be enabled
    if not enabled_models:
        logger.warning("Ensemble reranker is enabled but no models are enabled. Disabling ensemble.")
        return None
    
    # Log which models are being used
    if len(enabled_models) < len(required_models):
        logger.info(f"Ensemble reranker using {len(enabled_models)}/{len(required_models)} models: {enabled_keys}")
    
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


def build_qwen_reranker(config: Dict[str, Any], device: Optional[str] = None) -> Optional[QwenReranker]:
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
    use_4bit = qwen_config.get("use_4bit", True)
    exp_num = qwen_config.get("exp_num", None)
    
    qwen = QwenReranker(
        model_name=qwen_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        device=device,
        threshold=qwen_config.get("threshold", 0.8),
        max_new_tokens=qwen_config.get("max_new_tokens", 50),
        temperature=qwen_config.get("temperature", 0.1),
        use_4bit=use_4bit,
        exp_num=exp_num
    )
    
    # Set max_content_length if specified
    if "max_content_length" in qwen_config:
        qwen.max_content_length = qwen_config.get("max_content_length", 2000)
    
    return qwen


def build_all_rerankers(config_path: str = "config/config.yaml", device: Optional[str] = None) -> Dict[str, Any]:
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
                logger.warning(f"Failed to build {model_key} reranker: {e}")
                result["singles"][model_key] = None
    
    return result


def get_reranker(config_path: str = "config/config.yaml", device: Optional[str] = None) -> EnsembleReranker:
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