"""Configuration management for the fact-checking system."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    phobert_model: str = "vinai/phobert-base"
    xlm_roberta_model: str = "xlm-roberta-base"
    marianmt_model: str = "Helsinki-NLP/opus-mt-vi-en"
    llm_provider: str = "gemini"  # "gemini", "groq", or "local"
    llm_model: str = "gemini-1.5-flash"
    max_length: int = 512
    batch_size: int = 8
    use_quantization: bool = True
    quantization_bits: int = 8


@dataclass
class SearchConfig:
    """Configuration for search and crawling."""
    google_api_key: str = ""
    google_search_engine_id: str = ""
    max_search_results: int = 10
    rate_limit_rpm: int = 15
    cache_ttl_hours: int = 24
    approved_sources: list = field(default_factory=lambda: [
        "vnexpress.net",
        "vtv.vn",
        "vov.vn",
        "tuoitre.vn",
        "thanhnien.vn",
        "baochinhphu.vn"
    ])


@dataclass
class AgentConfig:
    """Configuration for ReAct agent."""
    max_iterations: int = 10
    max_evidence_pieces: int = 20
    min_evidence_pieces: int = 3
    confidence_threshold: float = 0.7
    gemini_api_key: str = ""
    groq_api_key: str = ""


@dataclass
class ColabConfig:
    """Configuration for Colab Pro environment."""
    drive_mount_path: str = "/content/drive/MyDrive/vietnamese-fact-checking"
    checkpoint_interval_minutes: int = 30
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    max_gpu_memory_gb: float = 15.0


@dataclass
class SystemConfig:
    """Main system configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)
    log_level: str = "INFO"
    random_seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            search=SearchConfig(**data.get('search', {})),
            agent=AgentConfig(**data.get('agent', {})),
            colab=ColabConfig(**data.get('colab', {})),
            log_level=data.get('log_level', 'INFO'),
            random_seed=data.get('random_seed', 42)
        )
    
    @classmethod
    def from_json(cls, path: str) -> "SystemConfig":
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            search=SearchConfig(**data.get('search', {})),
            agent=AgentConfig(**data.get('agent', {})),
            colab=ColabConfig(**data.get('colab', {})),
            log_level=data.get('log_level', 'INFO'),
            random_seed=data.get('random_seed', 42)
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Load API keys from environment
        config.search.google_api_key = os.getenv('GOOGLE_API_KEY', '')
        config.search.google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
        config.agent.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        config.agent.groq_api_key = os.getenv('GROQ_API_KEY', '')
        
        # Override other settings if provided
        if os.getenv('LOG_LEVEL'):
            config.log_level = os.getenv('LOG_LEVEL')
        if os.getenv('RANDOM_SEED'):
            config.random_seed = int(os.getenv('RANDOM_SEED'))
        
        return config


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Load system configuration from file or environment.
    
    Args:
        config_path: Path to config file (YAML or JSON). If None, loads from environment.
    
    Returns:
        SystemConfig object
    """
    if config_path is None:
        return SystemConfig.from_env()
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix in ['.yaml', '.yml']:
        return SystemConfig.from_yaml(config_path)
    elif path.suffix == '.json':
        return SystemConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
