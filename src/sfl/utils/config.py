"""
Configuration management for SFL.

This module provides a unified configuration system that supports:
- YAML configuration files
- Environment variables (SFL_* prefix)
- CLI argument overrides
- Sensible defaults

Configuration Priority (highest to lowest):
1. CLI arguments
2. Environment variables
3. Configuration file
4. Defaults
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

import yaml
from dotenv import load_dotenv

from sfl.types import (
    ClientConfig,
    FederationConfig,
    LoggingConfig,
    NVFlareConfig,
    ServerConfig,
    SFLConfig,
)

# Load .env file if present
load_dotenv()

# Module-level config cache
_config: Optional[SFLConfig] = None

T = TypeVar("T")


def _get_env(key: str, default: T, cast: type = str) -> Union[T, Any]:
    """Get environment variable with type casting.
    
    Args:
        key: Environment variable name (without SFL_ prefix).
        default: Default value if not set.
        cast: Type to cast the value to.
    
    Returns:
        The environment variable value or default.
    """
    env_key = f"SFL_{key.upper()}"
    value = os.getenv(env_key)
    
    if value is None:
        return default
    
    if cast == bool:
        return value.lower() in ("true", "1", "yes", "on")
    
    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary.
        override: Dictionary with override values.
    
    Returns:
        Merged dictionary (base is not modified).
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> SFLConfig:
    """Load configuration from file, environment, and CLI.
    
    Args:
        config_path: Path to YAML configuration file.
        cli_overrides: Dictionary of CLI overrides.
    
    Returns:
        Complete SFLConfig object.
    
    Example:
        >>> config = load_config("config/default.yaml")
        >>> print(config.federation.num_clients)
        2
    """
    global _config
    
    # Start with defaults
    config_dict: Dict[str, Any] = {
        "federation": {},
        "client": {},
        "server": {},
        "nvflare": {},
        "logging": {},
    }
    
    # Load from YAML file if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                file_config = yaml.safe_load(f) or {}
                config_dict = _merge_dict(config_dict, file_config)
    
    # Override with environment variables
    env_overrides = {
        "federation": {
            "num_clients": _get_env("NUM_CLIENTS", None, int),
            "num_rounds": _get_env("NUM_ROUNDS", None, int),
            "min_available_clients": _get_env("MIN_AVAILABLE_CLIENTS", None, int),
        },
        "client": {
            "base_secret": _get_env("CLIENT_BASE_SECRET", None, float),
        },
        "nvflare": {
            "job_name": _get_env("JOB_NAME", None, str),
            "stream_metrics": _get_env("STREAM_METRICS", None, bool),
        },
        "logging": {
            "level": _get_env("LOG_LEVEL", None, str),
            "format": _get_env("LOG_FORMAT", None, str),
        },
    }
    
    # Remove None values
    for section in env_overrides:
        env_overrides[section] = {
            k: v for k, v in env_overrides[section].items() if v is not None
        }
    
    config_dict = _merge_dict(config_dict, env_overrides)
    
    # Apply CLI overrides
    if cli_overrides:
        config_dict = _merge_dict(config_dict, cli_overrides)
    
    # Build config objects
    _config = SFLConfig(
        federation=FederationConfig(**config_dict.get("federation", {})),
        client=ClientConfig(**config_dict.get("client", {})),
        server=ServerConfig(**config_dict.get("server", {})),
        nvflare=NVFlareConfig(**config_dict.get("nvflare", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
    )
    
    return _config


def get_config() -> SFLConfig:
    """Get the current configuration.
    
    Returns:
        Current SFLConfig, or default if not loaded.
    
    Raises:
        RuntimeError: If config has not been loaded yet.
    """
    global _config
    
    if _config is None:
        # Load with defaults
        _config = load_config()
    
    return _config


def reset_config() -> None:
    """Reset the configuration cache.
    
    Useful for testing or reloading configuration.
    """
    global _config
    _config = None
