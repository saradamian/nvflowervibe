"""
Content staging for NVFlare job execution.

NVFlare's FlowerRecipe needs a clean directory containing:
- pyproject.toml (with [tool.flwr.app] pointing to the correct apps)
- src/sfl/ (the SFL framework code)
- config/ (optional YAML configs)

This module stages those files from the project root into a temp
directory, avoiding .git/, .venv/, __pycache__, etc.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

# Map app type to Flower entry points
_APP_ENTRYPOINTS = {
    "base": ("sfl.server.app:app", "sfl.client:app"),
    "esm2": ("sfl.esm2:server_app", "sfl.esm2:client_app"),
    "llm": ("sfl.llm:server_app", "sfl.llm:client_app"),
}


def stage_flower_content(
    project_root: Path,
    app_type: str,
    run_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Stage Flower content into a clean temp directory for NVFlare.

    Creates a temporary directory with:
    - ``src/sfl/`` copied from project
    - ``config/`` copied if present
    - ``pyproject.toml`` generated with correct app entry points

    Args:
        project_root: Path to the SFL project root.
        app_type: One of "base", "esm2", "llm".
        run_config: Optional dict of Flower run_config values to embed
            in ``[tool.flwr.app.config]``.

    Returns:
        Path to the staging directory. Caller is responsible for cleanup
        via ``shutil.rmtree()``.

    Raises:
        ValueError: If app_type is not recognized.
    """
    if app_type not in _APP_ENTRYPOINTS:
        raise ValueError(
            f"Unknown app_type {app_type!r}. Must be one of: "
            f"{', '.join(_APP_ENTRYPOINTS)}"
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=f"sfl_{app_type}_nvflare_"))

    # Copy source
    src_dir = project_root / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, staging_dir / "src")
    else:
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    # Copy config if present
    config_dir = project_root / "config"
    if config_dir.exists():
        shutil.copytree(config_dir, staging_dir / "config")

    # Write pyproject.toml
    server_app, client_app = _APP_ENTRYPOINTS[app_type]
    _write_pyproject(staging_dir, server_app, client_app, run_config or {})

    logger.info("Staged Flower content for %s at %s", app_type, staging_dir)
    return staging_dir


def _write_pyproject(
    staging_dir: Path,
    server_app: str,
    client_app: str,
    run_config: Dict[str, Any],
) -> None:
    """Generate a pyproject.toml for Flower app discovery."""
    config_lines = "\n".join(
        f'{k} = {_toml_value(v)}' for k, v in run_config.items()
    )

    content = f"""\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "sfl-nvflare-job"
version = "0.1.0"

[tool.hatch.build.targets.wheel]
packages = ["src/sfl"]

[tool.flwr.app]
publisher = "sfl"

[tool.flwr.app.components]
serverapp = "{server_app}"
clientapp = "{client_app}"

[tool.flwr.app.config]
{config_lines}

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 2
address = "127.0.0.1:9093"
insecure = true
"""
    (staging_dir / "pyproject.toml").write_text(content)


def _toml_value(v: Any) -> str:
    """Format a Python value as a TOML value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f'"{v}"'
