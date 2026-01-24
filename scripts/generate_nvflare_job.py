#!/usr/bin/env python3
"""
Generate NVFlare Job from Flower App.

This script creates a proper NVFlare job structure from the SFL Flower app
that can be submitted to a POC or production NVFlare deployment.

Usage:
    python scripts/generate_nvflare_job.py --output /tmp/nvflare/poc/jobs/sfl-job
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_minimal_flower_content(project_root: Path, temp_dir: Path) -> Path:
    """Create a minimal Flower content directory without .venv."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy only necessary files
    files_to_copy = [
        "pyproject.toml",
    ]
    
    dirs_to_copy = [
        "src",
        "config",
    ]
    
    for f in files_to_copy:
        src = project_root / f
        if src.exists():
            shutil.copy(src, temp_dir / f)
    
    for d in dirs_to_copy:
        src = project_root / d
        if src.exists():
            if (temp_dir / d).exists():
                shutil.rmtree(temp_dir / d)
            shutil.copytree(src, temp_dir / d, ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '.pytest_cache', '.mypy_cache'
            ))
    
    return temp_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate NVFlare job from Flower app"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="/tmp/nvflare/poc/jobs/sfl-job",
        help="Output directory for the NVFlare job",
    )
    parser.add_argument(
        "--num-clients", "-n",
        type=int,
        default=2,
        help="Minimum number of clients",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="sfl-federated-sum",
        help="Name of the job",
    )
    
    args = parser.parse_args()
    
    # Import NVFlare
    try:
        from nvflare.app_opt.flower.flower_job import FlowerJob
    except ImportError as e:
        print(f"Error: Failed to import NVFlare: {e}")
        print("Install with: pip install nvflare==2.7.1")
        return 1
    
    # Get project root (where pyproject.toml is)
    project_root = Path(__file__).parent.parent
    
    if not (project_root / "pyproject.toml").exists():
        print(f"Error: pyproject.toml not found in {project_root}")
        return 1
    
    print(f"Creating NVFlare job from Flower app at {project_root}")
    print(f"  Job name: {args.job_name}")
    print(f"  Min clients: {args.num_clients}")
    print(f"  Output: {args.output}")
    
    # Create minimal content directory (without .venv)
    temp_content = Path("/tmp/sfl-flower-content")
    if temp_content.exists():
        shutil.rmtree(temp_content)
    
    print(f"\nCreating minimal Flower content...")
    create_minimal_flower_content(project_root, temp_content)
    
    # Create FlowerJob
    job = FlowerJob(
        name=args.job_name,
        flower_content=str(temp_content),
        min_clients=args.num_clients,
    )
    
    # Export the job
    output_path = Path(args.output)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting job to {output_path}...")
    job.export_job(str(output_path))
    
    # Cleanup temp
    shutil.rmtree(temp_content)
    
    print(f"\n✅ Job exported successfully!")
    print(f"\nTo submit to POC:")
    print(f"  1. Connect to admin: nvflare poc start -p admin")
    print(f"  2. Submit job: submit_job {args.job_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
