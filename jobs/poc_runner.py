#!/usr/bin/env python3
"""
NVFlare POC Mode Runner for SFL.

This script sets up and runs the SFL federated learning demo in NVFlare's
POC (Proof of Concept) mode, which uses separate processes for server and
clients - closer to a real production deployment.

Usage:
    python jobs/poc_runner.py prepare --num-clients 4
    python jobs/poc_runner.py start
    python jobs/poc_runner.py submit
    python jobs/poc_runner.py stop
    python jobs/poc_runner.py clean
"""

import argparse
import subprocess
import sys
import shutil
import time
from pathlib import Path


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def get_poc_workspace() -> Path:
    """Get the POC workspace directory."""
    # NVFlare 2.7+ uses /tmp/nvflare/poc by default
    return Path("/tmp/nvflare/poc")


def get_project_root() -> Path:
    """Get the SFL project root."""
    return Path(__file__).parent.parent


def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare POC environment."""
    print(f"\n{'='*60}")
    print("Preparing NVFlare POC Environment")
    print(f"{'='*60}\n")
    
    # Clean existing POC if requested
    poc_workspace = get_poc_workspace()
    if poc_workspace.exists() and args.clean:
        print(f"Cleaning existing POC workspace: {poc_workspace}")
        run_command(["nvflare", "poc", "clean"], check=False)
    
    # Prepare POC with specified number of clients
    print(f"\nPreparing POC with {args.num_clients} clients...")
    run_command(["nvflare", "poc", "prepare", "-n", str(args.num_clients)])
    
    # Generate NVFlare job from Flower app
    project_root = get_project_root()
    job_output = poc_workspace / "jobs" / "sfl-job"
    
    print(f"\nGenerating NVFlare job from Flower app...")
    result = run_command([
        "python", str(project_root / "scripts" / "generate_nvflare_job.py"),
        "--output", str(job_output),
        "--num-clients", str(args.num_clients),
    ], check=False)
    
    if result.returncode != 0:
        print("❌ Failed to generate job. Falling back to direct copy...")
        # Fallback: copy project directly
        jobs_dir = poc_workspace / "jobs" / "sfl"
        if jobs_dir.exists():
            shutil.rmtree(jobs_dir)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(project_root / "pyproject.toml", jobs_dir)
        shutil.copytree(project_root / "src", jobs_dir / "src")
        if (project_root / "config").exists():
            shutil.copytree(project_root / "config", jobs_dir / "config")
    
    print(f"\n✅ POC environment prepared!")
    print(f"   Workspace: {poc_workspace}")
    print(f"   Job: {job_output}")
    print(f"\nNext steps:")
    print(f"   1. Start POC:  python jobs/poc_runner.py start")
    print(f"   2. Submit job: python jobs/poc_runner.py submit")
    
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    """Start POC services."""
    print(f"\n{'='*60}")
    print("Starting NVFlare POC Services")
    print(f"{'='*60}\n")
    
    poc_workspace = get_poc_workspace()
    if not poc_workspace.exists():
        print("❌ POC not prepared. Run: python jobs/poc_runner.py prepare")
        return 1
    
    if args.component:
        # Start specific component
        print(f"Starting {args.component}...")
        run_command(["nvflare", "poc", "start", "-p", args.component])
    else:
        # Start all services
        print("Starting all POC services (server + clients)...")
        result = run_command(["nvflare", "poc", "start"], check=False)
        
        if result.returncode != 0:
            print("\n⚠️  Some services may have failed to start.")
            print("Try starting components individually:")
            print("  nvflare poc start -p server")
            print("  nvflare poc start -p site-1")
            return 1
    
    print("\n✅ POC services started!")
    print("\nTo submit a job:")
    print("   python jobs/poc_runner.py submit")
    print("\nTo connect to admin console:")
    print("   nvflare poc start -p admin")
    
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit job to POC."""
    print(f"\n{'='*60}")
    print("Submitting SFL Job to POC")
    print(f"{'='*60}\n")
    
    poc_workspace = get_poc_workspace()
    
    # Check for generated job
    job_dir = poc_workspace / "jobs" / "sfl-job" / "sfl-federated-sum"
    if not job_dir.exists():
        # Try legacy location
        job_dir = poc_workspace / "jobs" / "sfl"
    
    if not job_dir.exists():
        print("❌ Job not found. Run: python jobs/poc_runner.py prepare")
        return 1
    
    print(f"Job directory: {job_dir}")
    print("\n📝 Interactive Admin Console")
    print("-" * 40)
    print("Commands to use in admin console:")
    print(f"  > submit_job {job_dir}")
    print("  > list_jobs")
    print("  > check_status server")
    print("  > check_status client")
    print("  > abort_job <job_id>")
    print("  > bye")
    print("-" * 40)
    print("\nStarting admin console...\n")
    
    run_command(["nvflare", "poc", "start", "-p", "admin"], check=False)
    
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop POC services."""
    print(f"\n{'='*60}")
    print("Stopping NVFlare POC Services")
    print(f"{'='*60}\n")
    
    run_command(["nvflare", "poc", "stop"], check=False)
    
    print("\n✅ POC services stopped!")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Clean POC workspace."""
    print(f"\n{'='*60}")
    print("Cleaning NVFlare POC Workspace")
    print(f"{'='*60}\n")
    
    run_command(["nvflare", "poc", "clean"], check=False)
    
    print("\n✅ POC workspace cleaned!")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Check POC status."""
    print(f"\n{'='*60}")
    print("NVFlare POC Status")
    print(f"{'='*60}\n")
    
    poc_workspace = get_poc_workspace()
    
    if not poc_workspace.exists():
        print("❌ POC not prepared")
        print("   Run: python jobs/poc_runner.py prepare")
        return 1
    
    print(f"POC Workspace: {poc_workspace}")
    print("\nComponents:")
    
    for item in sorted(poc_workspace.iterdir()):
        if item.is_dir() and item.name not in ["jobs", "transfer"]:
            pid_file = item / "pid.fl"
            status = "🟢 Running" if pid_file.exists() else "⚪ Stopped"
            print(f"  {item.name}: {status}")
    
    jobs_dir = poc_workspace / "jobs"
    if jobs_dir.exists():
        print(f"\nJobs directory: {jobs_dir}")
        for job in jobs_dir.iterdir():
            print(f"  - {job.name}")
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NVFlare POC Mode Runner for SFL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full workflow
    python jobs/poc_runner.py prepare --num-clients 4
    python jobs/poc_runner.py start
    python jobs/poc_runner.py submit
    python jobs/poc_runner.py stop
    python jobs/poc_runner.py clean
    
    # Check status
    python jobs/poc_runner.py status
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare POC environment")
    prepare_parser.add_argument(
        "--num-clients", "-n",
        type=int,
        default=2,
        help="Number of clients (default: 2)",
    )
    prepare_parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean existing POC before preparing",
    )
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start POC services")
    start_parser.add_argument(
        "--component", "-p",
        type=str,
        help="Specific component to start (server, site-1, site-2, admin)",
    )
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit job to POC")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop POC services")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean POC workspace")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check POC status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "prepare": cmd_prepare,
        "start": cmd_start,
        "submit": cmd_submit,
        "stop": cmd_stop,
        "clean": cmd_clean,
        "status": cmd_status,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
