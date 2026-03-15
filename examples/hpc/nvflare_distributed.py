#!/usr/bin/env python3
"""
NVFlare Distributed Deployment Helper.

Automates the steps needed to run SFL across HPC centers using NVFlare
as the orchestration layer:

  1. Provision — generate startup kits (TLS certs, identity, auth policies)
  2. Distribute — copy startup kits to each site
  3. Launch — start server, clients, and admin console

Prerequisites:
  - nvflare >= 2.5 (pip install nvflare)
  - SSH access to remote sites (for distribution)
  - project.yml customized with your hostnames

Usage:
    # Step 1: Provision startup kits
    python examples/hpc/nvflare_distributed.py provision

    # Step 2: Copy kits to sites (via SSH)
    python examples/hpc/nvflare_distributed.py distribute \
        --site-local localhost \
        --site-hpc login01.hpc.example.edu

    # Step 3: Launch (run on each machine, or use SLURM)
    python examples/hpc/nvflare_distributed.py launch --role server
    python examples/hpc/nvflare_distributed.py launch --role client --site site-hpc

    # Or: submit an SFL Flower job via NVFlare admin
    python examples/hpc/nvflare_distributed.py submit-job \
        --runner esm2 \
        --num-rounds 50 --dp --dp-noise 0.5
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PROJECT_YML = SCRIPT_DIR / "project.yml"
DEFAULT_WORKSPACE = Path.home() / ".nvflare" / "sfl-federation"


def provision(args: argparse.Namespace) -> int:
    """Generate NVFlare startup kits from project.yml."""
    project_yml = Path(args.project_yml)
    if not project_yml.exists():
        print(f"ERROR: {project_yml} not found.")
        print("Copy examples/hpc/project.yml and customize hostnames.")
        return 1

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nvflare.lighter.provision",
        "-p", str(project_yml),
        "-w", str(workspace),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("ERROR: Provisioning failed. Is nvflare installed?")
        print("  pip install nvflare>=2.5")
        return 1

    print(f"\nStartup kits generated in: {workspace}")
    print("\nContents:")
    for kit_dir in sorted(workspace.rglob("startup")):
        print(f"  {kit_dir.parent.name}/startup/")
    return 0


def distribute(args: argparse.Namespace) -> int:
    """Copy startup kits to remote sites via SSH/rsync."""
    workspace = Path(args.workspace)
    prod_dir = _find_prod_dir(workspace)
    if prod_dir is None:
        print(f"ERROR: No provisioned kits found in {workspace}")
        print("Run 'provision' first.")
        return 1

    site_map = {}
    if args.site_local:
        site_map["site-local"] = args.site_local
    if args.site_hpc:
        site_map["site-hpc"] = args.site_hpc

    if not site_map:
        print("ERROR: Specify at least one --site-* target.")
        return 1

    # Always distribute server kit too
    server_host = args.server_host or "localhost"

    errors = 0
    for site_name, host in site_map.items():
        kit_dir = prod_dir / site_name
        if not kit_dir.exists():
            print(f"WARNING: No kit for '{site_name}' in {prod_dir}")
            errors += 1
            continue

        dest = args.remote_path or f"/tmp/nvflare-kits/{site_name}"
        if host in ("localhost", "127.0.0.1"):
            # Local copy
            dest_path = Path(dest)
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(kit_dir, dest_path / "startup", dirs_exist_ok=True)
            print(f"  {site_name} -> {dest_path} (local)")
        else:
            # rsync over SSH
            cmd = [
                "rsync", "-avz", "--mkpath",
                str(kit_dir) + "/",
                f"{host}:{dest}/",
            ]
            print(f"  {site_name} -> {host}:{dest}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  ERROR: rsync to {host} failed")
                errors += 1

    # Also distribute server kit
    server_kit = prod_dir / "server"
    if server_kit.exists():
        dest = args.remote_path or "/tmp/nvflare-kits/server"
        if server_host in ("localhost", "127.0.0.1"):
            dest_path = Path(dest)
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(server_kit, dest_path / "startup", dirs_exist_ok=True)
            print(f"  server -> {dest_path} (local)")
        else:
            cmd = [
                "rsync", "-avz", "--mkpath",
                str(server_kit) + "/",
                f"{server_host}:{dest}/",
            ]
            print(f"  server -> {server_host}:{dest}")
            subprocess.run(cmd)

    return 1 if errors else 0


def launch(args: argparse.Namespace) -> int:
    """Launch an NVFlare participant (server, client, or admin)."""
    workspace = Path(args.workspace)
    prod_dir = _find_prod_dir(workspace)
    if prod_dir is None:
        print(f"ERROR: No provisioned kits found in {workspace}")
        return 1

    role = args.role
    site = args.site

    if role == "server":
        kit_dir = prod_dir / "server"
    elif role == "client":
        if not site:
            print("ERROR: --site required for client role (e.g., --site site-hpc)")
            return 1
        kit_dir = prod_dir / site
    elif role == "admin":
        kit_dir = prod_dir / "admin@sfl"
    else:
        print(f"ERROR: Unknown role '{role}'. Use: server, client, admin")
        return 1

    startup_dir = kit_dir / "startup"
    if not startup_dir.exists():
        print(f"ERROR: Startup kit not found at {startup_dir}")
        return 1

    # Find the start script
    if role == "admin":
        start_script = startup_dir / "fl_admin.sh"
    else:
        start_script = startup_dir / f"start.sh"

    if not start_script.exists():
        # NVFlare 2.5+ may use different names
        candidates = list(startup_dir.glob("*.sh"))
        if candidates:
            start_script = candidates[0]
        else:
            print(f"ERROR: No start script found in {startup_dir}")
            return 1

    print(f"Launching NVFlare {role}" + (f" ({site})" if site else ""))
    print(f"  Kit: {startup_dir}")
    print(f"  Script: {start_script}")

    os.chmod(start_script, 0o755)
    result = subprocess.run(["bash", str(start_script)], cwd=str(startup_dir))
    return result.returncode


def submit_job(args: argparse.Namespace) -> int:
    """Package and submit an SFL Flower job via NVFlare admin API."""
    workspace = Path(args.workspace)
    prod_dir = _find_prod_dir(workspace)
    if prod_dir is None:
        print(f"ERROR: No provisioned kits found in {workspace}")
        return 1

    runner = args.runner  # esm2 or llm
    runner_script = PROJECT_ROOT / "jobs" / f"{runner}_runner.py"
    if not runner_script.exists():
        print(f"ERROR: Runner not found: {runner_script}")
        return 1

    # Build the NVFlare job config
    job_dir = Path(args.job_dir or f"/tmp/nvflare-jobs/sfl-{runner}")
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create meta.json
    meta = {
        "name": f"sfl-{runner}-federated",
        "resource_spec": {},
        "min_clients": args.num_clients,
        "deploy_map": {
            "app": ["@ALL"],
        },
    }

    import json
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Create app directory with config
    app_dir = job_dir / "app" / "config"
    app_dir.mkdir(parents=True, exist_ok=True)

    # Build runner command args
    runner_args = [
        f"--num-rounds {args.num_rounds}",
        f"--num-clients {args.num_clients}",
    ]
    if args.dp:
        runner_args.append("--dp")
        if args.dp_noise:
            runner_args.append(f"--dp-noise {args.dp_noise}")
    if args.secagg:
        runner_args.append("--secagg")

    # config_fed_server.json — FlowerRecipe wrapping the SFL app
    server_config = {
        "format_version": 2,
        "workflows": [
            {
                "id": "fl_workflow",
                "path": "nvflare.app_opt.flower.flower_job.FlowerJob",
                "args": {
                    "flower_content": ".",
                    "min_clients": args.num_clients,
                    "num_rounds": args.num_rounds,
                },
            }
        ],
    }
    (app_dir / "config_fed_server.json").write_text(json.dumps(server_config, indent=2))

    # Copy source into the job
    src_dest = job_dir / "app" / "custom"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    shutil.copytree(PROJECT_ROOT / "src", src_dest / "src")
    shutil.copy2(runner_script, src_dest / f"{runner}_runner.py")

    print(f"Job packaged at: {job_dir}")
    print(f"\nTo submit via NVFlare admin console:")
    print(f"  > submit_job {job_dir}")
    print(f"\nOr use the NVFlare CLI:")
    print(f"  nvflare job submit -j {job_dir}")

    return 0


def _find_prod_dir(workspace: Path):
    """Find the production directory inside the NVFlare workspace."""
    # NVFlare provision creates: workspace/<project-name>/prod_XX/
    for d in sorted(workspace.rglob("prod_*"), reverse=True):
        if d.is_dir():
            return d
    # Also check direct workspace structure
    if (workspace / "server").exists():
        return workspace
    return None


def main():
    parser = argparse.ArgumentParser(
        description="NVFlare distributed deployment for SFL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace", default=str(DEFAULT_WORKSPACE),
        help=f"NVFlare workspace directory (default: {DEFAULT_WORKSPACE})",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # provision
    p_prov = sub.add_parser("provision", help="Generate startup kits from project.yml")
    p_prov.add_argument(
        "--project-yml", default=str(DEFAULT_PROJECT_YML),
        help="Path to project.yml (default: examples/hpc/project.yml)",
    )

    # distribute
    p_dist = sub.add_parser("distribute", help="Copy startup kits to sites via SSH")
    p_dist.add_argument("--site-local", help="Hostname for site-local")
    p_dist.add_argument("--site-hpc", help="Hostname for site-hpc")
    p_dist.add_argument("--server-host", default="localhost", help="Server hostname")
    p_dist.add_argument("--remote-path", help="Remote path for kits")

    # launch
    p_launch = sub.add_parser("launch", help="Start an NVFlare participant")
    p_launch.add_argument("--role", required=True, choices=["server", "client", "admin"])
    p_launch.add_argument("--site", help="Site name (required for client role)")

    # submit-job
    p_job = sub.add_parser("submit-job", help="Package and submit an SFL job")
    p_job.add_argument("--runner", required=True, choices=["esm2", "llm"])
    p_job.add_argument("--num-rounds", type=int, default=10)
    p_job.add_argument("--num-clients", type=int, default=2)
    p_job.add_argument("--dp", action="store_true")
    p_job.add_argument("--dp-noise", type=float)
    p_job.add_argument("--secagg", action="store_true")
    p_job.add_argument("--job-dir", help="Directory to package the job")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "provision": provision,
        "distribute": distribute,
        "launch": launch,
        "submit-job": submit_job,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
