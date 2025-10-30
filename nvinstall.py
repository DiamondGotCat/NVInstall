#!/usr/bin/env python3

# -- NVInstall -------------------------------------------------- #
# nvinstall.py on NVInstall                                       #
# Made by DiamondGotCat, Licensed under MIT License               #
# Copyright (c) 2025 DiamondGotCat                                #
# ---------------------------------------------- DiamondGotCat -- #

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple

try:
    from nercone_modern.logging import ModernLogging  # type: ignore
    from nercone_modern.progressbar import ModernProgressBar  # type: ignore
except ImportError:
    class ModernLogging:  # type: ignore
        def __init__(self, process_name: str = "NVInstall") -> None:
            self.process_name = process_name

        def log(self, msg: str, level: str = "INFO") -> None:
            print(f"[{self.process_name}][{level}] {msg}")

    class ModernProgressBar:  # type: ignore
        def __init__(self, total: int, process_name: str = "Progress") -> None:
            self.total = max(total, 1)
            self.cur = 0
            self.process_name = process_name

        def start(self) -> None:
            self._render()

        def update(self) -> None:
            self.cur += 1
            self._render()

        def finish(self) -> None:
            self.cur = self.total
            self._render()
            print()

        def _render(self) -> None:
            pct = int(self.cur * 100 / self.total)
            bar = "#" * (pct // 5)
            print(f"\r[{self.process_name}] [{bar:<20}] {pct:3d}% ", end="", flush=True)

VERSION = "v1.0.0"

@dataclass
class InstallerConfig:
    distro: str
    arch: str
    module: str  # "open" or "proprietary"
    variant: str  # "full", "compute-only", "desktop-only"
    dry_run: bool

    def __post_init__(self) -> None:
        self.distro = self.distro.lower()
        self.arch = self.arch.lower()
        self.module = self.module.lower()
        self.variant = self.variant.lower()

def detect_os_release() -> Tuple[str, str]:
    distro_id = ""
    version_id = ""
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                value = value.strip('"')
                if key == "ID":
                    distro_id = value
                elif key == "VERSION_ID":
                    version_id = value
    except FileNotFoundError:
        distro_id = platform.system().lower()
    return distro_id, version_id

def detect_architecture() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("aarch64", "arm64", "armv8"):
        return "sbsa"
    return machine

def distro_key(distro_id: str, version_id: str) -> str:
    d = distro_id.lower()
    v = version_id.lower()
    if d == "ubuntu" and v.startswith("22.04"):
        return "ubuntu22.04"
    if d == "debian" and v.startswith("12"):
        return "debian12"
    if d == "fedora":
        return "fedora42"
    if d in ("opensuse", "opensuse-leap", "suse", "sles"):
        return "opensuse15"
    if d in ("rhel", "redhat"):
        if v.startswith("8"):
            return "rhel8"
        if v.startswith("9"):
            return "rhel9"
        if v.startswith("10"):
            return "rhel10"
    if d in ("rocky", "almalinux", "oraclelinux"):
        if v.startswith("8"):
            return "rhel8"
        if v.startswith("9"):
            return "rhel9"
        if v.startswith("10"):
            return "rhel10"
    if d in ("amzn", "amazon", "amazonlinux", "amazon linux") and ("2023" in v or v == "2023"):
        return "amazonlinux2023"
    if d in ("azurelinux", "azure", "azl", "azlinux") and (v.startswith("3") or v == "3"):
        return "azurelinux3"
    return f"{d}{v and '-'+v}"

def run_command(cmd: str, logger: object, dry_run: bool, use_sudo: bool) -> None:
    full_cmd = f"sudo -E {cmd}" if use_sudo else cmd
    logger.log(f"$ {full_cmd}", level="INFO")
    if dry_run:
        return
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        logger.log(f"Command failed with exit code {result.returncode}", level="CRITICAL")
        sys.exit(result.returncode)

def _warn_variant_if_needed(logger: ModernLogging, variant: str) -> None:
    if variant in ("compute-only", "desktop-only"):
        logger.log(
            "Notice: current distro recipes do not separate compute-only/desktop-only. "
            f'Variant "{variant}" will be treated as "full".',
            level="WARNING",
        )

def build_commands(config: InstallerConfig, logger: ModernLogging) -> List[str]:
    commands: List[str] = []
    key = distro_key(config.distro, detect_os_release()[1])
    module = config.module
    _warn_variant_if_needed(logger, config.variant)

    if key == "amazonlinux2023":
        commands += [
            "dnf upgrade -y",
            "dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo",
            "dnf clean all",
        ]
        if module == "open":
            commands.append("dnf -y module install nvidia-driver:open-dkms")
        else:
            commands.append("dnf -y module install nvidia-driver:latest-dkms")

    elif key == "azurelinux3":
        commands += [
            "tdnf upgrade -y",
            "curl https://developer.download.nvidia.com/compute/cuda/repos/azl3/x86_64/cuda-azl3.repo | tee /etc/yum.repos.d/cuda-azl3.repo",
            "tdnf -y install azurelinux-repos-extended",
            "tdnf clean all",
        ]
        commands.append("tdnf -y install nvidia-open")

    elif key == "debian12":
        commands += [
            "apt-get upgrade -y",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        ]
        if module == "open":
            commands.append("apt-get install -y nvidia-open")
        else:
            commands.append("apt-get install -y cuda-drivers")

    elif key == "fedora42":
        commands += [
            "dnf upgrade -y",
            "dnf config-manager addrepo --from-repofile https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo",
            "dnf clean all",
        ]
        if module == "open":
            commands.append("dnf -y install nvidia-open")
        else:
            commands.append("dnf -y install cuda-drivers")

    elif key == "opensuse15":
        commands += [
            "zypper update -y",
            "zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/cuda-opensuse15.repo",
            "zypper refresh",
        ]
        if module == "open":
            commands.append("zypper install -y nvidia-open")
        else:
            commands.append("zypper install -y cuda-drivers")

    elif key in ("rhel8", "rhel9"):
        ver = "8" if key == "rhel8" else "9"
        commands += [
            "dnf upgrade -y",
            f"dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel{ver}/x86_64/cuda-rhel{ver}.repo",
            "dnf clean all",
        ]
        if module == "open":
            commands.append("dnf -y module install nvidia-driver:open-dkms")
        else:
            commands.append("dnf -y module install nvidia-driver:latest-dkms")

    elif key == "rhel10":
        commands += [
            "dnf upgrade -y",
            "dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel10/x86_64/cuda-rhel10.repo",
        ]
        if module == "open":
            commands.append("dnf -y install nvidia-open")
        else:
            commands.append("dnf -y install cuda-drivers")

    elif key == "ubuntu22.04":
        commands += [
            "apt-get upgrade -y",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        ]
        if module == "open":
            commands.append("apt-get install -y nvidia-open")
        else:
            commands.append("apt-get install -y cuda-drivers")

    else:
        msg = (
            f'Distribution/OS "{config.distro}" (normalized "{key}") is not supported by this recipe. '
            "Supported targets: amazonlinux2023, azurelinux3, debian12, fedora42, "
            "opensuse15, rhel8, rhel9, rhel10, ubuntu22.04."
        )
        raise NotImplementedError(msg)

    return commands

def main() -> None:
    distro_id, distro_version = detect_os_release()
    default_arch = detect_architecture()

    parser = argparse.ArgumentParser(
        prog="nvinstall",
        description="Automatic NVIDIA driver installation tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--distro",
        default=distro_id,
        help="Linux distribution ID (e.g. ubuntu, debian, fedora, rhel, opensuse, amzn, azurelinux)",
    )
    parser.add_argument(
        "--arch",
        default=default_arch,
        help="CPU architecture (x86_64 or sbsa). Detected from platform by default.",
    )
    parser.add_argument(
        "--module",
        choices=["open", "proprietary"],
        default="open",
        help="Choose open or proprietary kernel modules.",
    )
    parser.add_argument(
        "--variant",
        choices=["full", "compute-only", "desktop-only"],
        default="full",
        help="Variant is currently treated as full on all supported distros.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing them.",
    )
    args = parser.parse_args()

    logger = ModernLogging(process_name="NVInstall")
    logger.log(f"NVInstall {VERSION} by Nercone(DiamondGotCat)")
    logger.log("Automatic NVIDIA driver installation tool")
    logger.log("MIT License, Copyright (c) 2025 DiamondGotCat")
    logger.log("")

    logger.log("-- Installation information --")
    logger.log(f"Linux Distribution: {args.distro} ({distro_version})")
    logger.log(f"CPU Architecture: {args.arch}")
    logger.log(f"NVIDIA Driver Module: {args.module}")
    logger.log(f"NVIDIA Driver Variant: {args.variant}")
    logger.log(f"Options: dry_run={'True' if args.dry_run else 'False'}")
    logger.log("")

    config = InstallerConfig(
        distro=args.distro,
        arch=args.arch,
        module=args.module,
        variant=args.variant,
        dry_run=args.dry_run,
    )

    try:
        commands = build_commands(config, logger)
    except NotImplementedError as exc:
        logger.log(str(exc), level="CRITICAL")
        sys.exit(1)

    progress_bar = ModernProgressBar(total=len(commands), process_name="Driver installation")
    progress_bar.start()
    use_sudo = os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0
    for cmd in commands:
        run_command(cmd, logger, config.dry_run, use_sudo)
        progress_bar.update()
    progress_bar.finish()
    logger.log("NVIDIA driver installation completed.", level="INFO")

if __name__ == "__main__":
    main()
