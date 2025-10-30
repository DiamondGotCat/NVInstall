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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from nercone_modern.logging import ModernLogging  # type: ignore
    from nercone_modern.progressbar import ModernProgressBar  # type: ignore
except ImportError as exc:
    NerconeModern = None  # type: ignore # pylint: disable=invalid-name

VERSION = "v1.0.0"

@dataclass
class InstallerConfig:
    distro: str
    arch: str
    module: str  # "open" "proprietary"
    variant: str  # "full", "compute-only", "desktop-only"
    dry_run: bool

    def __post_init__(self) -> None:
        self.distro = self.distro.lower()
        self.arch = self.arch.lower()
        self.module = self.module.lower()
        self.variant = self.variant.lower()

def detect_os_release() -> Tuple[str, str]:
    distro_id = ''
    version_id = ''
    try:
        with open('/etc/os-release', 'r', encoding='utf-8') as f:
            for line in f:
                if '=' not in line:
                    continue
                key, value = line.strip().split('=', 1)
                value = value.strip('"')
                if key == 'ID':
                    distro_id = value
                elif key == 'VERSION_ID':
                    version_id = value
    except FileNotFoundError:
        distro_id = platform.system().lower()
    return distro_id, version_id

def detect_architecture() -> str:
    machine = platform.machine().lower()
    if machine in ('x86_64', 'amd64'):
        return 'x86_64'
    if machine in ('aarch64', 'arm64', 'armv8'):
        return 'sbsa'
    return machine

def run_command(cmd: str, logger: object, dry_run: bool, use_sudo: bool) -> None:
    full_cmd = cmd
    if use_sudo:
        full_cmd = f"sudo -E {cmd}"
    logger.log(f'$ {full_cmd}', level='INFO')
    if dry_run:
        return
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        logger.log(
            f'Command failed with exit code {result.returncode}', level='CRITICAL'
        )
        sys.exit(result.returncode)

def build_commands(config: InstallerConfig) -> List[str]:
    commands: List[str] = []
    distro = config.distro
    arch = config.arch
    module = config.module  # open, proprietary
    variant = config.variant  # full, compute-only, desktop-only
    if distro in ('ubuntu', 'debian'):
        commands.append('apt update')
        commands.append('apt install -y linux-headers-$(uname -r)')
        if distro == 'debian':
            commands.append('add-apt-repository contrib')
        repo_arch = 'x86_64' if arch == 'x86_64' else 'sbsa'
        keyring_url = (
            f'https://developer.download.nvidia.com/compute/cuda/repos/{distro}/'
            f'{repo_arch}/cuda-keyring_1.1-1_all.deb'
        )
        commands.append(
            f'wget -O /tmp/cuda-keyring.deb {keyring_url} || '
            f'curl -fsSL {keyring_url} -o /tmp/cuda-keyring.deb'
        )
        commands.append('dpkg -i /tmp/cuda-keyring.deb')
        commands.append('apt update')
        if module == 'open':
            if variant == 'compute-only':
                commands.append('apt install -y nvidia-driver-cuda nvidia-kernel-open-dkms')
            elif variant == 'desktop-only':
                commands.append('apt install -y nvidia-driver nvidia-kernel-open-dkms')
            else:  # full
                commands.append('apt install -y nvidia-open')
        else:  # proprietary
            if variant == 'compute-only':
                commands.append('apt install -y nvidia-driver-cuda nvidia-kernel-dkms')
            elif variant == 'desktop-only':
                commands.append('apt install -y nvidia-driver nvidia-kernel-dkms')
            else:
                commands.append('apt install -y cuda-drivers')
    elif distro in ('fedora'):
        commands.append('dnf install -y kernel-devel-matched kernel-headers')
        repo_arch = 'x86_64' if arch == 'x86_64' else 'sbsa'
        repo_url = (
            f'https://developer.download.nvidia.com/compute/cuda/repos/{distro}/'
            f'{repo_arch}/cuda-{distro}.repo'
        )
        commands.append(
            f'dnf config-manager addrepo --from-repofile={repo_url}'
        )
        commands.append('dnf clean expire-cache')
        if module == 'open':
            commands.append('dnf module enable -y nvidia-driver:open-dkms')
        else:
            commands.append('dnf module enable -y nvidia-driver:latest-dkms')
        if module == 'open':
            if variant == 'compute-only':
                commands.append('dnf install -y nvidia-driver-cuda kmod-nvidia-open-dkms')
            elif variant == 'desktop-only':
                commands.append('dnf install -y nvidia-driver kmod-nvidia-open-dkms')
            else:
                commands.append('dnf install -y nvidia-open')
        else:
            if variant == 'compute-only':
                commands.append('dnf install -y nvidia-driver-cuda kmod-nvidia-latest-dkms')
            elif variant == 'desktop-only':
                commands.append('dnf install -y nvidia-driver kmod-nvidia-latest-dkms')
            else:
                commands.append('dnf install -y cuda-drivers')
    elif distro in ('rhel', 'rocky', 'almalinux', 'oraclelinux', 'amazon', 'amazonlinux'):
        commands.append('dnf install -y kernel-devel-$(uname -r) kernel-headers')
        repo_arch = 'x86_64' if arch == 'x86_64' else 'sbsa'
        repo_url = (
            f'https://developer.download.nvidia.com/compute/cuda/repos/{distro}/'
            f'{repo_arch}/cuda-{distro}.repo'
        )
        commands.append(
            f'dnf config-manager --add-repo {repo_url}'
        )
        commands.append('dnf clean expire-cache')
        if module == 'open':
            commands.append('dnf module enable -y nvidia-driver:open-dkms')
        else:
            commands.append('dnf module enable -y nvidia-driver:latest-dkms')
        if module == 'open':
            if variant == 'compute-only':
                commands.append('dnf install -y nvidia-driver-cuda kmod-nvidia-open-dkms')
            elif variant == 'desktop-only':
                commands.append('dnf install -y nvidia-driver kmod-nvidia-open-dkms')
            else:
                commands.append('dnf install -y nvidia-open')
        else:
            if variant == 'compute-only':
                commands.append('dnf install -y nvidia-driver-cuda kmod-nvidia-latest-dkms')
            elif variant == 'desktop-only':
                commands.append('dnf install -y nvidia-driver kmod-nvidia-latest-dkms')
            else:
                commands.append('dnf install -y cuda-drivers')
    elif distro in ('suse', 'opensuse-leap', 'opensuse', 'sles'):
        commands.append('zypper --non-interactive install -y kernel-default-devel=$(uname -r | sed "s/-default//")')
        repo_arch = 'x86_64' if arch == 'x86_64' else 'sbsa'
        repo_url = (
            f'https://developer.download.nvidia.com/compute/cuda/repos/{distro}/'
            f'{repo_arch}/cuda-{distro}.repo'
        )
        commands.append(f'zypper addrepo {repo_url}')
        commands.append('zypper --non-interactive refresh')
        if module == 'open':
            if variant == 'compute-only':
                commands.append('zypper --non-interactive install -y nvidia-compute-G06 nvidia-open-driver-G06')
            elif variant == 'desktop-only':
                commands.append('zypper --non-interactive install -y nvidia-video-G06 nvidia-open-driver-G06')
            else:
                commands.append('zypper --non-interactive install -y nvidia-open')
        else:
            if variant == 'compute-only':
                commands.append('zypper --non-interactive install -y nvidia-compute-G06 nvidia-driver-G06')
            elif variant == 'desktop-only':
                commands.append('zypper --non-interactive install -y nvidia-video-G06 nvidia-driver-G06')
            else:
                commands.append('zypper --non-interactive install -y cuda-drivers')
    else:
        msg = (
            f'Distribution/OS "{distro}" is not supported by this installer. '
            'Supported distros include ubuntu, debian, fedora, rhel/rocky/amazon '
            'and suse/openSUSE.'
        )
        raise NotImplementedError(msg)
    return commands

def main() -> None:
    distro_id, distro_version = detect_os_release()
    default_arch = detect_architecture()

    parser = argparse.ArgumentParser(
        prog="nvinstall",
        description='Automatic NVIDIA driver installation tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--distro',
        default=distro_id,
        help='Linux distribution ID (e.g. ubuntu, debian, fedora, rhel, suse)'
    )
    parser.add_argument(
        '--arch',
        default=default_arch,
        help='CPU architecture (x86_64 or sbsa).  Detected from platform by default.'
    )
    parser.add_argument(
        '--module',
        choices=['open', 'proprietary'],
        default='open',
        help='Choose open or proprietary kernel modules.'
    )
    parser.add_argument(
        '--variant',
        choices=['full', 'compute-only', 'desktop-only'],
        default='full',
        help='Select installation variant: full installs all components; '
             'compute-only installs only compute libraries; desktop-only installs '
             'display components without compute parts.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands instead of executing them.'
    )
    args = parser.parse_args()

    config = InstallerConfig(
        distro=args.distro,
        arch=args.arch,
        module=args.module,
        variant=args.variant,
        dry_run=args.dry_run
    )

    logger = ModernLogging(process_name="NVInstall")
    logger.log(f"NVInstall {VERSION} by Nercone(DiamondGotCat)")
    logger.log("Automatic NVIDIA driver installation tool")
    logger.log("MIT License, Copyright (c) 2025 DiamondGotCat")
    logger.log("")

    logger.log("-- Installation information --")
    logger.log(f"Linux Distribution: {args.distro}")
    logger.log(f"CPU Architecture: {args.arch}")
    logger.log(f"NVIDIA Driver Module: {args.module}")
    logger.log(f"NVIDIA Driver Variant: {args.variant}")
    logger.log(f"Options: dry_run={'True' if args.dry_run else 'False'}")
    logger.log("")

    try:
        commands = build_commands(config)
    except NotImplementedError as exc:
        logger.log(str(exc), level='CRITICAL')
        sys.exit(1)
    progress_bar = ModernProgressBar(total=len(commands), process_name='Driver installation')
    progress_bar.start()
    use_sudo = os.name != 'nt' and os.geteuid() != 0
    for cmd in commands:
        run_command(cmd, logger, config.dry_run, use_sudo)
        progress_bar.update()
    progress_bar.finish()
    logger.log('NVIDIA driver installation completed successfully.', level='INFO')

if __name__ == '__main__':
    main()
