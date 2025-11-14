# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HPC Submission Launcher for Hydra
# Author: Carel van Niekerk
# Year: 2025
# Group: Dialogue Systems and Machine Learning Group
# Institution: Heinrich Heine University Düsseldorf
# --------------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."
"""HPC Submission Launcher for Hydra."""

from dataclasses import dataclass

from hydra_hpc_launcher.launcher.types import HPCQueue, PackageManager, Template

__all__ = ["HPCSubmissionConfig"]


@dataclass
class HPCSubmissionConfig:
    """Configuration for HPC submission launcher."""

    _target_: str = (
        "hydra_plugins.hpc_submission_launcher.launcher.HPCSubmissionLauncher"
    )
    queue: HPCQueue = HPCQueue.DSML
    ncpus: int = 2
    memory: int = 16
    ngpus: int = 1
    walltime: str = "48:00:00"
    template: Template = Template.A100_40GB
    package_manager: PackageManager = PackageManager.UV
