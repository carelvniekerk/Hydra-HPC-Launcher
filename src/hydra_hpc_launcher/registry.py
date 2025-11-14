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
"""Hydra plugin registration."""

from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher

from hydra_hpc_launcher.config import HPCSubmissionConfig
from hydra_hpc_launcher.launcher import HPCSubmissionLauncher

__all__ = ["register_plugin"]


def register_plugin(
    plugin_name: str = "hpc_submission",
    config: type[HPCSubmissionConfig] = HPCSubmissionConfig,
    launcher: type[Launcher] = HPCSubmissionLauncher,
) -> None:
    """Register the HPC Submission Launcher."""
    ConfigStore.instance().store(
        group="hydra/launcher",
        name=plugin_name,
        node=config,
    )
    Plugins.instance().register(launcher)
