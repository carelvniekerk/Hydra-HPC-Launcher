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
"""HPC Submission Launcher configuration handler."""

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import _save_config
from omegaconf import DictConfig, OmegaConf, open_dict

__all__ = ["handle_output_dir_and_save_configs"]


def handle_output_dir_and_save_configs(
    config: DictConfig,
    job_dir_key: str = "hydra.sweep.dir",
    job_subdir_key: str = "hydra.sweep.subdir",
) -> None:
    """Handle output directories and save configs.

    Args:
    ----
        config (DictConfig): The hydra sweeper config
        job_dir_key (str): The key to the output directory
        job_subdir_key (str): The key to the output subdirectory

    """
    orig_hydra_cfg = HydraConfig.instance().cfg

    # init Hydra config for config evaluation
    HydraConfig.instance().set_config(config)

    output_dir: Path = Path(OmegaConf.select(config, job_dir_key))
    if job_subdir_key is not None:
        subdir = Path(OmegaConf.select(config, job_subdir_key))
        output_dir = output_dir / subdir

    # Temporarily allow modification of the read-only config
    with open_dict(config):
        OmegaConf.set_readonly(config.hydra.runtime, value=False)
        config.hydra.runtime.output_dir = output_dir.resolve()
        OmegaConf.set_readonly(config.hydra.runtime, value=True)

    # update Hydra config
    HydraConfig.instance().set_config(config)

    try:
        # handle output directories here
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)

        if config.hydra.output_subdir is not None:
            hydra_output = Path(config.hydra.runtime.output_dir) / Path(
                config.hydra.output_subdir,
            )
            _save_config(config, "config.yaml", hydra_output)
            _save_config(HydraConfig.instance().cfg, "hydra.yaml", hydra_output)  # type: ignore  # noqa: PGH003
            _save_config(config.hydra.overrides.task, "overrides.yaml", hydra_output)
    finally:
        HydraConfig.instance().cfg = orig_hydra_cfg
