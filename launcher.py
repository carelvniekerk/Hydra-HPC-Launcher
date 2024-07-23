# coding=utf-8
# --------------------------------------------------------------------------------
# Project: HPC Submission Launcher for Hydra
# Author: Carel van Niekerk
# Year: 2024
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

import re
import subprocess
import sys
from enum import StrEnum
from logging import getLogger
from pathlib import Path

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    _save_config,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from hydra_zen import builds
from omegaconf import DictConfig, OmegaConf, open_dict


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


class Queue(StrEnum):
    """Queue options for HPC submission."""

    DSML = "DSML"
    CUDA = "CUDA"


class Template(StrEnum):
    """Template options for HPC submission."""

    DSML = "DSML"
    GTX1080 = "GTX1080"
    RTX6000 = "RTX6000"
    RTX8000 = "RTX8000"
    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"


logger = getLogger("__main__")


class HPCSubmissionLauncher(Launcher):
    """HPC Submission Launcher for Hydra."""

    def __init__(  # noqa: PLR0913, D107
        self,
        queue: Queue = Queue.CUDA,
        ncpus: int = 2,
        memory: int = 32,
        ngpus: int = 1,
        walltime: str = "16:00:00",
        template: Template = Template.RTX6000,
        **kwargs: dict,
    ) -> None:
        super().__init__(**kwargs)
        self.queue = queue
        self.ncpus = ncpus
        self.memory = memory
        self.ngpus = ngpus
        self.walltime = walltime
        self.template = template

    def setup(
        self,
        config: DictConfig,
        task_function: TaskFunction,
        hydra_context: HydraContext,
    ) -> None:
        """Set up the HPC Submission Launcher."""
        self.task_function = task_function
        self.config = config
        self.hydra_context = hydra_context

    def launch(
        self,
        job_overrides: list,
        initial_job_idx: int = 0,
    ) -> list[JobReturn]:
        """Launch the jobs with the given overrides."""
        results = []
        for idx, job_override in enumerate(job_overrides):
            # Job name is the task function name and the job index
            job_name: str = (
                f"{self.task_function.func.__name__}_{initial_job_idx + idx}"
            )
            # If the job script is in the bin directory (ie. a poetry script) then use
            # the prun option in the submission command
            job_script: Path = Path(sys.argv[0])
            if job_script.suffix == "" and job_script.resolve().parent.name == "bin":
                job_script = Path(f"prun:{job_script.name}")
            job_script_args: str = " ".join(job_override)
            launch_command = [
                "hpc",
                "run",
                f"--job_name {job_name}",
                f"--job_script {job_script}",
                f'--job_script_args "{job_script_args}"',
                f"--template {self.template}",
                f"--queue {self.queue}",
                f"--ncpus {self.ncpus}",
                f"--memory {self.memory}",
                f"--ngpus {self.ngpus}",
                f"--walltime {self.walltime}",
            ]

            # Submit job to HPC
            launch_command = " ".join(launch_command)
            log_msg = f"Submitting job with command: {launch_command}"
            logger.info(log_msg)
            output = subprocess.run(
                launch_command,
                shell=True,
                capture_output=True,
                check=True,
            )

            # Get the sweeper configuration
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config,
                list(job_override),
            )
            with open_dict(sweep_config):
                # Assign HPC job id to the sweep configuration
                match = re.search(r"with ID (\d+)\.", output.stdout.decode())
                job_id = match.group(1) if match else "Failed"
                sweep_config.hydra.job.id = job_id
                sweep_config.hydra.job.num = idx
            HydraConfig.instance().set_config(sweep_config)

            handle_output_dir_and_save_configs(sweep_config)

            results.append(
                JobReturn(
                    overrides=job_override,
                    status=JobStatus.COMPLETED,
                    cfg=sweep_config,
                ),
            )

        return results


# Register the HPC Submission Launcher
HPCSubmissionLauncherConfig = builds(
    HPCSubmissionLauncher,
    populate_full_signature=True,
)
ConfigStore.instance().store(
    group="hydra/launcher",
    name="hpc_submission",
    node=HPCSubmissionLauncherConfig,
)
Plugins.instance().register(HPCSubmissionLauncher)
