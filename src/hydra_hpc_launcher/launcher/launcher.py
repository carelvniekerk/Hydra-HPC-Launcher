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

import re
import subprocess
import sys
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict

from hydra_hpc_launcher.config.handler import handle_output_dir_and_save_configs
from hydra_hpc_launcher.launcher.types import HPCQueue, PackageManager, Template

__all__ = ["HPCSubmissionLauncher"]

logger = getLogger("HydraHPCLauncher")


class HPCSubmissionLauncher(Launcher):
    """HPC Submission Launcher for Hydra."""

    def __init__(  # noqa: PLR0913, D107
        self,
        queue: HPCQueue = HPCQueue.DSML,
        ncpus: int = 2,
        memory: int = 16,
        ngpus: int = 1,
        walltime: str = "-1:00:00",
        template: Template = Template.RTX6000,
        package_manager: PackageManager = PackageManager.UV,
        **kwargs: dict,
    ) -> None:
        super().__init__(**kwargs)
        self.queue: HPCQueue = queue
        self.ncpus = ncpus
        self.memory = memory
        self.ngpus = ngpus
        self.walltime = walltime
        self.template = template
        self.package_manager = package_manager

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

    def launch(  # noqa: C901, PLR0912, PLR0915
        self,
        job_overrides: Sequence[Sequence[str]],
        initial_job_idx: int = 0,
    ) -> Sequence[JobReturn]:
        """Launch the jobs with the given overrides."""
        results: list[JobReturn] = []
        for idx, job_override in enumerate(job_overrides):
            # Job name is the task function name and the job index
            try:
                job_name: str = (
                    f"{self.task_function.func.__name__}_{initial_job_idx + idx}"  # type: ignore[attr-defined]
                )
            except AttributeError:
                job_name = f"{self.task_function.__name__}_{initial_job_idx + idx}"
            # If the job script is in the bin directory (ie. a poetry script) then use
            # the prun option in the submission command
            job_script: Path = Path(sys.argv[0])
            if any("+launch.script=" in arg for arg in job_override):
                job_script = Path(
                    next(arg for arg in job_override if "+launch.script=" in arg).split(
                        "=",
                        1,
                    )[1],
                )
                job_override.pop(  # type: ignore[unresolved-attribute]
                    job_override.index(
                        next(arg for arg in job_override if "+launch.script=" in arg),
                    ),
                )
            if job_script.suffix == "" and job_script.resolve().parent.name == "bin":
                job_script = Path(f"{self.package_manager.value}:{job_script.name}")

            # Reformat string overrides to handle spaces in the values
            overrides_list: list[str] = []
            launch_command_overrides: list[str] = []
            for override in job_override:
                if (
                    "\\" not in override
                    and "$" not in override
                    and "+launch" not in override
                ):
                    overrides_list.append(override)
                    continue

                override_key, override_value = override.split("=", 1)
                override_value = override_value.replace("\\", "")

                if "(" in override_value:
                    override_value = override_value.replace("(", "\\(")
                if ")" in override_value:
                    override_value = override_value.replace(")", "\\)")
                if "{" in override_value:
                    override_value = override_value.replace("{", "\\{")
                if "}" in override_value:
                    override_value = override_value.replace("}", "\\}")
                if "$" in override_value:
                    override_value = override_value.replace("$", "\\\\\\\\$")

                # Handle launch command overrides
                if "+launch" in override_key:
                    override_key = override_key.replace("+launch.", "").replace(
                        "+launch/",
                        "",
                    )
                    launch_command_overrides.append(
                        f'{override_key}=\\"{override_value}\\"',
                    )
                    continue
                overrides_list.append(f'{override_key}=\\"{override_value}\\"')

            if launch_command_overrides:
                overrides_list.append("--")
                overrides_list.extend(launch_command_overrides)

            job_script_args: str = " ".join(overrides_list)
            launch_command_list: list[str] = [
                "hpc",
                "run",
                f"--job_name {job_name}",
                f"--job_script {job_script}",
                f'--job_script_args "{job_script_args}"',
                f"--template {self.template}",
                f"--ncpus {self.ncpus}",
                f"--memory {self.memory}",
                f"--ngpus {self.ngpus}",
                f"--walltime={self.walltime}",
            ]
            if self.queue != HPCQueue.DEFAULT:
                launch_command_list.append(
                    f"--queue {self.queue}",
                )

            # Submit job to HPC
            launch_command: str = " ".join(launch_command_list)
            log_msg: str = f"Submitting job with command: {launch_command}"
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
