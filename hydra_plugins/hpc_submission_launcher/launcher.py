import subprocess
import sys

from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from omegaconf import DictConfig


class HPCSubmissionLauncher(Launcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(
        self,
        config: DictConfig,
        task_function,
        hydra_context,
    ) -> None:
        self.task_function = task_function

    def launch(self, job_overrides: list, initial_job_idx) -> list[JobReturn]:
        results = []
        for idx, job_override in enumerate(job_overrides):
            job_name: str = (
                f"{self.task_function.func.__name__}_{initial_job_idx + idx}"
            )
            job_script_args: str = " ".join(job_override)
            launch_command = [
                "hpc",
                "run",
                f"--job_name {job_name}",
                f"--job_script {sys.argv[0]}",
                f'--job_script_args "{job_script_args}"',
            ]
            subprocess.run(" ".join(launch_command), shell=True)
            results.append(
                JobReturn(overrides=job_override, status=JobStatus.COMPLETED)
            )
        return results


Plugins.instance().register(HPCSubmissionLauncher)
