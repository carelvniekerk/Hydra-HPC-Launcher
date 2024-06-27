import sys

from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from omegaconf import DictConfig


class PrintCommandLauncher(Launcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(
        self,
        config: DictConfig,
        task_function,
        hydra_context,
    ) -> None:
        self.task_function = task_function

    def launch(self, job_overrides: list, initial_job_idx) -> JobReturn:
        results = []
        for idx, job_override in enumerate(job_overrides):
            print_command = f"python {sys.argv[0]} {' '.join(job_override)}"
            print(print_command)
            results.append(
                JobReturn(overrides=job_override, status=JobStatus.COMPLETED)
            )
        return results


Plugins.instance().register(PrintCommandLauncher)
