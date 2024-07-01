from hydra.conf import HydraConf
from hydra_zen import ZenStore, builds, zen

from hydra_plugins.hpc_submission_launcher.launcher import HPCSubmissionLauncher


def train_model(learning_rate: float = 0.01, batch_size: int = 32, epochs: int = 10):
    print(
        f"Training model with learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}"
    )
    # Your actual training code here


ModelConfig = builds(train_model, populate_full_signature=True)

store = ZenStore()
store(ModelConfig, name="main")


if __name__ == "__main__":
    hydra_conf = HydraConf(
        launcher={
            "_target_": "hydra_plugins.hpc_submission_launcher.launcher.HPCSubmissionLauncher",
        },
    )
    store(hydra_conf)
    store.add_to_hydra_store()

    zen(train_model).hydra_main(
        config_name="main",
        version_base="1.3",
        config_path=None,
    )
