from hydra_zen import store, zen

from hydra_plugins.hpc_submission_launcher.launcher import (
    HPCSubmissionLauncher,  # noqa: F401
)


@store(
    name="main",
    hydra_defaults=["_self_", {"override hydra/launcher": "hpc_submission"}],
)
def train_model(learning_rate: float = 0.01, batch_size: int = 32, epochs: int = 10):
    print(
        f"Training model with learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}"
    )


if __name__ == "__main__":
    store.add_to_hydra_store()

    zen(train_model).hydra_main(
        config_name="main",
        version_base="1.3",
        config_path=None,
    )
