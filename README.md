# HPC Submission Launcher for Hydra

--------------------------------------------------------------------------------
Project: HPC Submission Launcher for Hydra
Author: Carel van Niekerk  
Year: 2024  
Group: Dialogue Systems and Machine Learning Group  
Institution: Heinrich Heine University Düsseldorf
--------------------------------------------------------------------------------

## Installation

```bash
mkdir -p hydra_plugins
git clone https://gitlab.cs.uni-duesseldorf.de/dsml/HydraHPCLauncher.git hydra_plugins/hpc_submission_launcher
```

## Usage

Add to your Hydra Config:

```yaml
defaults:
    - hydra/launcher = hpc_submission
```

Or to your hydra zen wrapper:

```python
@store(
    name="main",
    hydra_defaults=["_self_", {"override hydra/launcher": "hpc_submission"}],
)
```
