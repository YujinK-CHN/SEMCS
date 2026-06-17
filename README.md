# Learning Communication Skills in Multi-task Multi-agent Deep Reinforcement Learning
This repository includes the implementation of MCS.


## Installation
### Create environment and install packages
``` Bash
conda create -n transfer python=3.9
conda activate transfer
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```

install torch_scatter for TGCNet
```Bash
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.8.0%2Bcu129.html
```

## Environments
### Alice and Bob
#### With different semantics in observations
Alice Bob
- Key pos: place in the upper row randomly before the start of the entire game
- Goal pos: place in the bottom row randomly before the start of the entire game
- Agents: place in the map randomly before the start of each episode.

IDs
- sample ids for keys and goals from a pool of ids with the maximum of 5+5 numbers

### SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

In the file:
~/anaconda3/envs/transfer/lib/python3.9/site-packages/pysc2/lib/sc_process.py
```python
  def _launch(self, run_config, args, **kwargs):
    """Launch the process and return the process object."""
    # del kwargs
    try:
      with sw("popen"):
        return subprocess.Popen(args, cwd=run_config.cwd, env=run_config.env, **kwargs)
    except OSError as e:
      logging.exception("Failed to launch")
      raise SC2LaunchError("Failed to launch: %s" % args) from e
```

### Football

See the [environment](https://github.com/google-research/football)

``` Bash
conda install -c conda-forge gcc=12.1.0
pip install setuptools==65.5.0 pip==21 wheel==0.38.0  # before install gym
pip install gym==0.20.0
```

## How to run
When your environment is ready, you could run shell scripts provided.

### Testing tasks
**Test on Main clusters**
```Bash
# run tasks sequentially
bash run_sbatch.sh --run-mode sequential no_wandb AliceBob 5000 test
bash run_sbatch.sh --run-mode sequential no_wandb StarCraft 5000 test
bash run_sbatch.sh --run-mode sequential no_wandb Football 5000 test

