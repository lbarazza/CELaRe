This repository contains the code developed for the Independent Research Project as part of the Applied Computational Science and Engineering MSc program at Imperial College London (Academic Year 2023/2024).

<div align="center">
    <img src="https://github.com/user-attachments/assets/70442575-9d02-4b3a-bf2e-1fc509c888e4" alt="LunarLander-v2" />
    <p><em>CELaRe agent landing a rocket in the LunarLander-v2 environment and histogram of the proximity of the hidden layer to each cluster.</em></p>
</div>

# CELaRe
This project proposes Clustering-Enhanced Layer Regularisation, a novel method for improving the interpretability of reinforcement learning models while maintaining their performance. Our method regularizes hidden layers by inducing the formation and combination of clusters with VQ-VAEs and uses discretized oscillating coefficients to ensure performance is not compromised. For more information on the method and the performed experiments see `deliverables/lb1223-final-report.pdf`.

## Table of Contents

- [CELaRe](#celare)
- [Installation](#installation)
- [Structure of the Repository](#structure-of-the-repository)
- [Usage](#usage)
  - [Use the CELaRe package](#use-the-celare-package)
  - [Train a CELaRe Agent](#train-a-celare-agent)
  - [Run Set of Experiments and Replicate Work](#run-set-of-experiments-and-replicate-work)
  - [Visualize Results](#visualize-results)
    - [Visualizing Agent and Clusters](#visualizing-agent-and-clusters)
    - [Visualizing Hidden Layers](#visualizing-hidden-layers)
- [Deliverables](#deliverables)
- [License](#license)

## Installation

To install the repository code locally follow the steps below.

1.  Clone the repository and change the current working directory to it.
    ```bash
    $ git clone https://github.com/ese-msc-2023/irp-lb1223.git
    $ cd irp-lb1223
    ```

2.  Install swig (required for the Box2D dependency to work correctly) and ffmpeg. On macOS, this can be done with the following.
    ```bash
    $ brew install swig
    $ brew install ffmpeg
    ```

3.  Create a conda environment with the required dependencies provided in `environment.yml`.
    ```bash
    $ conda env create -f environment.yml -n celare
    $ conda activate celare
    ```
    Note that we don't provide a `requirements.txt` file as some dependencies can only be obtained through conda.
    > **⚠️ Warning:** It is well known that installing the Box2D dependency (required for the Box2D environments) alongside Gymnasium can cause issues. The installation process provided has been tested exclusively on an M1 MacBook. Therefore, if the operating system version or computer differs, a more tailored installation approach may be required.
 
4. Install the CELaRe package provided in `src/CELaRe/`.
    ```bash
    $ pip install -e src/CELaRe
    ```

The code should now be installed and ready to use.


## Structure of the Repository

The code for the project is contained in the `src/` folder. The structure of this folder is shown below


```
src/
├── CELaRe/
│   └── celare/
└── experiments/
    ├── config/
    ├── training/
    ├── visualizations/
    └── weights/

```

The `CELaRe/` folder contains the implementations of the CELaRe architecture and the agents used in the experiments. The `experiments/` folder contains all the code used to perform the experiments discussed in the report. It is divided into 4 different folders:

 - `config/` contains the files necessary to configure an experiment.
 - `training/` contains the files needed to initialize and run the experiments as configured in `config/`.
 - `visualizations/` contains the files used to perform the visualizations and analyses discussed in Section 5.1 of the report. This includes:
   - Visualizing the agent and the current cluster as it interacts with the environment (useful for labeling the clusters)
   - Visualizing the hidden layers
- `weights/` contains the weights resulting from the experiments. We include two example weight files for LunarLander-v2 and CarPole-v1.


## Usage

### Use the CELaRe package
The package offers four main components:

 - The `ClustLinear` class, which implements the CELaRe architecture. It can be used as any other PyTorch module

   **Example Use:**
   ```python
    from celare import ClustLinear

    class MyModel(nn.Module):
        def __init__(self, ...):
            super(MyModel, self).__init__()

            self.clust = ClustLinear(dim=64, codebook_size=3, latent_dim=2, detach=False, alpha=1.0)
            # Other layer or module initializations here

        def forward(self, x):
            # Some forward operations on x here
            
            # Use ClustLinear inside the forward method
            x, quant_loss, recons_loss, entropy_loss, indices, probs = self.clust(x)
            
            # Some other forward operations on x here
   ```
   
 - The `oscill_coef` function, responsible for the discretized oscillating coefficient as described in the report.

    **Example Use:**
    ```python
    from celare import oscill_coef
    coef = oscill_coef(update_step, freq=1/100, mean=0.1, amp=0.05, n=7)
    ```


 - The `AgentDiscrete` class, an actor-critic agent implementation designed for discrete action environments using the CELaRe architecture.
 - The `AgentContinuous` class, an actor-critic agent implementation tailored for continuous action environments, also utilizing the CELaRe architecture.

For more details on how to use `AgentDiscrete` and `AgentContinuous`, see `training/train_discrete.py` and `training/train_continuous.py`.

### Train a CELaRe Agent

To train a CELaRe agent on a given environment, use `train_discrete.py` or `train_continuous.py`. 

```bash
$ python experiments/training/train_<discrete|continuous>.py
```
> To ensure that the weights are saved in the correct folder it is recommended to run this and the following scripts from within the `src` folder.

Specific configurations for the training can be set by using additional arguments. For example:

 - `--env-id <environment-name>` sets the environment to use in training
 - `--agent-mode <0|1|2>` sets the mode of the agent to either the baseline (0), the constant coefficient version of CELaRe (1) or the oscillating coefficient version (2).

For a complete list of all possible configuration parameters see `config.py`.

We use [Weights & Biases](https://wandb.ai/site) for logging the results of training. All results can be seen at the URL printed on the console at the beginning of training. Checkpoints of the weights throughout the run are saved in `weights/`.

### Run Set of Experiments and Replicate Work

The `experiments/` folder contains all the necessary code to replicate the analyses and experiments discussed in the report as well as running other experiments.

The code uses Weights & Biases (wandb) sweeps to perform set of experiments and log the results. To configure a new sweep, add a corresponding `<experiment_name>.yaml` file in the `config/` folder, specifying the parameters of the various experiments. To help replicate the results in our report, we provide two files: `sweep_config_discrete.yaml` and `sweep_config_continuous.yaml` which contain the experiment parameters necessary to run our experiments. To replicate our work, simply use these as the configuration files. Once the experiment configuration has been determined, initialize the sweep with

```bash
$ python experiments/training/sweep_init.py --entity <your-wandb-entity> --project <your-wandb-project-name> --config-file experiments/config/<experiment_name>.yaml
```

This will produce a sweep ID. To then run the sweep of experiments, run the following

```bash
$ ./experiments/training/sweep_run.sh <mode> <num-workers> <your-wandb-entity> <your-wandb-project-name> <sweep-ID>
```

where `<mode>` takes one of two value: `--d` and `--c` for experiments on discrete action and continuous action environments, respectively. `<num-workers>` takes on an integer number and specifies the number of parallel workers to use to perform the experiments.

All the results for the experiments can then be found on Weight & Biases at the specified entity and project. Analogously to manually training the single agent, checkpoints of the weights are saved in `weights/` throughout training. 

> **More Detailed Sweep Configs:** to allow for environment-specific parameters when configuring a sweep, the `run_discrete_sweep` and `run_continuous_sweep` functions in `train_discrete.py` and `train_continuous.py` can be modified. This allows for more fine-grained sweep configurations.

### Visualize Results

#### Visualizing Agent and Clusters

To visualize the agent as it interacts with the environment together with the corresponding cluster at each timestep, use `visualizations/vis_agent.py`. It can be run with the following

```bash
$  python experiments/visualizations/vis_agent.py --<d|c> --weights-path <path-to-weights-file> --env-id <env-id>
```

Additional parameters can be specified as in [Train a CELaRe Agent](#train-a-celare-agent).

This will produce an animation showing the agent interacting with the environment and an histogram of the relative proximity of the current hidden layer state to each cluster. An example animation is at the beginning of this README.

#### Visualizing Hidden Layers

To visualize the hidden layer of the agent, it should first be recorded as the agent interacts with the environment. This can be done with

```bash
 $ python experiments/visualizations/record_hidden.py --<d|c> --weights-path <path-to-weights-file> --env-id <env-id> --collection-steps <n-collection-steps>
```

As it can be seen, an additional `--collection-steps` argument (in addition to the ones introduce before) can be specified to adjust the amount of steps to record the agent for. Running the above will save a `.csv` file containing the hidden layer values in the `visualizations/data/` folder.

Once the `.csv` has been saved, it can then be used by `vis_hidden.py` to perform PCA and visualize it in 2 dimensions. This can be done with

```bash
$ python experiments/visualizations/vis_hidden.py
```

which will generate a plot of the result. Below is an example plot obtained by training a CELaRe agent (right) and a baseline (left) on CartPole-v1.

<div align="center">
    <img src="https://github.com/user-attachments/assets/4b1da96f-980b-4428-98cb-fbfd15d043b9" alt="Hidden Cartpole 1M" />
    <p><em>Visualization of the hidden layer of an agent trained on CartPole-v1 with CELaRe (right) and the baseline (left)</em></p>
</div>

## Deliverables

`deliverables/` contains 3 files explaining the project in more depth

 - `lb1223-project-plan.pdf` contains the initial project plan.
 - `lb1223-final-report.pdf` contains the final report explaining the proposed method and the performed experiments.
 - `lb1223-presentation.pdf` contains the slides for a presentation explaining the project.

## License

This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for more details.
