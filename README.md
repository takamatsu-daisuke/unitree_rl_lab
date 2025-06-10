# Unitree RL environments migration to Isaac Lab

## Overview

This folder migrates Unitree RL environments to [Isaac Lab](https://github.com/isaac-sim/IsaacLab), which is verified with the latest release.

## Installation

- Install Isaac Sim and Isaac Lab
    - We recommend using the conda installation as it simplifies calling Python scripts from the terminal.
        ```bash
        conda create -n env_isaaclab python=3.10
        conda activate env_isaaclab
        # notice the CUDA version available on your system
        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
        pip install --upgrade pip
        pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
        ```
    - The installation process takes about 10 minutes. Once done, please validate the installation by:
        ```bash
        # note: you can pass the argument "--help" to see all arguments possible.
        isaacsim
        ```

    - by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), you can also refer to the binary installation and docker deployment.

- Install the Unitree RL IsaacLab standalone environments.
    - Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
        ```bash
        git clone https://github.com/unitreerobotics/unitree_rl_lab.git
        cd unitree_rl_lab
        ```

    - Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:
        ```bash
        conda activate env_isaaclab
        python -m pip install -e source/unitree_rl_lab
        ```
- Download unitree usd files
    - NOTE: this's a temporary solution, till USD files are added to Isaac Sim assets.

    - Download unitree usd files from [here](https://nvidia-my.sharepoint.com/:f:/r/personal/sharronl_nvidia_com/Documents/workspace/unitree_usd?csf=1&web=1&e=QqC2g2), keeping folder structure

    - Config 'UNITREE_ASSET_ROOT_DIR' in 'source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py'.
        ```bash
        UNITREE_ASSET_ROOT_DIR = "</home/user/projects/unitree_usd>"
        ```

- Verify that the environments are correctly installed by:

    - Listing the available tasks:

        ```bash
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        python scripts/rsl_rl/train.py --task unitree_g1_23dof_rev_1_0 --num_envs 4096 --headless --max_iterations <10000>
        ```

    - Inference with a trained agent:

        ```bash
        python scripts/rsl_rl/play.py --task unitree_g1_23dof_rev_1_0_play
        ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/unitree_rl_lab/unitree_rl_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `IsaacLabExtensionTemplate/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Docker setup

### Building Isaac Lab Base Image

Currently, we don't have the Docker for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html).

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building Isaac Lab Template Image

Following above, you can build the docker container for this project. It is called `isaac-lab-template`. However,
you can modify this name inside the [`docker/docker-compose.yaml`](docker/docker-compose.yaml).

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build isaac-lab-template
```

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# isaac-lab-template               latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### Running the container

After building, the usual next step is to start the containers associated with your services. You can do this with:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up
```

This will start the services defined in your `docker-compose.yaml` file, including isaac-lab-template.

If you want to run it in detached mode (in the background), use:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} isaac-lab-template /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/unitree_rl_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
