# Unitree Isaac Lab 强化学习环境

## 概览

本项目将Unitree强化学习环境移植至[Isaac Lab](https://github.com/isaac-sim/IsaacLab)。已在最新版本中验证。

## 安装配置

- 系统要求
    - 操作系统：Ubuntu 22.04
	- 显卡：NVIDIA RTX显卡（如RTX5880，L20）
	- 内存：32 GB RAM，16 GB VRAM
	- 驱动版本：535或更高版本
	
	其他要求参考[Isaac Sim 系统配置](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements)
	
- 安装 Isaac Sim 和 Isaac Lab 
    - 推荐使用MiniConda安装，它可以简化从终端调用python脚本。
        ```bash
        conda create -n env_isaaclab python=3.10
        conda activate env_isaaclab
        # notice the CUDA version available on your system
        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 
        pip install --upgrade pip
        pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
        ```
    - 安装过程大约10分钟。安装结束后，做个验证：
        ```bash
        # note: you can pass the argument "--help" to see all arguments possible.
        isaacsim
        ```

    - 如果想通过binary安装，或者通过容器部署，可参考 [Isaac Lab 安装手册](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)。

- 安装本项目，Unitree IsaacLab强化学习环境
    - 从github下载本项目
        ```bash
        git clone https://github.com/unitreerobotics/unitree_rl_lab.git
        cd unitree_rl_lab
        ```

    - 安装本项目
        ```bash
        conda activate env_isaaclab
        python -m pip install -e source/unitree_rl_lab
        ```
- 下载Unitree USD 资产文件
    - 注意：这一步骤是临时方案。将来这些资产文件可以添加至Isaac Sim资产库。

    - 下载USD资产, 保持相同的目录结构: https://nvidia-my.sharepoint.com/:f:/r/personal/sharronl_nvidia_com/Documents/workspace/unitree_usd?csf=1&web=1&e=QqC2g2

    - 配置资产路径 'UNITREE_ASSET_ROOT_DIR'，在文件'source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py'中
        ```bash
        UNITREE_ASSET_ROOT_DIR = "</home/user/projects/unitree_usd>"
        ```

- 运行本项目

    - 列出已注册的强化学习环境

        ```bash
        python scripts/list_envs.py
        ```

    - 运行强化学习训练。可以通过--task参数指定机器人

        ```bash
        python scripts/rsl_rl/train.py --task unitree_g1_23dof_rev_1_0 --num_envs 4096 --headless --max_iterations <10000>
        ```

    - 运行强化学习推理，验证已训练的模型

        ```bash
        python scripts/rsl_rl/play.py --task unitree_g1_23dof_rev_1_0_play
        ```

- Sim2Sim Mujoco 部署

    - 下载示例所需文件 到depoly/example_g1 (见下 文件结构说明 部分)

    - mujoco 运行示例

        ```bash
        python depoly/sim2sim.py --config_file g1_lab.yaml
        ```
    - 文件结构说明
        ```bash
        - deploy/sim2sim.py # mjc inference 主程序
        - deploy/sim2sim_helper.py  # 实现读取isaaclab参数配置, 不同仿真环境下joint映射等
        - deploy/g1_lab.yaml # sim2sim.py 的输入config文件
        
        - deploy/example_g1
                ./2025-06-06_06-06-06 # IsaacLab训练的示例目录，需要其中的policy.pt 和 env.yaml
                ./g1_desc #机器人描述文件， 由IsaacLab中的 G1_MINIMAL_CFG 对应的usd文件生成

        ```
        ![Sim2Sim Demo](doc/sim2sim.gif)