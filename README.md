# Remote GPU Guide

Guide on how to use MLDA GPU servers for Machine Learning

Click the icon at the top left corner to view Table of Contents

Last update: 1 Sep 2021

## Connect to the GPU server

### NTU VPN

You can only connect to the GPU server if you are connected to NTU network. If you are using Hall internet or home internet, set up NTU VPN to connect to NTU network.

Install and setup NTU VPN: https://ntuvpn.ntu.edu.sg/

### SSH

**OpenSSH** You will use SSH (Secure Shell Protocol) to connect to the GPU server. For MacOS, Linux and Windows (build 1809 or later), OpenSSH is built-in.

Open your Terminal (MacOS and Linux) or Command Prompt (Windows) and use the command: `ssh <username>@<domain.name>`

```bash
ssh acct2009_demo@dl2080-09.dynip.ntu.edu.sg
```

For first-time ssh, type "yes" when you are asked to accept the SSH key fingerprint. Type in your password and press Enter to finish connecting to the server. Now you should see info about the GPU server and your username in the terminal.

The username, domain name, and password should be provided to you when you sign up for a GPU account.

**(Optional) Using SSH key** Using an SSH key is more secure than using a password. You also don't need to type in a password when you use an SSH key. Follow the tutorial [here](https://www.ssh.com/academy/ssh/keygen)

**PuTTy** For earlier versions of Windows, please use PuTTy. Download PuTTY here: https://www.chiark.greenend.org.uk/~sgtatham/putty/

## Basic navigation

Command | Description
--------|------------
`cd` | Change directory (folder)
`mkdir` | Make directory (folder)
`pwd` | Print working (current) directory
`mv` | Move (files or folders). Can use this command to rename also
`rm` | Remove files. Add `-rf` to remove folders
`cp` | Copy

Linux Command Cheat Sheet: [link](https://www.guru99.com/linux-commands-cheat-sheet.html)

## Python for Data Science and Machine Learning

The recommended way to install and use Python is using Anaconda. You don't need admin rights to install and use Python with Anaconda. We will use Miniconda and install packages as we need.

Official Installation Guide: [link](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Note:
- Make sure to initialize conda (`conda init`) when it is prompted. This will add `conda` to path so that you can use conda the next time you log in to your account
- Exit and log in to your account again. You should now see `(base)` in front your username in the terminal

### Set up Python environment

It is recommended to create a separate virtual environment for each project ([What is a virtual environment?](https://realpython.com/python-virtual-environments-a-primer/))

To create a new environment (change `3.8` if you need a different Python version)

```bash
conda create -n <name of your environment> python=3.8
```

To activate the environment

```bash
conda activate <name of your environment>
```

Note: Name of the active environment should be reflected before your username in the terminal

To install packages / libraries to your environment

```bash
conda install <package name>
pip install <package name>
```

Note:

- You can use either `conda` or `pip` to install packages. Generally `conda` resolves dependency issues better. However there are many packages only available on `pip`
- For libraries that depend on GPU (using CUDA), you must install it with `conda`. This is because `conda` will install the correct CUDA/cuDNN version for your libraries. See how to install TensorFlow and PyTorch below.

To list installed packages

```bash
conda list
pip freeze
```

To deactivate the environment (when you finish working on a project)

```bash
conda deactivate
```

To see what environments are available

```bash
conda env list
```

### Install Deep Learning libraries

#### TensorFlow

Install TensorFlow with `conda` (DO NOT follow the official installation instruction)

```bash
conda install tensorflow-gpu
```

Verify TensorFlow is working and is using GPU. Run this in Python

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

Note: as of 1 Sep 2021, the newest TensorFlow version on `conda` is `2.4.1`. If you want to use newer versions, you can try the following, although we don't guarantee it will work:

- Visit this [link](https://www.tensorflow.org/install/source#gpu) and check the CUDA version for your desired version
- Install `cudatoolkit` with `conda`, specifying the version you find above
- Install `tensorflow` with `pip`, specifying the version you want

```bash
conda install cudatoolkit=11.2 -c nvidia
pip install tensorflow==2.6.0
```

#### PyTorch

Install PyTorch with `conda` (you can follow the official instruction here)

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Verify PyTorch is working and is using GPU. Run this in Python

```python
import torch
print(torch.cuda.is_available())
```

## Code editors

### Jupyter Notebook / Jupyter Lab

Jupyter Notebook / Jupyter Lab is a popular web interface to quickly build Data science projects. It is a Python package so you can also install it with `conda` or `pip`

Install Jupyter Notebook / Lab: https://jupyter.org/install

Note: If you have multiple environments, and you want to switch between environments within Jupyter, install `nb_conda_kernels` in the environment with Jupyter, and install `ipykernel` in the environments you want to access from Jupyter. More information [here](https://github.com/Anaconda-Platform/nb_conda_kernels#installation)

To run Jupyter Notebook / Lab on the GPU server, type this in the terminal

```bash
jupyter notebook
```

Now the Jupyter Notebook is running on your GPU server. However, you need port forwarding to access the Jupyter Notebook from your personal computer. You can use ssh to do this by adding the `-NfL` flags

```bash
ssh -NfL localhost:<remote port>:localhost:<local port> <username>@xxx.dynip.ntu.edu.sg
```

Jupyter Notebook uses port `8888` by default. If someone else is using that port, you will need to forward the port that is used by your Jupyter Notebook.

```bash
ssh -NfL localhost:8888:localhost:8888 acct0000@0000.dynip.ntu.edu.sg
```

Now you can access Jupyter on your local machine at `localhost:8888`. You might need to use a token string shown in the terminal to authenticate Jupyter from your local machine.

### VS Code

Follow this tutorial: [link](https://code.visualstudio.com/docs/remote/ssh)

### PyCharm

Follow this tutorial: [link](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html) and [link](https://www.jetbrains.com/help/pycharm/running-ssh-terminal.html)

## Others

### Common tasks with GPU

Check GPU usage

```bash
nvidia-smi
```

Choose a specific GPU for training

- Export environment variable
```bash
export CUDA_VISIBLE_DEVICES=1
```

- Prefix environment variable when running a programme
```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

- Within Python, can be used in Jupyter also
```python
import os
os.environ[“CUDA_VISIBLE_DEVICES”] = “1”    # must be a string
```

Run detached session for long training schedule

```bash
screen -S <session name> python train.py
```

- Ctrl + A and then Ctrl + D: exit the session, while it is still running in the background
- `screen -r`: resume a background session

Kill a process

```bash
ps -aux | grep "command you used e.g. python train.py"
kill 1234
```

### File transfer

#### With your local machine

Use `scp` to copy files between the GPU server and your personal computer.

```bash
scp <username>@<domain.name>:/remote/path local/path
scp local/path <username>@<domain.name>:/remote/path
```

Alternatively, you can also use a GUI app, such as [FileZilla](https://filezilla-project.org/)

Note: transfer speed will be slow when you are connected via NTU VPN.

#### For code

Use GitHub to backup and store your code. It is also a good practice to use `git` for version control in case you need to revert your code to an earlier version.

The easiest way to use GitHub on remote server is to use [GitHub CLI](https://github.com/cli/cli)

```bash
conda install gh -c conda-forge
gh auth login
```

Some `git` commands to help you get started

```bash
git clone
git pull
git add .
git commit -m "add code"
git push
```

Note: before making a git commit, you have to set your git name and email.

```bash
git config user.name=”Andrew Ng”
git config user.email=”andrew@gmail.com”
```

#### Download files from the web

Use `wget`

```bash
wget <download link>
```

#### Download files from Google Drive

For small files, you can still download them with `wget`. However, there are usually problems with downloading large files from Google Drive. In this case, you should use `gdown`, a Python package

```bash
pip install gdown
```

Gdown guide: https://github.com/wkentaro/gdown
