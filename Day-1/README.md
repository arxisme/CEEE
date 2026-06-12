# Day 1: ML Environment Architecture and Setup

Welcome to Day 1. This session focuses on the foundational architecture required for Data Science and Deep Learning. We will cover environment isolation, package management, and hardware acceleration.

## The Problem: Dependency Hell
Machine Learning projects rely on hundreds of interconnected libraries. Installing them globally on your system inevitably leads to version conflicts (e.g., Project A requires `numpy 1.20`, Project B requires `numpy 1.24`). 

**The Solution:** We use Virtual Environments. A virtual environment is an isolated sandbox directory that contains its own Python executable and independent package installations.

## Setup Architectures

### 1. Python Venv (Standard Isolation)
`venv` is built into Python. It is lightweight and perfect for traditional ML (Scikit-Learn, Pandas).

**Steps:**
1. Create the sandbox: `python -m venv ml_env`
2. Activate it:
   - Windows: `ml_env\Scripts\activate`
   - Mac/Linux: `source ml_env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

### 2. Conda (Data Science Standard)
Conda is heavier but more powerful. Unlike `venv`, Conda can manage non-Python binaries (like C++ libraries or CUDA toolkits) making it the industry standard for complex Deep Learning environments.

**Steps:**
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create environment from our file: `conda env create -f environment.yaml`
3. Activate it: `conda activate college_ml_lab`

### 3. Google Colab (Cloud Architecture)
Colab provides an ephemeral Linux environment hosted by Google. It abstracts away environment setup and, crucially, provides free access to GPUs.

## Hardware Acceleration (CPU vs GPU)
Traditional ML models (Random Forests, linear regression) train sequentially on the CPU. Deep Learning models (Neural Networks) require parallel matrix multiplications. 
- GPUs (Graphics Processing Units) contain thousands of smaller cores perfectly suited for this parallel math. 
- PyTorch bridges the hardware gap using APIs like CUDA (for Nvidia GPUs) or MPS (for Apple Silicon).

## Lab Exercises
1. Choose either `venv` or `conda` and set up your local environment.
2. Run `python diagnostic.py` to verify the installation.
3. Launch `jupyter notebook` and open `Hello_ML.ipynb`.
4. Execute the notebook to observe Pandas visualization, a Scikit-Learn training loop, and a PyTorch neural network hardware check.
