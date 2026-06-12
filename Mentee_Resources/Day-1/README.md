# Day 1: Environment Configuration and Architecture

This session details the architectural prerequisites for Machine Learning development, covering environment isolation, dependency management, and hardware acceleration mechanisms.

## Conceptual Overview: Dependency Management
Data science projects require strict dependency versioning. Global installations lead to version conflicts across concurrent projects (often termed "Dependency Hell"). Virtual environments resolve this by providing project-specific, isolated execution contexts containing independent Python binaries and site-packages.

## Local Configuration Methods

### Method 1: Python Virtual Environments (`venv`)
`venv` is a lightweight, native Python module suitable for standard Data Science workflows.

**Execution:**
1. Initialize: `python -m venv ml_env`
2. Activate (Windows): `ml_env\Scripts\activate`
3. Activate (Unix): `source ml_env/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

### Method 2: Conda Environments
Conda is a cross-platform package manager capable of resolving non-Python dependencies (e.g., system-level C libraries, CUDA runtimes). It is the preferred standard for deep learning workloads.

**Execution:**
1. Require [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation.
2. Initialize from specification: `conda env create -f environment.yaml`
3. Activate: `conda activate college_ml_lab`

## Cloud Configuration: Google Colab
Google Colab provides a managed Jupyter Notebook environment executing on ephemeral Linux instances. It eliminates local configuration overhead and provides native hardware acceleration access.

## Hardware Acceleration Paradigms
- **Central Processing Unit (CPU):** Utilized for sequential logic and traditional ML algorithms (e.g., Scikit-Learn decision trees).
- **Graphics Processing Unit (GPU):** Utilized for massively parallel tensor operations inherent to Deep Learning (e.g., PyTorch Neural Networks). 
- **APIs:** Frameworks like PyTorch interface with GPUs via CUDA (Nvidia architecture) or MPS (Apple Metal Performance Shaders).

## Verification Protocol
1. Execute `diagnostic.py` via the terminal to confirm the presence of core libraries (`numpy`, `pandas`, `sklearn`, `torch`, `matplotlib`, `jupyter`).
2. Execute the `Hello_ML.ipynb` notebook to validate data visualization pipelines, classical ML inference, and PyTorch tensor operations on available hardware.
