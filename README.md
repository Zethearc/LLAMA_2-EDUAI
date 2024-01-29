# Clone the Project
```bash
git clone https://github.com/Zethearc/LLAMA_2-EDUAI.git
```
# Set Up the Development Environment
## Install virtualenv (if not installed)
```bash
pip install virtualenv
```
## Create and activate a virtual environment
```bash
cd LLAMA_2-EDUAI
virtualenv venv
source venv/bin/activate   # On Unix/Linux systems
# .\venv\Scripts\activate  # On Windows
```
## Install project dependencies
```bash
pip install -r requirements.txt
```
## Install the CUDA toolkit (if using an NVIDIA GPU)
```bash
sudo apt install nvidia-cuda-toolkit
```
## Verify the presence of an NVIDIA GPU
```bash
nvidia-smi
nvcc --version
```

## Install llamacpp

```bash
git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

## Perform a Linux/Ubuntu upgrade
```bash
sudo apt-get update
sudo apt-get install build-essential
```
# Run the Application
```bash
python3 app/main.py
```