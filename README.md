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