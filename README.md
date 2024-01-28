# Clone the Project
git clone https://github.com/Zethearc/LLAMA_2-EDUAI.git

# Set Up the Development Environment
## Install virtualenv (if not installed)
pip install virtualenv

## Create and activate a virtual environment
cd LLAMA_2-EDUAI
virtualenv venv
source venv/bin/activate   # On Unix/Linux systems
# .\venv\Scripts\activate  # On Windows

## Install project dependencies
pip install -r requirements.txt

## Install the CUDA toolkit (if using an NVIDIA GPU)
sudo apt install nvidia-cuda-toolkit

## Verify the presence of an NVIDIA GPU
nvidia-smi

## Perform a Linux/Ubuntu upgrade
sudo apt-get update
sudo apt-get install git python cmake build-essential libomp-dev

# Run the Application
python3 app/main.py
