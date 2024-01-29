# FlowMI
Implementation of code in Flow Moment Matching paper

## Setting Up Enviornment and downloading requirements
### Enviornement Requirements
```bash
conda create -n idad_code
conda activate idad_code
```
All the code in this repo will only work with python 
version 3.10
```bash
conda install python=3.10
```


## Requirements for Beyond Gaussian Benchmark/McAlester
```bash
pip install flowjax
pip install pydantic
pip install tensorflow-probability
pip install plotly
```
## Requirements for Sequential Design Experiments
```bash 
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install pyro-ppl
pip install mlflow
pip install git+https://github.com/google-research/torchsde.git
```

## For SIR Experiment
To run the SIR experiment, first run the iDAD code to generate data
```bash
python3 epidemic_simulate_data.py \
    --num-samples=100000 \
    --device <DEVICE>
```
