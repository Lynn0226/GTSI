# GTSI: Source Identification in Social Networks Based on Graph Transformer

Source code for ***Source Identification in Social Networks Based on Graph Transformer***.

This repository provides the implementation for identifying the source of information diffusion in social networks using GTSI.

## Get Started

The code has been tested under the following environment:

- Python 3.8  
- PyTorch 1.8  

### Create and activate the conda environment

```bash
conda create -n GTSI python=3.8
conda activate GTSI
pip install -r requirements.txt
```
### Download data
Download the dataset from the following link and the extraction code is `zx7u`:  
[https://pan.baidu.com/s/1fW4_hezplQIeEPogUUCGuA](https://pan.baidu.com/s/1fW4_hezplQIeEPogUUCGuA)  

After downloading, please place all the files in the folder to the directory: `./data/PROP/`


### Train the model
Basic training parameters are specified in the ```./configs/``` directory. Start training with the following command:
```bash
python main_PROP_source_classification.py --gpu_id 0 --config configs PROP_GraphTransformer_LapPE_KARATE_inf0.2_full_graph_BN.json
```