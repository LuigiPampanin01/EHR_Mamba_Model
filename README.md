# Selective_SSM_for_EHR_Classification
Project 24 for 02456 2024

# Background
This repository allows you to train and test a variety of electronic health record (EHR) classification models on mortality prediction for the Physionet 2012 Challenge (`P12`) dataset. More information on the dataset can be found here (https://physionet.org/content/challenge-2012/1.0.0/). Note that the data in the repository has already been preprocessed (outliers removed, normalized) in accordance with https://github.com/ExpectationMax/medical_ts_datasets/tree/master and saved as 5 randomized splits of train/validation/test data. Adam is used for optimization.

# Create Environment
The dependencies are listed for python 3.9.

To create an environment and install required packages, run one of the following: 

## Venv way
```
# CD into the project folder
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

## Conda way [NOT INSTALLED IN THE HPC]
```
# CD into the project folder
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
conda create --name <your-env-name> python=3.9
conda activate <your-env-name> 
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html
```





# Run models 
4 baseline models have been implemented in `Pytorch` and can be trained/tested on `P12`. Each has a unique set of hyperparameters that can be modified, but I've gotten the best performance by running the following commands (_Note: you should unzip the data files before running these, and change the output paths in the commands_):

To unzip the data files, run the following command:

```bash
python extract_P12.py
```

> **Note:** `--output_path` is a directory where the model checkpoints and logs will be saved. i suggest you to:
> 1. Create a directory (e.g. `mkdir output`)
> 2. Run the desired command with `--output_path=./output` (remember the dot)

> **Note 2:** Running one of this commands will create files also in `processed_datasets`

**AT LEAST IN THE BEGINNING PLEASE DO NOT PUSH YOUR OUTPUT DIR OR `./processed_datasets` FILES TO THE REPO ESPECIALLY IF WE DON'T KNOW IF THEY'RE NEEDED**

![image](https://pbs.twimg.com/media/GBeYh5dW4AA4Urp.jpg)

> Non sto scherzando vi sparo

## [Transformer](https://arxiv.org/abs/1706.03762)

```bash
python cli.py --epochs=100 --batch_size=16 --model_type=transformer --dropout=0.2 --attn_dropout=0.1 --layers=3 --heads=1 --pooling=max --lr=0.0001 --output_path=your/path/here
``` 


## [SEFT](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series)

```bash
python cli.py --output_path=your/path/here --model_type=seft --epochs=100 --batch_size=128 --dropout=0.4 --attn_dropout=0.3 --heads=2 --lr=0.01 --seft_dot_prod_dim=512 --seft_n_phi_layers=1 --seft_n_psi_layers=5 --seft_n_rho_layers=2 --seft_phi_dropout=0.3 --seft_phi_width=512 --seft_psi_width=32 --seft_psi_latent_width=128 --seft_latent_width=64 --seft_rho_dropout=0.0 --seft_rho_width=256 --seft_max_timescales=1000 --seft_n_positional_dims=16
```

## [GRUD](https://github.com/PeterChe1990/GRU-D/blob/master/README.md)

```bash
python cli.py --output_path=your/path/here --model_type=grud --epochs=100 --batch_size=32 --lr=0.0001 --recurrent_dropout=0.2 --recurrent_n_units=128
```

## [ipnets](https://github.com/mlds-lab/interp-net)

```bash
python cli.py --output_path=your/path/here --model_type=ipnets --epochs=100 --batch_size=32 --lr=0.001 --ipnets_imputation_stepsize=1 --ipnets_reconst_fraction=0.75 --recurrent_dropout=0.3 --recurrent_n_units=32
```


# DIY
You are welcome to fork the repository and make your own modifications :) 
