# Explaining, Evaluating and Enhancing Neural Networks' Learned Representations

### Overview
This repository contains the code implementation 
for the submission *Explaining, Evaluating and Enhancing Neural Networks' Learned Representations*. 
We use [PyTorch Lightening](https://github.com/PyTorchLightning/pytorch-lightning) for 
training the models in our experiments.

This repository is organised as follows:
* `scores/` contains the implementation for our score and our implementation for computing the gradient-based attributions.
* `dataset_drift.py` is a script for running and reproducing the experiments described in Section 4.4.
* `train_constrs.py` is a script for running and reproducing the experiments described in Section 4.5.

#### Requirements
To run our examples, the main requirements are listed in the `environment_gpu.yml` file. The main requirements used are the following:
```
python=3.8.5
pytest=6.2.1
cudatoolkit=10.1
cudnn=7.6.5
numpy=1.19.2
scipy=1.5.2
pytorch=1.7.1
```

#### Conda
We suggest creating a new environment to run our experiments:
```
conda env create -f environment_gpu.yml
conda activate emb_XAI
```

