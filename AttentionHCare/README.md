# trajectories-prediction

## Project setup
We recommend using conda with python 3.7, numpy 1.19 and tensorflow 1.15:

```
conda create --name py37_tf115 python=3.7
conda activate py37_tf115
conda install -y -c conda-forge numpy==1.19
conda install -y -c conda-forge scikit-learn
```

For CPU:
```
pip install tensorflow==1.15
```

For GPU: 
```
pip install tensorflow-gpu==1.15
```

### Execution example (from project main dir):

#### Training (for MIMIC ICD-9 preprocessed data)
`python3.7 AttentionHCare-train.py "data/mimic_90-10_855" compiled_models/encdec-model --hiddenDimSize=[1084]`

#### Evaluating best saved model (for MIMIC ICD-9 preprocessed data)
`python3.7 AttentionHCare-evaluate.py "data/mimic_90-10_855" compiled_models/encdec-model.50`

> Data is not included in this project. If you want to obtain it, please contact one of the authors