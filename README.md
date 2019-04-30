# MeanSum: A Model for Unsupervised Neural Multi-Document Abstractive Summarization

Corresponding paper, accepted to ICML 2019: [https://arxiv.org/abs/1810.05739](https://arxiv.org/abs/1810.05739).

## Requirements

Main requirements:
- python 3
- torch 0.4.0

Rest of python packages in ```requirements.txt```.
Tested in Docker, image = ```pytorch/pytorch:0.4_cuda9_cudnn7```.

## General setup 

Execute inside ```scripts/```:

##### Create directories that aren't part of the Git repo (checkpoints/, outputs/):

```
bash setup_dirs.sh
```

##### Install python packages:

```
bash install_python_pkgs.sh
```

##### The default parameters for Tensorboard(x?) cause texts from writer.add_text() to not show up. Update by:

```
python update_tensorboard.py
```



## Downloading data and pretrained models

### Data

1. Download Yelp data: https://www.yelp.com/dataset and place files in ```datasets/yelp_dataset/```
2. Run script to pre-process script and create train, val, test splits:
    ```
    bash scripts/preprocess_data.sh
    ```
3. Download subword tokenizer built on Yelp and place in 
```datasets/yelp_dataset/processed/```: 
[link](https://s3.us-east-2.amazonaws.com/unsup-sum/subwordenc_32000_maxrevs260_fixed.pkl)

### Pre-trained models

1. Download summarization model and place in 
```stable_checkpoints/sum/mlstm/yelp/batch_size_16-notes_cycloss_honly-sum_lr_0.0005-tau_2.0/```: 
[link](https://s3.us-east-2.amazonaws.com/unsup-sum/sum_e0_tot3.32_r1f0.27.pt)
2. Download language model and place in 
```stable_checkpoints/lm/mlstm/yelp/batch_size_512-lm_lr_0.001-notes_data260_fixed/```: 
[link](https://s3.us-east-2.amazonaws.com/unsup-sum/lm_e24_2.88.pt)
3. Download classification model and place in 
```stable_checkpoints/clf/cnn/yelp/batch_size_256-notes_data260_fixed/```: 
[link](https://s3.us-east-2.amazonaws.com/unsup-sum/clf_e10_l0.6760_a0.7092.pt)


### Reference summaries

Download from: [link](https://s3.us-east-2.amazonaws.com/unsup-sum/summaries_0-200_cleaned.csv).
Each row contains "Input.business_id", "Input.original_review_\<num\>\_id", 
"Input.original_review__\<num\>\_", "Answer.summary", etc. The "Answer.summary" is the
reference summary written by the Mechanical Turk worker.


## Running

Testing with pretrained mode. This will output and save the automated metrics. 
Results will be in ```outputs/eval/yelp/n_docs_8/unsup_<run_name>```

NOTE: Unlike some conventions, 'gpus' option here represents the GPU ID (the one which is visible) and NOT the number of GPUs. Hence, for a machine with a single GPU, you will give gpus=0
```
python train_sum.py --mode=test --gpus=0 --batch_size=16 --notes=<run_name>
```

Training summarization model (using pre-trained language model and default hyperparams).
The automated metrics results will be in ```checkpoints/sum/mlstm/yelp/<hparams>_<additional_notes>```.:
```
python train_sum.py --batch_size=16 --gpus=0,1,2,3 --notes=<additional_notes> 
```
