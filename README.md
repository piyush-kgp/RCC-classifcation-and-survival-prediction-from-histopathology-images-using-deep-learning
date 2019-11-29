
# RCC classifcation and survival prediction from histopathology images using deep learning
This is an implementation of the paper [Pan-Renal-Cell-Carcinoma-classifcation-and-survival-prediction-from-histopathology-images-using-deep-learning]((https://www.nature.com/articles/s41598-019-46718-3.pdf) by Sairam et. al.


Implementing this paper can be broken down to following steps:
- Downloading SVS files containing WSIs from TCGA (~558 GB)
- Patch extraction and pre-processing
- Cancer vs Normal classification for RCC subtypes KIRC, KIRP, KICH using CNN
- Subtype classification using CNN and CNN + DAG-SVM
- Downloading clinical supplement XML files containing survival information from TCGA
- Pre-process XML files and store patient-id, days_to_death tuples in a CSV file
- For each patient's corresponding tissue images, find the patches with highest probability of cancer (Top 100). Use the model in step 3 for this.
- Segment these patches using some technique such as watershed algorithm
- Get shape features such as total area, perimeter, eccentricity etc for each patient
- Fit a COX regression model for X = shape features, Y = days survived


## Downloading SVS files containing whole slide images from TCGA
Set up envirionment for `gdc-client`
```
python2.7 -m virtualenv v2env
source v2env/bin/activate
pip install -r gdc_req_p2.txt
```

Use [gdc-client](https://github.com/NCI-GDC/gdc-client/) to download with multiple processes:
```
gdc-client download -n 10 -m manifest.txt
```
- [KIRC manifest]()
- [KIRP manifest]()
- [KICH manifest]()

---
Here on we will use Python3. To set up the envirionment:
```
python3 -m virtualenv v3env
source v3env/bin/activate
pip install -r requirements.txt
```
---

## Patch extraction and preprocessing
For this you need to install [openslide]() on a Python3 envirionment.
Then run patch extraction code:
```
python3 patch_extraction.py --root_dir $ROOT_DIR --dest_dir $DEST_DIR --level $LEVEL
```

`level` argument is 0 for 40x and 1 for 20x magnification.

---

The code for classifier is  generalized. It can be used as is for any number of classes.

---
## Cancer vs Normal classification for RCC subtypes KIRC, KIRP, KICH using CNN
Dependencies: PyTorch:1.1.0 Cuda 7.3 CuDNN 10.0
```
python3 classifier.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR
```
It accepts following arguments from command line:
`batch_size`, `num_epochs`, `learning_rate`, `log_dir`, `model_checkpoint`.

`model_checkpoint` can be used to resume train from a checkpoint-ed model (parameters must be same).


## Subtype classification using CNN and CNN + DAG-SVM

```
python3 classifier.py --train_dir $TRAIN_DIR --val_dir $VAL_DIR
```

```
python dag_svm.py \
--kirc_train_file $kirc_train_file \
--kirp_train_file $kirp_train_file \
--kich_train_file $kich_train_file \
--kirc_valid_file $kirc_valid_file \
--kirp_valid_file $kirp_valid_file \
--kich_valid_file $kich_valid_file
```

## Downloading  XML files containing survival information from TCGA

[KIRC Clinical supplement Manifest]()

## Pre-process XML files and store in a CSV file
`python3 survival_analysis/collate_survival_data.py`


## Shape Feature extraction

## COX regression
