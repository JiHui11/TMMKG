# TMMKG
the repository is the implementation of TMMKG
## Requirements
```
Python 3.7
Pytorch 1.7
timm 0.4.5
numpy 1.21.6
scikit-learn 1.0.2
```
## Download Datasets
Please download the datasets from [Google Drive] and save them into the `data` folder.
## Running
### 1 Semantic Alignment
```
cd SemAlign
python ave.py --train --threshold=0.095
```
### 2. TMMKG Generation
```
cd TMMKG-Gen
```
* Training TMMKG Generation
```
python tva-triplets.py --status train --kgc_mode three_modality --threshold=0.095 --entity_threshold=0.40
```
* Testing TMMKG Generation
```
python tva-triplets.py --status test --kgc_mode three_modality --threshold=0.095 --entity_threshold=0.40
```
