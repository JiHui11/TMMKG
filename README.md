# TMMKG
the repository is the implementation of TMMKG
## Requirements
```
Python 3.7
Pytorch 1.7
timm 0.4.5
numpy 1.21.6
scikit-learn 1.0.2
imageio 2.31.2
pydub 0.25.1
```
## Download Datasets
Please download the datasets from [阿里云盘](https://www.alipan.com/s/Z3woAzaXa5c) (密码 zh0l). and save them into the `data` folder.
## Running
### 1. Semantic Alignment
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
python tva-triplets.py --status train --kgg_mode three_modality --threshold=0.095 --entity_threshold=0.40
```
* Testing TMMKG Generation
```
python tva-triplets.py --status test --kgg_mode three_modality --threshold=0.095 --entity_threshold=0.40
```
### 3. TMMLP
```
cd TMMLP
```
* Train:
```
python main.py --train --nb_epoch=30
```
* Test:
```
python main.py --model_path ./kgc_models/best_model.tar
```
