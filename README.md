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
Please download the datasets from [阿里云盘](https://www.alipan.com/s/939eFaK2ygg). and save them into the `data` folder.
## Running
### 1. Semantic Alignment
```
cd SemAlign
python ave.py --train --threshold=0.095
```

### 2. Generate Temporal Visual Knowledge Graphs and Audio Labels
1. **Temporal Visual Knowledge Graphs**  
   Use  **[STTran](https://github.com/yrcong/STTran)** to generate the following files:
   - `train_video.json`
   - `test_video.json`

2. **Temporal Audio Labels**  
   Use **[AST](https://github.com/YuanGongND/ast)** to generate the audio classification model:
   - `best_audio_model.pth`
     
Save all the generated files into the `data` folder.

### 3. TMMKG Generation

To generate the **Temporal Multi-Modal Knowledge Graphs (TMMKG)**, follow the steps below:

```
cd TMMKG-Gen
```
* To generate TMMKG of the training data, run the following command:
```
python tva-triplets.py --status train --kgg_mode three_modality --threshold=0.095 --entity_threshold=0.40
```
* To generate TMMKG of the testing data, run the following command:
```
python tva-triplets.py --status test --kgg_mode three_modality --threshold=0.095 --entity_threshold=0.40
```
### 4. TMMLP
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
