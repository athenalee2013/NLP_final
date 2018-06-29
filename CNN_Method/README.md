# 1. 執行前處理code，執行過程中同時會下載Word Embedding
```Bash
# 由於處理中有用到nltk，如果套件沒安裝齊全會不能跑CreateTrainTestFiles.py
# 而CreateTrainTestFiles.py產生的東西有直接上傳了
# 如果想直接執行preprocess.py也可以
python CreateTrainTestFiles.py
python preprocess.py
```
# 2. Tesing
First, you need to download the trained model to predict the results. The download link is shown below:<br> 
https://drive.google.com/file/d/1j6TlNNBDXtdhxevr1jMMdGFhCuHeEhOh/view?usp=sharing

Then, you can predict the results using the command below:
```Bash
python3 predict.py --model_name model_best.h5 --predict_file_name Results.txt
```
# 3. Evaluation
```Bash
perl semeval2010_task8_scorer-v1.2.pl <--predict_file_name> answer_key.txt
# <predict_file_name> is 'Results.txt' in 2.
```
# 4. Training (Optional)
```Bash
python3 train_cnn.py --model_name model.h5
```
