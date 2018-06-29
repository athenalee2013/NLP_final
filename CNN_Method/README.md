### 獲得處理後的資料，方法有兩種
1. 直接下載要用來訓練的.pkl
```Bash
wget -O ./pkl/sem-relations.pkl.gz 'https://www.dropbox.com/s/0nmgkf2l2eosq6h/sem-relations.pkl.gz?dl=1'
```
2. 執行前處理code
```Bash
python CreateTrainTestFiles.py
python preprocess.py
```
