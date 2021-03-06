##

## Setup
```
$ pip install -r requirements.txt
```

## Quick Start
- Exp(Data,Model)
  - Data : 時系列データ
    - https://qiita.com/simonritchie/items/8068a41765a9a43d0fea
  - Model : Prophet
  - Exeperiment : Basic_ExpTrain, Basic_ExpEvaluate

- 実行方法
    ```
    $ python3 main.py Experiment/DataABC/data/demand-forecasting-kernels-only/train.csv Experiment/DataABC/data/demand-forecasting-kernels-only/test.csv
    ```
    - result/ 以下に、
      - 予測結果.csv
      - グラフ.png
    ができるはず。
