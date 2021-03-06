
## Setup
```
$ pip install -r requirements.txt
```

## Quick Start
- 実行方法
    ```
    $ python3 main.py 【データセット名】
    ```
    - result/ 以下に、
      - 予測結果.csv
      - グラフ.png
    ができるはず。


## 理論
### Exp(Data,Model)
- 「機械学習タスクを解くシステムの実装」を、Data, Model および Experiment（実験設定）に分ける。
  - それぞれで abstruct クラスを作成した。
  - それを継承することで、柔軟かつ高速に、「機械学習タスク」を実装できるようにした。
- 具体的には、
  1. 実験定義： Data, Model のインスタンスを辞書にまとめる。
  2. 実験の実行： Experient に **1.実験定義** を読み込んで、実行（.exec()） する。
```
exp_set = []
exp_set.append(
    {
        'exp_name' : "demand_forecasting",
        'dataABC'  : dataABC,
        'modelABC' : model,
    }
)

# Train
exp_train = Basic_ExpTrain(exp_set)
exp_train.exec()
# Evaluate
exp_eval = Basic_ExpEvaluate(exp_set)
exp_eval.exec(print_eval=True, is_output_csv=True, output_rootpath=output_rootpath)
```
- Exp(Data,Model) 形式を前提に、
  - 統計 (Statistic)
  - Machine Learning (ML)
  - Deep Learning (DL)
  のモデルをラップした。
  - これにより、
    - 複数モデルでの比較
    - 実験設定の変更の反映
    を素早く、安全に実装できるようにすることを目指す。

### Data
- **I/O 規格**： XQA（VQA形式の拡張。(X, Q) --Model-> A）
  - X : 説明変数(X)、画像(V)、問題文、...
    - csv
      - class_n  : 辞書。分類問題（カテゴリー、タイプ、グループ など）を想定。
        - データ型は、int。（カテゴリ番号）
          - str でも、自動で番号に変換される。
        - int（番号）をカテゴリに変換するための辞書（の集合）も同時に作る。
          - `dataABC.int_to_name(dict_num=n)`
      - num_n   : int, float など。回帰問題を想定。
        - pandas における int64, float64
      - date_n  : 日時。
        - ~~pandas における datetime64~~
        - 年／月／日、時／分／秒　に column を分解する。（未）
        - stream(時系列性) として**扱いたくない**場合は、こちらにする。
      - stream_n : 時系列データ。
        - pandas における datetime64 や id。
        - stream(時系列性) として**扱いたい**場合は、こちらにする。
          - Prophet などを想定。
      - text_n  : 文、文章
      - img_n   : 画像
        - データ型は str。（ファイルパス（.csv からの相対パス）のため）
    - json
      - 構造情報。
      - csvデータの複数のカラム間の、階層構造 や 順序、位置関係 を定義する。
        - ので、csv には、構造的な情報は含まない。
        - object や category(class) は、できる限り json で表現する。
        - = csv には、str は書かないのが理想。
  - Q : タスク名、質問文(Q)、コマンド
    - kaggle や [現実的なタスク](https://ainow.ai/2020/12/17/246963/) を見ながら研究開発中。
    - 例）
      - 分類： クラス、画像、文章、...
      - 回帰： 
      - 画像： 分類、Detection, Segmentation, 生成, ...
      - 音声： 
      - 問題文： QA、VQA、ドメイン特化、...
    - 
  - Y : 目的変数(Y)、回答(A)、行動(a)、...
- row： 
  - dataPPP.split_train_valid_test()
    - train, valid, test に分割する。
    - XQA それぞれに、train, valid, test ができる。
- **examples**
  - [x] `demand_forecasting`
    - 参考： https://qiita.com/simonritchie/items/8068a41765a9a43d0fea
  - [ ] `restaurant_revenue`
    - 参考：https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
  - [ ] `recruit_restaurant`
    - 参考：https://www.codexa.net/kaggle-recruit-restaurant-visitor-forecasting-handson/#Kaggle-3
  - [ ] `ashrae_energy` （未）
    - 参考：https://www.deep-percept.co.jp/blog/category02/20200623423/
    - 現在調整中...

### Model :
  - I/O
    - X : 説明変数(X)、画像(V)、
    - Q : タスク名、質問文(Q)、コマンド
    - Y : 目的変数(Y)、回答(A)、行動(a)、...
  - アルゴリズム
    - 統計
      - [ ] 月〜金 で分解して加重平均を取る。
    - 機械学習
      - [x] Prophet
      - [ ] LightBGM
    - DeepLearning
      - [ ] LSTM
      - [ ] Attention系

- Exeperiment : Basic_ExpTrain, Basic_ExpEvaluate

