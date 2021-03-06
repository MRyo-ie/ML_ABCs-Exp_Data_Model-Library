import sys

from Experiment import (  # .Experiment,  .
    # DataABC
        DataPPP,
        Raw_Data,
    # ModelABC
        Prophet_Model,
    # ExpermentABC
        Basic_ExpTrain,
        Basic_ExpEvaluate,
)



def demand_forecasting(args):
    for i, arg in enumerate(args):
        print("Path{} : {}".format(i, arg))

    ###  train  ###
    dataPPP = get_demand_forecasting_dataPPP(args[1])
    dataPPP.X_train = dataPPP.X
    dataPPP.Y_train = dataPPP.Y
    # dataPPP.split_train_valid_test()   ## 今回は、すでに train/test に分かれているので不要。

    ###  test  ###
    dataPPP_test = get_demand_forecasting_dataPPP(args[2])
    dataPPP.X_test = dataPPP_test.X
    dataPPP.Y_test = dataPPP_test.Y
    
    exp_set = [
        {
            'exp_name' : "demand_forecasting",
            'dataABC'  : Raw_Data(dataPPP, exist_valid=False, exist_Q=False),
            'modelABC' : Prophet_Model(),
        }, 
    ]

    # Train
    exp_train = Basic_ExpTrain(exp_set)
    exp_train.exec()
    # Evaluate
    exp_eval = Basic_ExpEvaluate(exp_set)
    exp_eval.exec(print_eval=True, is_output_csv=True, output_dirpath=dataPPP.dir_path)


def get_demand_forecasting_dataPPP(path):
    dataPPP = DataPPP()
    dataPPP.load_data(path)

    dataPPP.df = dataPPP.df.rename(
            columns={'date': 'ds', 'sales': 'y'})
    
    # ストアID = 1, 商品ID = 1  に限定して予測してみる。
    df = dataPPP.df
    dataPPP.df = df[(df.store == 1) & (df.item == 1)]
    # dataPPP.show_info()

    # XQY に分解
    dataPPP.split_XQY(
        X_clms=['ds', 'store', 'item'],
        Y_clms=['y'],
        Q_clms=None)   # ['store', 'item']

    return dataPPP



if __name__ == '__main__':
    # lines = []
    # for l in sys.stdin:
    #     lines.append(l.rstrip('\r\n'))
    
    args = sys.argv
    demand_forecasting(args)




