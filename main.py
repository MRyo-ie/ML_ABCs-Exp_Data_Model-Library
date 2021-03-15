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
from Experiment.example_datas import (
    # 試しデータセット集
    Example_DemandForecasting,
    Example_RestaurantRevenue,
    Example_RecruitRestaurant,
    Example_AshraeEnergy, 
)




def main(exp_set, output_rootpath='.'):
    # Train
    exp_train = Basic_ExpTrain(exp_set)
    exp_train.exec()
    # Evaluate
    exp_eval = Basic_ExpEvaluate(exp_set)
    exp_eval.exec(print_eval=True, is_output_csv=True, output_rootpath=output_rootpath)



if __name__ == '__main__':
    # lines = []
    # for l in sys.stdin:
    #     lines.append(l.rstrip('\r\n'))
    
    args = sys.argv

    exp_set = []
    models = [Prophet_Model(), ]  # LightGBM_Model() LSTM_Model()

    for arg in args[1:]:
        for model in models:
            if arg == 'demand_forecasting':
                example_dataset = Example_DemandForecasting()
                store_item_id_tuple = (9,11)
                dataABC = example_dataset.get_dataABC(store_item_id_tuple)
            elif arg == 'restaurant_revenue':
                # LightGBM_Model のやつ
                # https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
                example_dataset = Example_RestaurantRevenue()
                dataABC = example_dataset.get_dataABC()
            elif arg == 'recruit_restaurant':
                # 平均値 のやつ
                pass
            elif arg == 'ashrae_energy':
                pass
            else:
                raise NotImplementedError('[Error] そのデータセットは未実装です。')

            # dataABC.dataPPP.show_info()
            exp_set.append(
                {
                    'exp_name' : "demand_forecasting",
                    'dataABC'  : dataABC,
                    'modelABC' : model,
                }
            )

    main(exp_set, output_rootpath='')
