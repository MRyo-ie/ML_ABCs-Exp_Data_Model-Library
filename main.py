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

    if args[1] == '-e':
        if args[2] == 'demand_forecasting':
            for m in models:
                ex_dem_fore = Example_DemandForecasting()
                limit_d=(9,11)
                dataABC = ex_dem_fore.get_dataABC(
                    limit_d
                )
                # dataABC.dataPPP.show_info()
                exp_set.append(
                    {
                        'exp_name' : "demand_forecasting",
                        'dataABC'  : dataABC,
                        'modelABC' : m,
                    }
                )
        if args[2] == 'restaurant_revenue':
            # LightGBM_Model のやつ
            # https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
            pass
        if args[2] == 'recruit_restaurant':
            # 平均値 のやつ
            pass
        if args[2] == 'ashrae_energy':
            pass

    main(exp_set, output_rootpath='')
