import sys

from Experiment import (  # .Experiment,  .
    # DataABC
        DataPPP,
        Raw_Data,
    # ModelABC
        Prophet_Model,
        LightGBM_Model,
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
    models = [
        # Prophet_Model(), 
        LightGBM_Model()
    ]  # LSTM_Model()

    dataPPP = None
    example_dataset = None
    arg = args[1]
    if arg == 'demand_forecasting':
        example_dataset = Example_DemandForecasting()
        # example_dataset.analyze_dataset()
        # sys.exit()
        store_item_id_tuple = (9,11)
        example_dataset.initialize_dataPPP(
            store_item_id_tuple=(3,8),
            clmns_conv_dict = {
                'date':'stream0', 'store':'class0', 'item':'class1',
                'sales':'num0',},
            clmns_XQA = {
                'X' : ['stream0', 'class0', 'class1',],
                'Q' : None,
                'Y' : ['num0'], },
            is_shuffle=True,)
        # dataPPP = example_dataset.dataPPP
        # dataPPP.show_analyzed(dataPPP.Y_valid)
        # print(dataPPP.clmns_XQA)
    elif arg == 'restaurant_revenue':
        # LightGBM_Model のやつ
        # https://career-tech.biz/2019/10/16/python-kaggle-restaurant/
        example_dataset = Example_RestaurantRevenue()
        example_dataset.initialize_dataPPP(
            clmns_conv_dict = {
                'Open Date':'stream0', 'City':'class0', 'City Group':'class1', 'Type':'class2',
                'P1':'num1', 'P2':'num2', 'P3':'num3', 'P4':'num4', 'P5':'num5', 'P6':'num6', 'P7':'num7', 'P8':'num8', 'P9':'num9', 'P10':'num10', 
                'P11':'num11', 'P12':'num12', 'P13':'num13', 'P14':'num14', 'P15':'num15', 'P16':'num16', 'P17':'num17', 'P18':'num18', 'P19':'num19', 'P20':'num20', 
                'P21':'num21', 'P22':'num22', 'P23':'num23', 'P24':'num24', 'P25':'num25', 'P26':'num26', 'P27':'num27', 'P28':'num28', 'P29':'num29', 'P30':'num30', 
                'P31':'num31', 'P32':'num32', 'P33':'num33', 'P34':'num34', 'P35':'num35', 'P36':'num36', 'P37':'num37',  'revenue':'num38'},
            clmns_XQA = {
                'X' : [
                    'stream0', #'class0', 'class1', 'class2',
                    'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 
                    'num11', 'num12', 'num13', 'num14', 'num15', 'num16', 'num17', 'num18', 'num19', 'num20', 
                    'num21', 'num22', 'num23', 'num24', 'num25', 'num26', 'num27', 'num28', 'num29', 'num30', 
                    'num31', 'num32', 'num33', 'num34', 'num35', 'num36', 'num37', ],
                'Q' : None,
                'Y' : ['num38'], },)
    elif arg == 'recruit_restaurant':
        # 平均値 のやつ
        pass
    elif arg == 'ashrae_energy':
        pass
    else:
        raise NotImplementedError('[Error] そのデータセットは未実装です。')

    for model in models:
        exp_set.append(
            {
                'exp_name' : "demand_forecasting",
                'dataABC'  : example_dataset.get_dataABC(),
                'modelABC' : model,
            }
        )

    main(exp_set, output_rootpath='')
