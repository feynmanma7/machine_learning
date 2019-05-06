from returing.recommend_system.online.context import Context

from returing.recommend_system.online.base_recall import Cid3PrefRecall

def main():

    context = None

    param_dict = None

    #param_dict['loc'] = 'cm'

    context = Context()()
    print(context)

    cm_recalls = Cid3PrefRecall()(**context)
    print(cm_recalls)

    #param_dict['loc'] = 'cf'
    cf_recalls = Cid3PrefRecall()(**context)
    #cf_recalls = UpDownFiltering(param_dict, context)(cf_recalls)

    #cm_predicts = CosSimPredict(param_dict, context)(cm_recalls)
    #cm_predicts = Cid25Rerank(param_dict, context)(cm_predicts)

    #cf_predicts = CosSimPredict(param_dict, context)(cf_recalls)
    #cf_predicts = Cid25Rerank(param_dict, context)(cf_predicts)

    #results = RatioMerge(ratios, context)(cm_predicts, cf_predicts)

    #return results


if __name__ == '__main__':
    main()