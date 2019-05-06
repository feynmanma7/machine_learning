from returing.recommend_system.online.base_module import Module


class Recall(Module):

    def __init__(self, *args, **kwargs):
        pass


class Cid3PrefRecall(Recall):
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return 'cid3Pref'



if __name__ == '__main__':
    context = {'debug':True, 'c':'hi'}
    result = Recall()(**context)
    print(result)
