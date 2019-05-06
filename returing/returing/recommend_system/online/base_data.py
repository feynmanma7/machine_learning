from returing.recommend_system.online.base_module import Module


class Data(Module):

    def __init__(self):
        pass

    def run(self, *args, **kwargs):
        return kwargs['hbase']


if __name__ == '__main__':
    context = {'debug':False, 'hbase':'hello'}
    result = Data()(**context)
    print(result)

