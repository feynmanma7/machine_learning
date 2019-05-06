from returing.recommend_system.online.base_module import Module


class Context(Module):

    def __init__(self, *args, **kwargs):
        super(Context, self).__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        context = {'a':'xxx', 'b':'yyy'}
        return context