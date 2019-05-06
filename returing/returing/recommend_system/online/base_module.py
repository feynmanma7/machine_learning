import time


class Module:

    def __init__(self, *args, **kwargs):
        pass

    def _on_start(self):
        self.start_time = time.time()

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        try:

            debug = False

            if 'debug' in kwargs:
                debug = kwargs['debug']

            if debug:
                self._on_start()

            result = self.run(self, *args, **kwargs)

            if debug:
                self._print_run_time()

            return result
        except ZeroDivisionError:
            print('Wrong', self.__class__)
            return -1


    def _get_run_time(self):
        run_time = self.end_time - self.start_time

        return run_time

    def _print_run_time(self):
        end_time = time.time()
        run_time = end_time - self.start_time
        print('run_time', run_time * 1000, 'ms')


    def _on_end(self):
        self.end_time = time.time()


if __name__ == '__main__':

    m = Module()

    args = []
    context = {'a':'hello', 'b':'world', 'debug':True}

    result = Module()(*args, **context)
    print(result)

    #m.time_monitor(*args, **context)