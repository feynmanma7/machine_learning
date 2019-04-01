
def test_args(pin, a=1, b=2, c=3):
    print(pin, a, b, c)


if __name__ == '__main__':
    a = ['1', '2', '3', '4']
    b = list(map(lambda x:(x, 0.01), a))
    print(b)
