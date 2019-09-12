

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]




def generate(res, cur, left, right):
    if left == 0 and right == 0:
        res.append(cur)

    if left > 0:
        generate(res, cur + '(', left - 1, right)
    if right > 0 and right > left:
        generate(res, cur + ')', left, right - 1)



def generateParenthesis(n):

    if n <= 0:
        return []

    left = n
    right = n
    res = []
    cur = ""
    generate(res, cur, left, right)

    return res


if __name__ == '__main__':

    n = 3
    res = generateParenthesis(n)
    print(res)