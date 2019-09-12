# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def build_list(nums):
    head = None
    tail = None

    for num in nums:
        node = ListNode(num)

        if not tail:
            head = node
            tail = head
        else:
            tail.next = node
            tail = tail.next

    return head


def print_list(head):

    while head:
        print(head.val, end=' ')
        head = head.next
