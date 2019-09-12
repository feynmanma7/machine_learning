from returing.algorithms.list.build_list import build_list, print_list


def mergeTwoLists(l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """

    head = None
    tail = head

    while l1 and l2:
        small = None

        if l1.val < l2.val:
            small = l1
            l1 = l1.next
        else:
            small = l2
            l2 = l2.next

        if not tail:
            head = small
            tail = head
        else:
            tail.next = small
            tail = tail.next

    while l1:
        if not tail:
            head = l1
            l1 = l1.next
            tail = head
            continue

        tail.next = l1
        l1 = l1.next
        tail = tail.next

    while l2:
        if not tail:
            head = l2
            l2 = l2.next
            tail = head
            continue

        tail.next = l2
        l2 = l2.next
        tail = tail.next

    return head


if __name__ == '__main__':
    arr1 = []
    arr2 = [0]

    l1 = build_list(arr1)
    print('l1')
    print_list(l1)
    print('\n*****')

    l2 = build_list(arr2)
    print('l2')
    print_list(l2)
    print('\n=====')

    head = mergeTwoLists(l1, l2)
    print('merge')
    print_list(head)



