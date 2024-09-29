from collections import defaultdict
from typing import Tuple



def data_partition(fname: str) -> Tuple[defaultdict[int, list[int]], defaultdict[int, list[int]], defaultdict[int, list[int]], int, int]:
    """
    Parse and partition raw dataset at given filepath.

    Args:
        fname (str) : filepath to the raw dataset (absolute / relative).

    Returns:
        user_train (defaultdict[int, list[int]]) : mapping from user id (int) to item id list (except last two items).
        user_valid (defaultdict[int, list[int]]) : mapping from user id (int) to list containing panultimate item id.
        user_test  (defaultdict[int, list[int]]) : mapping from user id (int) to list containing the last item id.
        usernum    (int)                         : number of users (assumes starting from 1, without omission).
        itemnum    (int)                         : number of items (assumes starting from 1, without omission).
    """
    usernum = 0
    itemnum = 0

    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    # assume user/item index starting from 1
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return user_train, user_valid, user_test, usernum, itemnum