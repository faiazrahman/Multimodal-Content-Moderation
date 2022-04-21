import itertools
import functools
import operator
import time

if __name__ == "__main__":
    print("Test!")

    list2d = [["hello", "there"], ["my"], ["name", "is", "faiaz"]]
    s1 = time.time()
    print(list(itertools.chain.from_iterable(list2d)))
    t1 = time.time()
    print(t1 - s1)

    s2 = time.time()
    print(list(functools.reduce(operator.iconcat, list2d, [])))
    t2 = time.time()
    print(t2 - s2)

    s3 = time.time()
    print([item for sublist in list2d for item in sublist])
    t3 = time.time()
    print(t3 - s3)
