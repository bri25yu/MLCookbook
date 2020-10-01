from LEETCODE_TEMPLATE import TEST_CASE_ATTRIBUTE_NAME, TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME


FN_NAME = "minSteps"
TEST_INPUTS = [
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "n": 3,
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 3,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "n": 5,
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 5,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "n": 218,
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 111,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "n": 49,
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 14,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "n": 2,
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 2,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "n": 1,
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 0,
    },
]

from collections import defaultdict


class Solution:
    def minSteps(self, n: int) -> int:
        if n == 1: return 0

        d, s, c = defaultdict(list, {1: [0]}), {(1, 0)}, 0
        while True:
            new_d = defaultdict(list)
            for number in d:
                Solution.check_and_add(n, (number, number), new_d, s)
                for to_add in d[number]:
                    if number + to_add == n:
                        return c + 1
                    Solution.check_and_add(n, (number + to_add, to_add), new_d, s)
            d = new_d
            c += 1

    @staticmethod
    def check_and_add(n, val, d, s):
        if val[0] < n and val not in s:
            s.add(val)
            d[val[0]].append(val[1])


if __name__ == "__main__":
    from LEETCODE_TEMPLATE import test
    test(Solution, TEST_INPUTS, FN_NAME)
