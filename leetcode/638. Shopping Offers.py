from typing import List

from LEETCODE_TEMPLATE import TEST_CASE_ATTRIBUTE_NAME, TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME


FN_NAME = "shoppingOffers"
TEST_INPUTS = [
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "price": [2,5],
            "special": [[3,0,5],[1,2,10]],
            "needs": [3,2],
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 14,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "price": [2,3,4],
            "special": [[1,1,0,4],[2,2,1,9]],
            "needs": [1,2,1],
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 11,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "price": [2,3,4],
            "special": [[1,1,0,4],[2,2,1,9]],
            "needs": [0,0,0],
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 0,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "price": [0,0,0],
            "special": [[1,1,0,4],[2,2,1,9]],
            "needs": [2,2,1],
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 0,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "price": [2,3],
            "special": [[1,0,1],[0,1,2]],
            "needs": [1,1],
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 3,
    },
    {
        TEST_CASE_ATTRIBUTE_NAME: {
            "price": [5,4,2,8,7,3],
            "special": [[3,3,0,5,5,0,3],[3,1,6,3,0,3,2],[0,5,4,4,3,4,22],[0,3,5,0,5,3,3],[5,3,0,6,0,5,26],[3,6,6,4,0,4,8],[5,5,1,6,3,3,23],[0,0,1,5,4,6,21],[0,0,1,3,4,1,4],[2,0,5,5,4,3,18],[4,0,1,3,5,1,17],[0,4,3,5,5,2,18],[0,6,0,5,6,0,23],[3,1,4,5,6,1,24],[1,4,5,6,2,4,17],[0,5,2,5,3,3,16],[6,5,5,6,4,3,26],[3,3,3,5,3,1,16],[0,2,3,2,4,6,23],[6,4,1,4,2,3,11],[3,4,4,2,2,0,21],[4,2,1,5,5,5,23],[6,4,3,0,4,4,24],[1,4,1,0,1,2,5],[1,3,5,5,0,3,8],[0,1,0,2,3,1,8],[0,6,2,6,1,3,28],[5,0,6,6,4,5,21],[3,5,2,3,1,0,2],[6,1,6,4,3,6,28],[4,6,5,4,3,5,12],[5,1,1,3,6,6,23],[2,4,0,5,3,0,23],[2,1,0,2,2,0,12],[3,4,2,1,3,0,11],[5,3,5,1,2,5,19],[1,4,5,3,1,4,22],[4,4,6,5,3,0,15],[6,5,2,0,2,4,7],[4,2,3,2,5,6,18],[0,4,5,0,1,2,2],[3,4,6,5,1,6,9],[3,4,1,2,0,2,8],[2,6,0,4,2,1,19],[5,0,5,4,5,4,26],[6,4,3,0,1,3,19],[0,2,5,3,4,5,18],[4,0,4,4,4,2,10],[4,3,6,1,3,5,20],[1,0,0,6,4,3,12],[3,6,6,3,6,5,6],[1,1,6,5,6,3,23],[1,3,3,6,6,0,6],[3,3,3,2,1,1,7],[3,4,4,6,2,3,23],[6,6,6,1,0,5,26],[6,0,5,6,3,6,27],[1,5,3,2,1,4,12],[3,5,4,6,0,6,19],[2,2,5,2,2,2,4],[0,5,5,5,5,4,5],[0,3,0,1,1,3,18],[1,5,5,2,1,0,7],[1,6,3,4,2,1,5],[2,1,5,2,0,3,15],[3,0,4,3,2,4,24],[1,5,3,0,1,2,24],[2,2,2,4,2,3,14],[2,3,1,4,4,6,26],[4,0,5,1,1,0,3],[1,3,5,0,6,1,23],[6,1,3,1,2,3,4],[4,0,0,4,3,6,23],[5,5,5,3,6,1,3],[5,0,4,4,3,0,12],[0,0,5,6,1,0,7],[1,3,0,5,1,1,22],[4,0,1,1,4,5,24],[1,4,4,4,4,6,16],[5,5,5,4,5,4,11],[4,4,0,6,5,1,10],[0,0,3,2,5,2,25],[0,5,5,1,3,0,25],[5,0,4,2,3,1,14],[6,3,0,3,0,1,22],[3,0,1,3,5,2,21],[3,4,0,2,1,6,9],[2,4,5,0,2,2,1],[5,1,6,4,4,0,21],[4,6,6,6,6,5,17],[4,0,6,5,3,5,1],[4,2,2,6,6,0,26],[2,0,5,0,2,2,25],[6,2,2,2,6,6,28],[3,5,1,5,5,6,5],[6,2,2,3,4,4,8],[3,2,3,5,1,6,5],[3,1,0,0,2,2,8],[1,6,6,5,4,6,11],[1,3,5,0,5,1,17]],
            "needs": [2,5,0,5,6,3],
        },
        TEST_CASE_EXPECTED_VALUE_ATTRIBUTE_NAME: 49,
    },
]

import heapq


class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        if Solution.is_goal(needs) or Solution.is_goal(price):
            return 0

        special.extend(Solution.prices_to_offer(price))
        special.sort(key=lambda offer: offer[-1])
        curr, s = [(0, *needs)], set()  # [price, need_1, ...]
        while True:
            current_need = heapq.heappop(curr)
            sol = Solution.apply_offer(special, current_need, s, curr)
            if sol:
                return sol[0]

    @staticmethod
    def apply_offer(offers, current_need, s, curr):
        for offer in offers:
            next_need = list(current_need)
            next_need[0] += offer[-1]
            for item, num in enumerate(offer[:-1]):
                next_need[item + 1] -= num
            next_need = tuple(next_need)

            if Solution.is_goal(next_need):
                return next_need
            if Solution.is_valid(next_need) and next_need not in s:
                s.add(next_need)
                heapq.heappush(curr, next_need)
        return False

    @staticmethod
    def prices_to_offer(prices):
        offers = []
        for i, price in enumerate(prices):
            offer = [0] * len(prices)
            offer[i] = 1
            offers.append(offer + [price])
        return offers

    @staticmethod
    def is_valid(need):
        return all(map(lambda v: v >= 0, need[1:]))

    @staticmethod
    def is_goal(need):
        return all(map(lambda v: v == 0, need[1:]))

if __name__ == "__main__":
    from LEETCODE_TEMPLATE import test
    test(Solution, TEST_INPUTS, FN_NAME)
