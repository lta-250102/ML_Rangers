from api.request import Request
from api.response import Response

class CacheFinder:
    def __init__(self):
        self.tree = {}

    def find_appeared_lst(self, request: Request) -> tuple[list[list], list[bool]]:
        keys = request.rows
        appeared_lst = []
        for key in keys:
            appeared_lst.append(self.get_cache(key))
        return appeared_lst, [x is not None for x in appeared_lst]
    
    def save_cache(self, keys: list[list], values: list[int]):
        for key, value in zip(keys, values):
            self.add_cache(key, value)

    def add_cache(self, key: list, value: int):
        cur_tree = self.tree
        for k in key[:-1]:
            if k not in cur_tree:
                cur_tree[k] = {}
            cur_tree = cur_tree[k]

        if key[-1] not in cur_tree:
            cur_tree[key[-1]] = value

    def get_cache(self, key: list) -> int:
        cur_tree = self.tree
        for k in key[:-1]:
            if k not in cur_tree:
                return None
            cur_tree = cur_tree[k]

        if key[-1] not in cur_tree:
            return None

        return cur_tree[key[-1]]
