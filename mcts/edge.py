from dataclasses import dataclass
from typing import TypeVar

TreeNode = TypeVar('TreeNode')


@dataclass
class Edge:
    visit_count: int = 0
    total_value: int = 0
    node: TreeNode = None

    def increment_visit_count(self):
        self.visit_count += 1

    def get_visit_count(self):
        return self.visit_count

    def add_val(self, val: float):
        self.total_value += val

    def get_avg(self):
        assert self.visit_count > 0, f'This edge has not been simulated before.'
        return self.total_value/self.visit_count

    def set_node(self, node: TreeNode):
        self.node = node

    def get_node(self):
        return self.node
