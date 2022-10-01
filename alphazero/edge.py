from dataclasses import dataclass
from typing import TypeVar

Node = TypeVar('Node')


@dataclass
class Edge:
    prior_prob: float
    visit_count: int = 0
    total_value: float = 0.0
    node: Node = None

    def get_prior(self):
        return self.prior_prob

    def increment_visit_count(self):
        self.visit_count += 1

    def get_visit_count(self):
        return self.visit_count

    def add_action_val(self, val: float):
        self.total_value += val

    def get_mean_action_val(self):
        return self.total_value/self.visit_count if self.visit_count > 0 else 0.0

    def set_node(self, node: Node):
        self.node = node

    def get_node(self):
        return self.node
