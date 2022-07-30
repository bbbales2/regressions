from typing import List, Set, Dict, Sequence
from dataclasses import dataclass, field


from .. import ast
from ..compiler2 import StatementInfo
from tatsu.model import NodeWalker



@dataclass
class POSetNode:
    value: Set[str]
    parents: Set["POSetNode"] = field(init=False, default_factory=list)
    children: Set["POSetNode"] = field(init=False, default_factory=list)


@dataclass
class SubscriptPOSet:
    nodes: Dict[Set[str], POSetNode] = field(init=False, default_factory=dict)

    def is_parent(self, parent_node: POSetNode, child_node: POSetNode) -> bool:
        """
        Checks if parent_node is a parent of child_node
        """
        def recurse(node: POSetNode):
            if node.value == parent_node.value:
                return True
            if node.parents:
                for parent in node.parents:
                    if recurse(parent):
                        return True

            return False

        return recurse(child_node)

    def get_highest_order(self) -> List[POSetNode]:
        toplevel_nodes = []
        for _, item in self.nodes.items():
            if not item.parents:
                toplevel_nodes.append(item)

        return toplevel_nodes

    def get_node(self, value: Set[str]):
        return self.nodes[value]

    def insert(self, value: Set[str], parents: Sequence[POSetNode]):
        node = POSetNode(value)
        node.parents = list(parents)
        for parent in parents:
            parent.children.add(node)

        if value in self.nodes:
            raise Exception(f"Key {value} already on the POSet!")
        self.nodes[value] = node


class SubscriptPOSCreator:
    def __init__(self, statement_infos: List[StatementInfo]):
        self.statement_infos = statement_infos
        self.poset = SubscriptPOSet()

    def build(self):
        class SubscriptIdentifier(NodeWalker):
            pass

