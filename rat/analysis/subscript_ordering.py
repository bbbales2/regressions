from typing import List, Set, Dict, Sequence, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .. import ast
from ..compiler2 import StatementInfo
from ..walker import RatWalker
from tatsu.model import NodeWalker
import functools


@dataclass
class POSetNode:
    subscript: Tuple[str]
    variables_used: Set[str]
    parents: Set["POSetNode"] = field(init=False, default_factory=set)
    children: Set["POSetNode"] = field(init=False, default_factory=set)

    def __eq__(self, other):
        if isinstance(other, POSetNode):
            return self.subscript == other.subscript
        return False

    def __hash__(self):
        return hash(self.subscript)

@dataclass
class SubscriptPOSet:
    nodes: Dict[Tuple[str], POSetNode] = field(init=False, default_factory=dict)
    variable_info: Dict[str, Set[Tuple]] = field(init=False, default_factory=lambda: defaultdict(set))

    def is_parent(self, parent_node: POSetNode, child_node: POSetNode) -> bool:
        """
        Checks if parent_node is a parent of child_node
        """

        def recurse(node: POSetNode):
            if node.subscript == parent_node.subscript:
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

    def get_node(self, value: Tuple[str]):
        return self.nodes[value]

    def insert(self, subscript: Tuple[str], variable_name: str, statement_highest_subscript):
        print(variable_name, subscript)
        if subscript in self.nodes:
            self.nodes[subscript].variables_used.add(variable_name)
            return

        if variable_name in self.variable_info:
            assert all([len(x) == len(subscript) for x in self.variable_info[variable_name]]), f"Subscript count must be consistent for variables! {variable_name} has inconsistent subscript length."
            self.create_node(subscript, {variable_name}, [self.nodes[s] for s in self.variable_info[variable_name]])
            return

        toplevel_nodes = self.get_highest_order()
        if subscript != statement_highest_subscript and statement_highest_subscript:
            toplevel_nodes.append(self.nodes[statement_highest_subscript])

        if not toplevel_nodes:
            self.create_node(subscript, {variable_name}, [])
            return

        def recurse(current_node: POSetNode) -> bool:
            if current_node.subscript == subscript:
                current_node.variables_used.add(variable_name)
                return True
            elif set(subscript).issubset(set(current_node.subscript)):
                if any([recurse(x) for x in current_node.children]):
                    return True
                else:
                    self.create_node(subscript, {variable_name}, [current_node])
                    return True
            elif variable_name in self.variable_info:
                if current_node.subscript in self.variable_info[variable_name] and len(current_node.subscript) == len(subscript):
                    self.create_node(subscript, {variable_name}, [current_node])
                    return True
                else:
                    return False
            else:
                return False

        created = False
        for node in toplevel_nodes:
            created = created & recurse(node)


    def add_variable_info(self, subscript: Tuple[str], variable_name: str):
        self.variable_info[variable_name].add(subscript)

    def create_node(self, subscript: Tuple[str], variable_names: Set[str], parents: Sequence[POSetNode]):
        node = POSetNode(subscript, variable_names)
        node.parents = list(parents)
        for parent in parents:
            parent.children.add(node)

        if subscript in self.nodes:
            raise Exception(f"Key {subscript} already on the POSet!")
        self.nodes[subscript] = node
        return node

    def print(self):
        def recurse(node: POSetNode, indentlevel=0):
            print(indentlevel * "    ", node.subscript, id(node))
            for child in node.children:
                recurse(child, indentlevel + 1)

        for node in self.get_highest_order():
            print("-" * 10)
            recurse(node)


class SubscriptPOSCreator:
    def __init__(self, statement_infos: List[StatementInfo]):
        self.statement_infos = statement_infos
        self.poset = SubscriptPOSet()

    def build(self):
        def combine(names):
            return functools.reduce(lambda x, y: x + y, filter(None, names))

        class GetSubscriptWalker(RatWalker):
            def walk_Logical(self, node: ast.Logical):
                return combine([self.walk(node.left), self.walk(node.right)])

            def walk_Binary(self, node: ast.Binary):
                return combine([self.walk(node.left), self.walk(node.right)])

            def walk_FunctionCall(self, node: ast.FunctionCall):
                return combine([self.walk(arg) for arg in node.arglist])

            def walk_Variable(self, node: ast.Variable):
                return (node.name,)

        @dataclass
        class GetVariableWalker(RatWalker):
            subscript_info: List[Tuple[str, Tuple[str]]] = field(default_factory=list)
            def walk_Variable(self, node: ast.Variable):
                name = node.name
                if node.arglist:
                    subscripts = combine([GetSubscriptWalker().walk(subscript) for subscript in node.arglist])
                    self.subscript_info.append((name, subscripts))


        for statement_info in self.statement_infos:
            subscript_walker = GetVariableWalker()
            subscript_walker.walk_Statement(statement_info.statement)
            subscript_walker.subscript_info = sorted(subscript_walker.subscript_info, key=lambda x: len(x[1]), reverse=True)

            # Check that a highest order subscript exists for each statement.
            highest_subscript = None
            if(len(subscript_walker.subscript_info) > 1):
                highest_subscript = subscript_walker.subscript_info[0]
                for name, subscript in subscript_walker.subscript_info[1:]:
                    if name == highest_subscript[0]: continue
                    assert highest_subscript[1] != len(subscript), "Each statement much have a subscript with the highest order between other subscripts in the statement"
                highest_subscript = highest_subscript[1]

            for variable_name, subscript in subscript_walker.subscript_info:
                self.poset.insert(subscript, variable_name, highest_subscript)

            for variable_name, subscript in subscript_walker.subscript_info:
                self.poset.add_variable_info(subscript, variable_name)


