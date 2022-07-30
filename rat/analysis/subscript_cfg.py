from typing import Set, List, Union, Dict, Tuple
from dataclasses import dataclass, field
from .. import ast
from ..compiler2 import StatementInfo, RatCompiler
from ..walker import NodeWalker, RatWalker

# The following imports are only used for debugging visualization
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.lines import Line2D
from collections import defaultdict


@dataclass
class BaseSCFGNode:
    input_queue: List[Tuple] = field(init=False, default_factory=set)
    output_nodes: List["BaseSCFGNode"] = field(init=False, default_factory=list)

    def add_queue(self, subscript: Tuple):
        """
        Add a subscript to the processing queue.
        """
        self.input_queue.append(subscript)

    def visit(self, subscript_names: Tuple[str]) -> bool:
        """
        Process values within the input queue.
        If the return value is true, the output nodes of the current node is added to the worknode.
        """
        raise NotImplementedError


@dataclass
class SCFGConditional(BaseSCFGNode):
    predicate_code: str
    true_nodes: List["BaseSCFGNode"] = field(init=False, default_factory=list)
    false_nodes: List["BaseSCFGNode"] = field(init=False, default_factory=list)

    def visit(self, subscript_names: Tuple[str]):
        while self.input_queue:
            subscript = self.input_queue.pop(0)
            predicate_result = eval(self.predicate_code, {}, {name: value for name, value in zip(subscript_names, subscript)})
            if predicate_result:
                for true_node in self.true_nodes:
                    true_node.input_queue.append(subscript)
            else:
                for false_node in self.false_nodes:
                    false_node.input_queue.append(subscript)


@dataclass
class SCFGTransferFunc(BaseSCFGNode):
    """
    The Transfer function is the "edge" of the control flow graph. Every variable(node) has a set of subscript
    values. When it invokes another node, it may pass on a transformed version of the subscript values. for example,
    when you're doing time series recursion, you would input x and will pass x - 1 to the next node. Transfer functions
    also fulfill the task, by passing on transformed subscript values.
    """

    transform_expr_strings: List[str]
    subscript_names: List[str]

    def visit(self, subscript_names: Tuple[str]):
        n_transform_codes = len(self.transform_expr_strings)
        assert len(subscript_names) == n_transform_codes, "Number of subscripts passed to visit must equal number of transfer functions!"
        transformed_subscripts = []
        while self.input_queue:
            pre_transform_subscript = self.input_queue.pop(0)
            assert (
                len(pre_transform_subscript) == n_transform_codes
            ), "Number of transfer expressions and input subscript length must match!"
            post_transform_subscript = tuple(
                map(
                    lambda expr: eval(expr, {}, {name: value for name, value in zip(subscript_names, pre_transform_subscript)}),
                    self.transform_expr_strings,
                )
            )
            transformed_subscripts.append(post_transform_subscript)

        for next_node in self.output_nodes:
            next_node.input_queue.extend(transformed_subscripts)


@dataclass
class SCFGVariable(BaseSCFGNode):
    name: str
    domain: Set[Tuple] = field(init=False, default_factory=set)

    def visit(self, subscript_names: Tuple[str]):
        prejoin_domain_size = len(self.domain)
        self.domain |= self.input_queue
        postjoin_domain_size = len(self.domain)
        return True if prejoin_domain_size == postjoin_domain_size else False


class SCFGExecutor:
    def __init__(self):
        pass

    def kildall_iterate(self, seed_variables):
        """
        Run the kildall iterative fixed-point algorithm.
        """


class SubscriptExpressionWalker(NodeWalker):
    def walk_Logical(self, node: ast.Logical):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_Binary(self, node: ast.Binary):
        return f"({self.walk(node.left)} {node.op} {self.walk(node.right)})"

    def walk_IfElse(self, node: ast.IfElse):
        return f"({self.walk(node.left)} if {self.walk(node.predicate)} else {self.walk(node.right)})"

    def walk_FunctionCall(self, node: ast.FunctionCall):
        arglist = []

        if node.arglist:
            arglist += [self.walk(arg) for arg in node.arglist]

        return f"{node.name}({','.join(arglist)})"

    def walk_Variable(self, node: ast.Variable):
        if node.arglist:
            raise Exception("Variables within subscript expression cannot be subscripted themselves!")
        return node.name

    def walk_Literal(self, node: ast.Literal):
        return f"{node.value}"


@dataclass
class SCFGBuilderWalker(NodeWalker):
    primary_variable_ast: ast.Variable
    scfg_variables: Dict[str, SCFGVariable]

    def walk_Statement(self, node: ast.Statement):
        return self.walk(node.left) + self.walk(node.right)

    def walk_IfElse(self, node: ast.IfElse):
        predicate = SubscriptExpressionWalker().walk(node.predicate)
        true_nodes = self.walk(node.left)
        false_nodes = self.walk(node.right)
        cond_node = SCFGConditional(predicate)
        cond_node.true_nodes.extend(true_nodes)
        cond_node.false_nodes.extend(false_nodes)
        if self.primary_variable_ast.arglist:
            primary_variable_subscript_names = [SubscriptExpressionWalker().walk(arg) for arg in self.primary_variable_ast.arglist]
            args = [SubscriptExpressionWalker().walk(arg) for arg in self.primary_variable_ast.arglist]
            transfer_node = SCFGTransferFunc(args, primary_variable_subscript_names)
            transfer_node.output_nodes.append(cond_node)
            return [transfer_node]
        assert not true_nodes and not false_nodes, "If primary variable doesn't have subscripts, other variables may not have subscripts!!"
        return []

    def walk_Variable(self, node: ast.Variable):
        if node.name == self.primary_variable_ast.name and node.prime:
            return []
        if node.arglist:
            args = [SubscriptExpressionWalker().walk(arg) for arg in node.arglist]
            variable_name = node.name
            if self.primary_variable_ast.arglist:
                # TODO: this won't work when data is the primary variable, since data variables naturally won't be subscripted
                primary_variable_subscript_names = [SubscriptExpressionWalker().walk(arg) for arg in self.primary_variable_ast.arglist]
            else:
                raise Exception(
                    f"varname {node.name} primary: {self.primary_variable_ast.name} If primary variable doesn't have subscripts, other variables may not have subscripts!!"
                )
            transfer_func = SCFGTransferFunc(args, primary_variable_subscript_names)
            if variable_name not in self.scfg_variables:
                self.scfg_variables[variable_name] = SCFGVariable(variable_name)

            transfer_func.output_nodes.append(self.scfg_variables[variable_name])

            return [transfer_func]
        else:
            return []

    def walk_Logical(self, node: ast.Logical):
        return self.walk(node.left) + self.walk(node.right)

    def walk_Binary(self, node: ast.Binary):
        return self.walk(node.left) + self.walk(node.right)

    def walk_Literal(self, node: ast.Literal):
        return []

    def walk_FunctionCall(self, node: ast.FunctionCall):
        ret = []
        if node.arglist:
            for arg in node.arglist:
                ret += self.walk(arg)
        return ret


class SCFGBuilder:
    def __init__(self, statement_info: List[StatementInfo]):
        self.statement_info: List[StatementInfo] = statement_info
        self.scfg_variables: Dict[str, SCFGVariable] = {}

    def build(self, data_variable_names):
        builder = SCFGBuilderWalker(primary_variable_ast=None, scfg_variables=self.scfg_variables)
        for statement in self.statement_info:
            primary_variable_name = statement.primary.name
            if primary_variable_name not in self.scfg_variables:
                self.scfg_variables[primary_variable_name] = SCFGVariable(primary_variable_name)
            builder.primary_variable_ast = statement.primary

            self.scfg_variables[primary_variable_name].output_nodes.extend(builder.walk_Statement(statement.statement))


def visualize_recurse(node: BaseSCFGNode, graph: nx.DiGraph, previous_node_name, visited):
    match node:
        case SCFGVariable():
            if id(node) not in visited:
                visited.add(id(node))
                # print("create node", node.name)
                graph.add_node(node.name, name=node.name)
            if node.output_nodes:
                for on in node.output_nodes:
                    visualize_recurse(on, graph, node.name, visited)
            return node.name

        case SCFGTransferFunc():
            name = f"Transfer({node.transform_expr_strings})"
            if id(node) in visited:
                return name
            visited.add(id(node))
            if node.output_nodes:
                for on in node.output_nodes:
                    res = visualize_recurse(on, graph, previous_node_name, visited)
                    if res:
                        print(f"{previous_node_name} -> {res}: {name}")
                        graph.add_edge(previous_node_name, res, label=name)
            return name
        case SCFGConditional():
            name = f"Conditional({node.predicate_code})"
            if id(node) in visited:
                return name
            visited.add(id(node))
            graph.add_node(name, name=name)
            # graph.add_edge(previous_node_name, name)
            # print(f"{previous_node_name} -> {name}")
            if node.true_nodes:
                for tn in node.true_nodes:
                    res = visualize_recurse(tn, graph, name, visited)
                    # if res:
                    #     graph.add_edge(name, res, label="true")

            if node.false_nodes:
                for fn in node.false_nodes:
                    res = visualize_recurse(fn, graph, name, visited)
                    # if res:
                    #    print("false", res)
                    # graph.add_edge(name, res, label="false")
            return name


def visualize(scfg_variables: Dict[str, SCFGVariable], entry_node: str):
    G = nx.MultiDiGraph()
    visited = set()
    visualize_recurse(scfg_variables[entry_node], G, "", visited)

    edge_names = tuple(set(nx.get_edge_attributes(G, "label").values()))

    cm = plt.get_cmap("gist_rainbow")
    cNorm = mcolors.Normalize(vmin=0, vmax=len(edge_names) - 1)
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=cm)
    colors = [scalarMap.to_rgba(i) for i in range(len(edge_names))]
    assert len(colors) == len(edge_names), "Not enough colors to draw all edges differently"
    color_map = {name: color for name, color in zip(edge_names, colors)}

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "name"), font_size=40)

    ax = plt.gca()
    ax.axis("off")
    ax.autoscale_view("tight")
    ax.set_aspect("auto")
    edge_count = defaultdict(lambda: 0)

    for index, e in enumerate(G.edges):
        ax.annotate(
            "",
            xy=pos[e[0]],
            xycoords="data",
            xytext=pos[e[1]],
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<|-",
                mutation_scale=40,
                color=color_map[list(nx.get_edge_attributes(G, "label").values())[index]],
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                connectionstyle="arc3,rad=rrr".replace("rrr", str(0.3 * edge_count[",".join(sorted(e[:2]))])),
            ),
        )
        edge_count[",".join(sorted(e[:2]))] += 1

    legend_elements = [Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_map.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), label_pos=0.5, rotate=True)
    print(G)
