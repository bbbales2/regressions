class Expr:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def code(self):
        pass

    def __str__(self):
        pass

class Distr(Expr):
    pass


class Normal(Distr):
    def __init__(self, lhs, mean, std):
        self.lhs = lhs
        self.mean = mean
        self.std = std

    def __iter__(self):
        return iter([self.lhs, self.mean, self.std])
    
    def code(self):
        return f"{self.lhs.code()} ~ normal({self.mean.code()}, {self.std.code()})"

    def __str__(self):
        return f"Normal({self.lhs.__str__()}, {self.mean.__str__()}, {self.std.__str__()})"

class RealConstant(Expr):
    def __init__(self, value):
        self.value = value

    def code(self):
        return f"{self.value}"

    def __str__(self):
        return f"RealConstant({self.value})"

class IntegerConstant(Expr):
    def __init__(self, value):
        self.value = value

    def code(self):
        return f"{self.value}"

    def __str__(self):
        return f"IntegerConstant({self.value})"

class Diff(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])
    
    def code(self):
        return f"{self.left.code()} - {self.right.code()}"

    def __str__(self):
        return f"Diff({self.left.__str__()}, {self.right.__str__()})"

class Sum(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} + {self.right.code()}"

    def __str__(self):
        return f"Sum({self.left.__str__()}, {self.right.__str__()})"

class Mul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} * {self.right.code()}"

    def __str__(self):
        return f"Mul({self.left.__str__()}, {self.right.__str__()})"

class Div(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} / {self.right.code()}"

    def __str__(self):
        return f"Div({self.left.__str__(), self.right.__str__()})"


class Mod(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} $ {self.right.code()}"

    def __str__(self):
        return f"Mod({self.left.__str__(), self.right.__str__()})"

class LogicalOR(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} || {self.right.code()}"

    def __str__(self):
        return f"LogicalOR({self.left.__str__(), self.right.__str__()})"

class LogicalAND(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} && {self.right.code()}"

    def __str__(self):
        return f"LogicalAND({self.left.__str__(), self.right.__str__()})"

class Equality(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} == {self.right.code()}"

    def __str__(self):
        return f"Equality({self.left.__str__(), self.right.__str__()})"

class Inequality(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} != {self.right.code()}"

    def __str__(self):
        return f"Inequality({self.left.__str__(), self.right.__str__()})"

class LessThan(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} < {self.right.code()}"

    def __str__(self):
        return f"LessThan({self.left.__str__(), self.right.__str__()})"

class LessThanOrEqual(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} <= {self.right.code()}"

    def __str__(self):
        return f"LessThanOrEqual({self.left.__str__(), self.right.__str__()})"

class GreaterThan(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} > {self.right.code()}"

    def __str__(self):
        return f"GreaterThan({self.left.__str__(), self.right.__str__()})"

class GreaterThanOrEqual(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])

    def code(self):
        return f"{self.left.code()} >= {self.right.code()}"

    def __str__(self):
        return f"GreaterThanOrEqual({self.left.__str__(), self.right.__str__()})"

class PrefixNegation(Expr):
    def __init__(self, subexpr):
        self.subexpr = subexpr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"-{self.subexpr.code()}"

    def __str__(self):
        return f"PrefixNegation({self.subexpr.__str__()})"

class PrefixLogicalNegation(Expr):
    def __init__(self, subexpr):
        self.subexpr = subexpr

    def __iter__(self):
        return iter([self.subexpr])

    def code(self):
        return f"!{self.subexpr.code()}"

    def __str__(self):
        return f"PrefixLogicalNegation({self.subexpr.__str__()})"

class Assignment(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} = {self.rhs.code()}"

    def __str__(self):
        return f"Assignment({self.lhs.__str__()}, {self.rhs.__str__()})"

class AddAssignment(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} += {self.rhs.code()}"

    def __str__(self):
        return f"AddAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"

class DiffAssignment(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} -= {self.rhs.code()}"

    def __str__(self):
        return f"DiffAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"

class MulAssignment(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} *= {self.rhs.code()}"

    def __str__(self):
        return f"MulAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"

class DivAssignment(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __iter__(self):
        return iter([self.lhs, self.rhs])

    def code(self):
        return f"{self.lhs.code()} /= {self.rhs.code()}"

    def __str__(self):
        return f"DivAssignment({self.lhs.__str__()}, {self.rhs.__str__()})"

class Data(Expr):
    def __init__(self, name):
        self.name = name
    
    def get_key(self):
        return self.name
    
    def populate(self, stan_data):
        self.stan_data = stan_data
    
    def code(self):
        return self.stan_data.get_stan_name()

class Placeholder(Expr):
    def __init__(self, name, index, value=None):
        self.name = name
        self.index = index
        self.value = value

    def __iter__(self):
        if self.index is not None:
            return iter([self.index])
        else:
            return iter([])

    def code(self):
        if self.index is not None:
            return f"{self.name}[{self.index.code()}]"
        else:
            return self.name

    def __str__(self):
        return f"Placeholder({self.name}, {self.index.__str__()})"
        #return f"Placeholder({self.name}, {self.index.__str__()}) = {{{self.value.__str__()}}}"

class Param(Expr):
    def __init__(self, name, index = None, lower = None, upper = None):
        self.name = name
        self.index = index
        self.lower = lower
        self.upper = upper

    def __iter__(self):
        if self.index is not None:
            return iter([self.index])
        else:
            return iter([])

    def get_key(self):
        return self.name

    def populate(self, stan_param):
        self.stan_param = stan_param
    
    def code(self):
        stan_name = self.stan_param.get_stan_name()
        if self.index is not None:
            return f"{stan_name}[{self.index.code()}]"
        else:
            return stan_name

class Index(Expr):
    def __init__(self, *args):
        self.args = args
    
    def get_key(self):
        return self.args

    def populate(self, stan_index):
        self.stan_index = stan_index
    
    def code(self):
        #return self.stan_index.get_stan_name()
        return ", ".join(x.code() for x in self.args)
    
    def __str__(self):
        return f"Index({', '.join(x.__str__() for x in self.args)})"

def find_type(type, expr):
    if isinstance(expr, type):
        return [expr]
    else:
        indices = []
        for child in expr:
            indices.extend(find_type(type, child))

        return indices

def populate_type(type, expr, stan_vars):
    if isinstance(expr, type):
        return expr.populate(stan_vars[expr.get_key()])
    else:
        for child in expr:
            populate_type(type, child, stan_vars)

def search_tree(type, expr):
    if isinstance(expr, type):
        yield expr
    else:
        for child in expr:
            for child_expr in search_tree(type, child):
                yield child_expr
