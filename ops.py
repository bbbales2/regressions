class Expr:
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

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
        return f"{self.lhs.code()} ~ normal({self.mean.code()}, {self.std.code()});"

class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def code(self):
        return f"{self.value}"
    
class Diff(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __iter__(self):
        return iter([self.left, self.right])
    
    def code(self):
        return f"{self.left.code()} - {self.right.code()}"

class Data(Expr):
    def __init__(self, name):
        self.name = name
    
    def get_key(self):
        return self.name
    
    def populate(self, stan_data):
        self.stan_data = stan_data
    
    def code(self):
        return self.stan_data.get_stan_name()

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
        return self.stan_index.get_stan_name()
    

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
