### Syntax for *rat*

We are going to follow stan's [grammar](https://mc-stan.org/docs/2_18/reference-manual/bnf-grammars.html), but alot of stuff stripped out 

#### Pipeline

1. Scanner
    - Input: a string
    - Output: a list of Tokens
2. Parser
    - Input: a list of Tokens
    - Output: an expression tree
3. Compiler
    - Input: a list of expression trees
    - Output: a python function representing the model
#### Tokens
1. `Identifier` Any variable/function/distribution/data
2. `RealLiteral` Real constant
3. `IntLiteral` Integer constant
4. `Operator`  Any operator
5. `Terminate`  `;` end of statement
6. `Special` Special semantic characters `["(", ")", ",", "[", "]", "~"]`
    - sequencing control and order of operations `"(", ")"`
    - expression lists(called `expressions`) `","`
    - subscripts `"[", "]"`
    - sampling `"~"`

##### Note on `<` and `>`
 You can see that angled brackets can mean `greater than` or `less than` when interpreted as operators,
but a complete set of them(`<lower=...>`) also represent the constraint region when in the left hand side of a
statement. In the scanner, which does not take into account semantics, they are just classified as an `Operator`
token. 

This means the parser will fail when evaluating constraints in a standard ll(k) approach because it will attempt 
to try and evaluate a non-existant rhs expression for the 'fake' `>` operator. Therefore, the one and only artificial modification 
I have made to the parser when evaluating constraints is to identify the angled brackets, and convert then from 
`Operator` to `Special` tokens, which makes the parser recognize them as delimiters instead of operators.   

#### Parser and Compiler

 Rat provides syntax to write (almost) all models given you provide data in which each row equates to a single instance of
an observation. Typically R data and pandas dataframes are represented in this format.
 Rat reduces loops into a subscript notation, which relieves the user from writing loops themselves or worrying about
 array broadcasting going wrong.

Rat models are defined in the context of the data dataframe. That is, without specifying the data, Rat cannot identify
parameters from the data. In short: dataframe in, dataframe out. Here's an example by Ben, who came up with the syntax:

Initially the user gives the following data dataframe:

```
   home_team away_team  score_diff  year
0        CLE       NYK          29  2016
1        POR       UTA           9  2016
2        GSW       SAS         -29  2016
```

and writes a Rat model: 

```
score_diff ~ normal(skills[home_team, year] - skills[away_team, year], sigma);
skills[team, year] ~ normal(skills_mu[year], tau);
tau ~ normal(0.0, 1.0);
sigma ~ normal(0.0, 10.0);
```
You can see that `skills, tau, sigma` are not defined in the dataframe and hence are parameters Rat will infer.

Note the brackets here, which at a first glance seems to be indexing arrays. However in Rat, they are not exactly indexing in the traditional sense.
Instead, they define along which factors should a parameter be created(think lme4/brms syntax). For example in the case of `skills[team, year]`,
this defines a 'skills' parameter for each team and year. For `tau` and `sigma`, we can see them defined as a single scalar,
denoting a pooled standard deviation parameter. If this gives you an easier way to see how the model works, Rat works *along each row*

#### BNF

```
expression ::= expression infixOps expression
             | prefixOp expression
             | identifier '[' expressions ']'
             | real_literal
             | integer_literal
             | '(' expression ')'
             | unaryFunction '(' expression ')'


; multiple arguments for standard functions
expressions ::= expression % ','

; subscript formats, which define dimensions and indexing of parameters/data
subscript_expressions ::= identifier % ','
                        | shift(identifier, integer) % ','

; binary operations, just arithmetic and logical
infixOps ::= ("+", "-", "*", "/", "%",
              "||", "&&", "==", "!=", "<", "<=", ">", ">=")

prefixOps ::= ("!", "-")

unaryFunction ::= ("exp", "abs", "floor", "ceil", "round")

; didn't implement control flow yet
statement ::= atomic_statement

; constraints
constraint ::= ?('<' constraint_range '>')

constraint_range ::= 'lower' '=' expressioni ',' 'upper' = expression
                   | 'lower' '=' expression
                   | 'upper' '=' expression

Distribution ::= ("normal")

lhs ::= identifier constraint ('[' expressions ']')*
assignmentOp ::= ("=", "+=", "-=", "*=", "/=")

; statements, assume assignments are allowed
atomic_statement ::=  lhs assignmentOp expression ';'
                   | expression '~' Distribution '(' expressions ')' ';'
```