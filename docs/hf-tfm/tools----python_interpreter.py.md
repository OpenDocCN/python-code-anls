# `.\tools\python_interpreter.py`

```
    """
    Evaluate an abstract syntax tree (AST) node representing a Python expression, using variables from `state` and 
    restricted to functions in `tools`.

    Args:
        expression (`ast.AST`):
            The AST node to evaluate.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to their current values.
        tools (`Dict[str, Callable]`):
            Allowed functions that can be called during evaluation.

    Returns:
        Any:
            The result of evaluating the expression represented by `expression`.

    Raises:
        InterpretorError:
            If evaluation encounters an unsupported operation or other error.
    """
    try:
        # Parse the provided AST node
        line_result = ast.literal_eval(expression, globals=state, locals=tools)
    except (ValueError, TypeError, SyntaxError) as e:
        # Capture and raise an InterpretorError for unsupported operations or syntax errors
        raise InterpretorError(f"Failed to evaluate expression: {e}")
    return line_result
    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abstract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignments.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    if isinstance(expression, ast.Assign):
        # If the expression is an assignment statement
        # Evaluate the assignment and return the assigned variable's value
        return evaluate_assign(expression, state, tools)
    elif isinstance(expression, ast.Call):
        # If the expression is a function call
        # Evaluate the function call and return its value
        return evaluate_call(expression, state, tools)
    elif isinstance(expression, ast.Constant):
        # If the expression is a constant value (literal)
        # Return the constant's value
        return expression.value
    elif isinstance(expression, ast.Dict):
        # If the expression is a dictionary literal
        # Evaluate all keys and values recursively and return a dictionary
        keys = [evaluate_ast(k, state, tools) for k in expression.keys]
        values = [evaluate_ast(v, state, tools) for v in expression.values]
        return dict(zip(keys, values))
    elif isinstance(expression, ast.Expr):
        # If the expression is an expression statement
        # Evaluate the expression and return its value
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.For):
        # If the expression is a for loop
        # Evaluate the loop and return its result
        return evaluate_for(expression, state, tools)
    elif isinstance(expression, ast.FormattedValue):
        # If the expression is a formatted value in an f-string
        # Evaluate the content and return its value
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.If):
        # If the expression is an if statement
        # Evaluate the condition and execute the appropriate branch
        return evaluate_if(expression, state, tools)
    elif hasattr(ast, "Index") and isinstance(expression, ast.Index):
        # If the expression is an index operation
        # Evaluate the indexed value and return it
        return evaluate_ast(expression.value, state, tools)
    elif isinstance(expression, ast.JoinedStr):
        # If the expression is a joined string (part of an f-string)
        # Evaluate the concatenated parts and return the resulting string
        return "".join([str(evaluate_ast(v, state, tools)) for v in expression.values])
    elif isinstance(expression, ast.List):
        # If the expression is a list literal
        # Evaluate all elements recursively and return a list
        return [evaluate_ast(elt, state, tools) for elt in expression.elts]
    elif isinstance(expression, ast.Name):
        # If the expression is a variable name
        # Retrieve its value from the state dictionary
        return evaluate_name(expression, state, tools)
    elif isinstance(expression, ast.Subscript):
        # If the expression is a subscript operation
        # Evaluate the subscripted value and return it
        return evaluate_subscript(expression, state, tools)
    else:
        # If the expression type is not recognized
        # Raise an interpreter error indicating the unsupported expression type
        raise InterpretorError(f"{expression.__class__.__name__} is not supported.")
# 对赋值表达式进行求值，更新状态并返回结果
def evaluate_assign(assign, state, tools):
    # 获取赋值表达式左侧的变量名列表
    var_names = assign.targets
    # 调用 evaluate_ast 函数求解赋值表达式右侧的值
    result = evaluate_ast(assign.value, state, tools)

    # 如果只有一个变量名，则直接将结果赋给状态中的对应变量
    if len(var_names) == 1:
        state[var_names[0].id] = result
    else:
        # 否则，检查结果的长度是否与变量名列表相符
        if len(result) != len(var_names):
            raise InterpretorError(f"Expected {len(var_names)} values but got {len(result)}.")
        # 遍历变量名列表和结果，逐个更新状态中的变量值
        for var_name, r in zip(var_names, result):
            state[var_name.id] = r
    # 返回结果
    return result


# 对函数调用表达式进行求值，返回调用结果
def evaluate_call(call, state, tools):
    # 如果调用的函数不是一个简单的名称，抛出错误
    if not isinstance(call.func, ast.Name):
        raise InterpretorError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func} of "
            f"type {type(call.func)}."
        )
    # 获取函数名
    func_name = call.func.id
    # 如果函数名不在提供的工具集中，抛出错误
    if func_name not in tools:
        raise InterpretorError(
            f"It is not permitted to evaluate other functions than the provided tools (tried to execute {call.func.id})."
        )

    # 获取函数对象
    func = tools[func_name]
    # 处理函数调用的参数
    args = [evaluate_ast(arg, state, tools) for arg in call.args]
    kwargs = {keyword.arg: evaluate_ast(keyword.value, state, tools) for keyword in call.keywords}
    # 调用函数并返回结果
    return func(*args, **kwargs)


# 对下标表达式进行求值，返回索引后的值
def evaluate_subscript(subscript, state, tools):
    # 求解下标和值
    index = evaluate_ast(subscript.slice, state, tools)
    value = evaluate_ast(subscript.value, state, tools)

    # 如果值是列表或元组，则返回索引对应的值
    if isinstance(value, (list, tuple)):
        return value[int(index)]
    # 如果索引存在于值中，则返回相应的值
    if index in value:
        return value[index]
    # 如果索引是字符串且值是映射类型，则找出最接近的键并返回其对应的值
    if isinstance(index, str) and isinstance(value, Mapping):
        close_matches = difflib.get_close_matches(index, list(value.keys()))
        if len(close_matches) > 0:
            return value[close_matches[0]]

    # 抛出错误，表示无法进行索引操作
    raise InterpretorError(f"Could not index {value} with '{index}'.")


# 对名称表达式进行求值，返回变量的值
def evaluate_name(name, state, tools):
    # 如果变量名存在于状态中，则返回其对应的值
    if name.id in state:
        return state[name.id]
    # 否则，查找变量名的最接近匹配，并返回对应的值
    close_matches = difflib.get_close_matches(name.id, list(state.keys()))
    if len(close_matches) > 0:
        return state[close_matches[0]]
    # 抛出错误，表示变量未定义
    raise InterpretorError(f"The variable `{name.id}` is not defined.")


# 对条件表达式进行求值，返回布尔值表示的条件结果
def evaluate_condition(condition, state, tools):
    # 如果条件包含多个操作符，抛出错误
    if len(condition.ops) > 1:
        raise InterpretorError("Cannot evaluate conditions with multiple operators")

    # 求解条件左侧和右侧的值
    left = evaluate_ast(condition.left, state, tools)
    comparator = condition.ops[0]
    right = evaluate_ast(condition.comparators[0], state, tools)

    # 根据比较符的类型，比较左右两侧的值并返回结果
    if isinstance(comparator, ast.Eq):
        return left == right
    elif isinstance(comparator, ast.NotEq):
        return left != right
    elif isinstance(comparator, ast.Lt):
        return left < right
    elif isinstance(comparator, ast.LtE):
        return left <= right
    elif isinstance(comparator, ast.Gt):
        return left > right
    elif isinstance(comparator, ast.GtE):
        return left >= right
    elif isinstance(comparator, ast.Is):
        return left is right
    elif isinstance(comparator, ast.IsNot):
        return left is not right
    # 如果比较符号是 'in'，则返回左操作数是否包含在右操作数中的布尔值
    elif isinstance(comparator, ast.In):
        return left in right
    # 如果比较符号是 'not in'，则返回左操作数是否不包含在右操作数中的布尔值
    elif isinstance(comparator, ast.NotIn):
        return left not in right
    else:
        # 如果比较符号不是以上两种情况，抛出解释器错误，显示不支持的操作符信息
        raise InterpretorError(f"Operator not supported: {comparator}")
# 根据条件语句评估条件并执行相应的操作，返回最后一个操作的结果
def evaluate_if(if_statement, state, tools):
    result = None
    # 如果条件为真，执行条件体内的语句
    if evaluate_condition(if_statement.test, state, tools):
        # 遍历条件体内的每一行语句
        for line in if_statement.body:
            # 评估并执行当前行的抽象语法树节点
            line_result = evaluate_ast(line, state, tools)
            # 如果结果不为空，更新结果
            if line_result is not None:
                result = line_result
    else:
        # 如果条件为假，执行否定体内的语句
        for line in if_statement.orelse:
            # 评估并执行当前行的抽象语法树节点
            line_result = evaluate_ast(line, state, tools)
            # 如果结果不为空，更新结果
            if line_result is not None:
                result = line_result
    # 返回最后执行的结果
    return result


# 根据for循环语句评估迭代器，并依次执行循环体内的操作，返回最后一个操作的结果
def evaluate_for(for_loop, state, tools):
    result = None
    # 评估迭代器表达式，获取迭代器对象
    iterator = evaluate_ast(for_loop.iter, state, tools)
    # 遍历迭代器对象中的每一个元素
    for counter in iterator:
        # 将当前元素赋值给循环目标变量
        state[for_loop.target.id] = counter
        # 遍历for循环体内的每一个表达式
        for expression in for_loop.body:
            # 评估并执行当前表达式的抽象语法树节点
            line_result = evaluate_ast(expression, state, tools)
            # 如果结果不为空，更新结果
            if line_result is not None:
                result = line_result
    # 返回最后执行的结果
    return result
```