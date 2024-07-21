# `.\pytorch\torch\fx\experimental\shape_inference\infer_symbol_values.py`

```
import re  # 导入正则表达式模块
from typing import Any, DefaultDict, Dict, List, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 数学计算库

import sympy as sp  # 导入 SymPy 符号计算库

import torch  # 导入 PyTorch 深度学习框架

square_brackets_pattern = r"\[([^]]+)\]"  # 匹配方括号中的内容的正则表达式模式
parentheses_pattern = r"\((.*?)\)"  # 匹配圆括号中的内容的正则表达式模式
s_pattern = r"s\d+"  # 匹配以 s 开头的数字的正则表达式模式


def infer_symbol_values(
    symints: List[Union[torch.SymInt, int]],  # 接受包含 torch.SymInt 或 int 的列表作为输入
    init_symints: List[Union[torch.SymInt, int]],  # 接受包含 torch.SymInt 或 int 的列表作为输入
    symbol_idx_dict: Dict[str, int],  # 接受字符串到整数的字典作为输入
    padding_constraints: DefaultDict[torch.SymInt, List[Union[sp.Expr, int]]],  # 接受 torch.SymInt 到包含 sp.Expr 或 int 的列表的默认字典作为输入
    constraint: str,  # 接受字符串作为输入
) -> None:
    if constraint.find("non-singleton") != -1:  # 如果约束字符串中包含 "non-singleton"
        left_expression, right_expression = re.findall(parentheses_pattern, constraint)  # 从约束字符串中提取圆括号中的内容
        calculate_value(left_expression, right_expression, symints, symbol_idx_dict)  # 调用 calculate_value 函数处理左右表达式

    elif constraint.find("first two dimensions of batch2 tensor to be") != -1:  # 如果约束字符串中包含 "first two dimensions of batch2 tensor to be"
        matches = re.findall(square_brackets_pattern, constraint)  # 从约束字符串中提取方括号中的内容
        left_expression, right_expression = (
            matches[i].split(",")[1].strip() for i in (0, 1)  # 分别处理匹配结果中的左右表达式
        )
        calculate_value(left_expression, right_expression, symints, symbol_idx_dict)  # 调用 calculate_value 函数处理左右表达式

    elif constraint.find("a and b must have same reduction dim") != -1:  # 如果约束字符串中包含 "a and b must have same reduction dim"
        matches = re.findall(square_brackets_pattern, constraint)  # 从约束字符串中提取方括号中的内容
        left_expression = matches[0].split(",")[1].strip()  # 处理第一个匹配结果中的左表达式
        right_expression = matches[1].split(",")[0].strip()  # 处理第二个匹配结果中的右表达式
        calculate_value(left_expression, right_expression, symints, symbol_idx_dict)  # 调用 calculate_value 函数处理左右表达式

    elif constraint.find("Split sizes add up to") != -1:  # 如果约束字符串中包含 "Split sizes add up to"
        match_1 = re.search(r"to\s+(.*?)\s+but", constraint)  # 从约束字符串中提取第一个匹配的值
        extracted_value_1 = match_1.group(1) if match_1 else None  # 获取第一个匹配结果的值
        match_2 = re.search(r"of\s+(.*?)$", constraint)  # 从约束字符串中提取第二个匹配的值
        extracted_value_2 = match_2.group(1) if match_2 else None  # 获取第二个匹配结果的值
        calculate_value(extracted_value_1, extracted_value_2, symints, symbol_idx_dict)  # 调用 calculate_value 函数处理提取的两个值
    # 检查约束字符串中是否包含特定文本片段 "is invalid for input of size"
    elif constraint.find("is invalid for input of size") != -1:
        # 使用正则表达式查找方括号中的内容
        matches = re.findall(square_brackets_pattern, constraint)
        # 将方括号中的内容按逗号分隔并存入列表 left_elements
        left_elements = matches[0].split(",")
        # 创建一个 SymPy 符号对象表示左侧方程的初始值为 1
        left_equation = sp.sympify(1)
        # 初始化左侧数字为 1
        left_num = 1
        # 提取约束字符串中 "size" 后面的内容，并去除两端的空白字符，创建右侧方程的 SymPy 表达式
        right_equation = sp.sympify(constraint.split("size")[1].strip())

        # 遍历左侧元素列表
        for left_element in left_elements:
            # 如果左侧元素是 "-1"，则跳过当前循环
            if sp.sympify(left_element) == sp.sympify("-1"):
                continue
            # 如果左侧元素是数值类型
            elif sp.sympify(left_element).is_number:
                # 将左侧数字乘以当前数值元素
                left_num *= int(left_element)
            else:
                # 否则，将左侧方程乘以当前符号表达式
                left_equation *= sp.sympify(left_element)
        
        # 简化右侧方程，除以左侧方程
        right_equation = sp.cancel(right_equation / left_equation)

        # 找出右侧方程中的自由符号变量
        right_vars = list(right_equation.free_symbols)
        # 遍历右侧变量列表
        for right_var in right_vars:
            # 如果右侧变量是 "s0"
            if sp.sympify(right_var) == sp.sympify("s0"):
                # 简化右侧方程除以 "s0"
                right_equation = sp.cancel(right_equation / right_var)
                # 从右侧变量列表中移除 "s0"
                right_vars.remove(right_var)

        # 获取右侧方程中的变量
        var = right_vars[0]
        # 根据变量查找其在 symbol_idx_dict 字典中的索引
        idx = symbol_idx_dict[str(var)]
        # 如果变量不在 padding_constraints 字典中，则将其初始化为空列表
        if var not in padding_constraints:
            padding_constraints[var].append(right_equation)
        # 更新方程
        update_equation(
            symints,
            init_symints,
            padding_constraints,
            padding_constraints[var][0],  # 指定类型为忽略的参数类型
            left_num,
            var,
            idx,
        )
def calculate_value(
    left_expression: Union[str, Any, None],  # 左表达式，可以是字符串、任意类型或None
    right_expression: Union[str, Any, None],  # 右表达式，可以是字符串、任意类型或None
    symints: List[Union[torch.SymInt, int]],  # 符号整数列表，包含torch.SymInt或整数
    symbol_idx_dict: Dict[str, int],  # 符号索引字典，映射变量名到索引位置
) -> None:
    var, val = solve_equation(left_expression, right_expression)  # 解方程得到变量名和值
    idx = symbol_idx_dict[var]  # 获取变量名在符号索引字典中的索引
    pre_equation = sp.sympify(f"{symints[idx]}")  # 使用符号整数列表中的值创建预设方程
    symints[idx] = pre_equation.subs(sp.sympify(var), val)  # 将变量名的值替换进预设方程


def solve_equation(
    left_expression: Union[str, Any, None],  # 左表达式，可以是字符串、任意类型或None
    right_expression: Union[str, Any, None],  # 右表达式，可以是字符串、任意类型或None
) -> Tuple[str, int]:
    expression = f"{left_expression} - {right_expression}"  # 构建表达式
    var = re.findall(s_pattern, expression)[0]  # 从表达式中提取变量名
    if re.findall(parentheses_pattern, expression):  # 如果表达式中包含括号
        sub_expression = re.findall(parentheses_pattern, expression)[0]  # 提取括号表达式
        var, coeff = sub_expression.split("//")  # 拆分出变量名和系数
        x = sp.symbols("x")  # 创建符号变量x
        sub_equation = sp.sympify(f"{var} - {coeff} * {x}")  # 构建子方程
        modified_equation = (
            sp.sympify(x) + sp.sympify(expression) - sp.sympify(sub_expression)
        )  # 构建修改后的方程

        solution = sp.solve((modified_equation, sub_equation), (x, var))  # 解方程组
        return (var, int(solution[sp.sympify(var)]))  # 返回变量名和解的整数值
    else:
        solution = sp.solve(expression, var)  # 解单个方程
        val = int(solution[0])  # 获取解的整数值
        return (var, val)  # 返回变量名和解的整数值


def update_equation(
    symints: List[Union[torch.SymInt, int]],  # 符号整数列表，包含torch.SymInt或整数
    init_symints: List[Union[torch.SymInt, int]],  # 初始符号整数列表，包含torch.SymInt或整数
    padding_constraints: DefaultDict[torch.SymInt, List[Union[sp.Expr, int]]],  # 填充约束字典，映射torch.SymInt到约束列表
    init_eq: sp.Expr,  # 初始方程
    new_mod_num: int,  # 新模数
    var: torch.SymInt,  # 符号整数
    idx: int,  # 索引
) -> None:
    padding_constraints[var].append(new_mod_num)  # 在填充约束字典中的变量对应列表中添加新模数
    mod_num = np.lcm.reduce(padding_constraints[var][1:])  # 计算列表中除第一个元素外的所有元素的最小公倍数
    eq = mod_num * init_symints[idx]  # 计算新的方程
    eq_const = [arg for arg in init_eq.args if arg.is_number]  # 提取初始方程中的常数项
    if eq_const:
        rem = int(eq_const[0] % mod_num)  # 计算常数项对新模数的余数
        eq -= rem  # 调整新方程，使其满足新模数
    symints[idx] = eq  # 更新符号整数列表中的值为新方程
```