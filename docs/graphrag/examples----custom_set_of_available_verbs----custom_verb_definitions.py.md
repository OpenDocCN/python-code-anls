# `.\graphrag\examples\custom_set_of_available_verbs\custom_verb_definitions.py`

```py
# 从datashaper模块导入TableContainer和VerbInput类
from datashaper import TableContainer, VerbInput

# 定义一个自定义函数str_append，用于向指定列添加字符串
def str_append(
    input: VerbInput, source_column: str, target_column: str, string_to_append: str
):
    """A custom verb that appends a string to a column"""
    # 按照约定，通常使用"column"作为输入列名，"to"作为输出列名，但你可以使用任何你喜欢的名称
    # 只要在工作流中的"args"与函数签名匹配即可
    # 从输入参数中获取输入数据
    input_data = input.get_input()
    # 复制输入数据，避免直接修改原始数据
    output_df = input_data.copy()
    # 使用lambda函数将指定列的每个元素与指定的字符串拼接
    output_df[target_column] = output_df[source_column].apply(
        lambda x: f"{x}{string_to_append}"
    )
    # 将处理后的数据封装成TableContainer对象并返回
    return TableContainer(table=output_df)

# 定义一个字典custom_verbs，将"str_append"作为键，str_append函数作为值
custom_verbs = {
    "str_append": str_append,
}
```