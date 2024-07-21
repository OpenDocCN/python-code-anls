# `.\pytorch\torch\distributed\checkpoint\_nested_dict.py`

```py
"""
TODO:
Need to add ability to handle tuple, OrderedDict, NamedTuple.
Update mappings from dict to a class.
Change set_element to recreate the right type for tuple, OrderedDict, and NamedTuple.
"""


# 字典类型，用于存储字符串到对象路径的映射
FLATTEN_MAPPING = Dict[str, OBJ_PATH]


# TODO: Update Docstring for nested_dict.py
# 函数用于将嵌套的状态字典转换为扁平化的状态字典
def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    # 初始化一个空的扁平化状态字典
    flattened: STATE_DICT_TYPE = {}
    # 初始化一个空的映射字典，用于记录原始状态字典和扁平化状态字典之间的映射关系
    mappings: FLATTEN_MAPPING = {}

    # 内部函数，用于递归遍历状态字典并将其扁平化
    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        # 将对象路径转换为以点分隔的字符串，作为新的扁平化状态字典的键
        new_fqn = ".".join(map(str, path))
        # 如果新键已经存在于扁平化状态字典中，则抛出值错误异常
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        # 将原始状态字典中的值复制到扁平化状态字典中，并记录映射关系
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    # 调用traverse_state_dict函数来遍历原始状态字典并进行扁平化处理
    traverse_state_dict(state_dict, flat_copy)
    # 返回扁平化后的状态字典及其映射关系
    return flattened, mappings


# 函数用于根据映射关系和扁平化状态字典，恢复原始的嵌套状态字典
def unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    # 初始化一个空的嵌套状态字典
    nested: STATE_DICT_TYPE = {}
    # 遍历扁平化状态字典中的每一项，并将其按照映射关系设置回嵌套状态字典中
    for key, value in state_dict.items():
        set_element(nested, mapping[key], value)
    # 返回恢复后的嵌套状态字典
    return nested
```