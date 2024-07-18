# `.\graphrag\tests\unit\indexing\workflows\helpers.py`

```py
# 定义一个模拟的动词字典，包含两个键值对，每个值是一个接受一个参数并返回该参数的 lambda 函数
mock_verbs = {
    "mock_verb": lambda x: x,
    "mock_verb_2": lambda x: x,
}

# 定义一个模拟的工作流字典，包含两个键值对，每个值是一个接受一个参数并返回包含操作步骤的列表的 lambda 函数
mock_workflows = {
    "mock_workflow": lambda _x: [
        {
            "verb": "mock_verb",  # 第一个步骤使用 mock_verb 动词
            "args": {
                "column": "test",  # 指定参数 column 的值为 "test"
            },
        }
    ],
    "mock_workflow_2": lambda _x: [
        {
            "verb": "mock_verb",  # 第一个步骤使用 mock_verb 动词
            "args": {
                "column": "test",  # 指定参数 column 的值为 "test"
            },
        },
        {
            "verb": "mock_verb_2",  # 第二个步骤使用 mock_verb_2 动词
            "args": {
                "column": "test",  # 指定参数 column 的值为 "test"
            },
        },
    ],
}
```