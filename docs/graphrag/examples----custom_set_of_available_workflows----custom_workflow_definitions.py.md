# `.\graphrag\examples\custom_set_of_available_workflows\custom_workflow_definitions.py`

```py
# 导入工作流定义模块，这里假设该模块为 graphrag.index.workflows
from graphrag.index.workflows import WorkflowDefinitions

# 设置自定义工作流列表，用于在流水线中使用
# 思路是你可以在任意数量的流水线中使用这些工作流
custom_workflows: WorkflowDefinitions = {
    # 定义名为 "my_workflow" 的工作流，接受一个配置参数 config
    "my_workflow": lambda config: [
        {
            "verb": "derive",  # 操作类型为“derive”
            "args": {
                "column1": "col1",  # 在数据集中查找名为 col1 的列
                "column2": "col2",  # 在数据集中查找名为 col2 的列
                "to": config.get(
                    # 允许用户指定输出列名，否则默认为 "output_column"
                    "derive_output_column",
                    "output_column",
                ),  # 新列名
                "operator": "*",  # 操作符为乘法
            },
        }
    ],
    # 定义名为 "my_unused_workflow" 的未使用工作流，接受一个占位符配置参数 _config
    "my_unused_workflow": lambda _config: [
        {
            "verb": "derive",  # 操作类型为“derive”
            "args": {
                "column1": "col1",  # 在数据集中查找名为 col1 的列
                "column2": "col2",  # 在数据集中查找名为 col2 的列
                "to": "unused_output_column",  # 固定为 "unused_output_column"
                "operator": "*",  # 操作符为乘法
            },
        }
    ],
}
```