# `.\AutoGPT\autogpts\autogpt\autogpt\agents\utils\prompt_scratchpad.py`

```py
# 导入 logging 模块
import logging
# 导入 Callable 类型提示
from typing import Callable

# 导入 BaseModel 和 Field 类
from pydantic import BaseModel, Field

# 导入自定义模块
from autogpt.core.resource.model_providers.schema import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema

# 获取 logger 对象
logger = logging.getLogger("PromptScratchpad")

# 定义 CallableCompletionModelFunction 类，继承自 CompletionModelFunction
class CallableCompletionModelFunction(CompletionModelFunction):
    # 定义 method 属性为 Callable 类型
    method: Callable

# 定义 PromptScratchpad 类，继承自 BaseModel
class PromptScratchpad(BaseModel):
    # 定义 commands 属性为字典，键为 str 类型，值为 CallableCompletionModelFunction 类型
    commands: dict[str, CallableCompletionModelFunction] = Field(default_factory=dict)
    # 定义 resources 属性为列表，元素为 str 类型
    resources: list[str] = Field(default_factory=list)
    # 定义 constraints 属性为列表，元素为 str 类型
    constraints: list[str] = Field(default_factory=list)
    # 定义 best_practices 属性为列表，元素为 str 类型
    best_practices: list[str] = Field(default_factory=list)

    # 定义 add_constraint 方法，用于向 constraints 列表中添加约束条件
    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Params:
            constraint (str): The constraint to be added.
        """
        # 如果约束条件不在 constraints 列表中，则添加
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    # 定义 add_command 方法，用于向 commands 字典中添加命令
    def add_command(
        self,
        name: str,
        description: str,
        params: dict[str, str | dict],
        function: Callable,
    ) -> None:
        """
        注册一个命令。

        *仅应由插件使用。* 本地命令应直接添加到CommandRegistry中。

        参数:
            name (str): 命令的名称（例如 `command_name`）。
            description (str): 命令的描述。
            params (dict, optional): 包含参数名称及其类型的字典。默认为空字典。
            function (callable, optional): 当执行命令时要调用的可调用函数。默认为None。
        """
        遍历参数字典
        for p, s in params.items():
            invalid = False
            检查参数类型是否有效
            if type(s) is str and s not in JSONSchema.Type._value2member_map_:
                invalid = True
                记录警告信息
                logger.warning(
                    f"Cannot add command '{name}':"
                    f" parameter '{p}' has invalid type '{s}'."
                    f" Valid types are: {JSONSchema.Type._value2member_map_.keys()}"
                )
            elif isinstance(s, dict):
                尝试从字典创建JSONSchema对象
                try:
                    JSONSchema.from_dict(s)
                except KeyError:
                    invalid = True
            如果参数无效，则返回
            if invalid:
                return

        创建CallableCompletionModelFunction对象
        command = CallableCompletionModelFunction(
            name=name,
            description=description,
            parameters={
                根据参数字典创建参数名称到JSONSchema对象的映射
                name: JSONSchema(type=JSONSchema.Type._value2member_map_[spec])
                if type(spec) is str
                else JSONSchema.from_dict(spec)
                for name, spec in params.items()
            },
            method=function,
        )

        检查命令是否已存在，如果存在则替换
        if name in self.commands:
            if description == self.commands[name].description:
                返回
                return
            记录警告信息
            logger.warning(
                f"Replacing command {self.commands[name]} with conflicting {command}"
            )
        将命令添加到commands字典中
        self.commands[name] = command
    # 将资源添加到资源列表中
    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Params:
            resource (str): The resource to be added.
        """
        # 检查资源是否已经存在于资源列表中，如果不存在则添加
        if resource not in self.resources:
            self.resources.append(resource)

    # 将最佳实践项添加到最佳实践列表中
    def add_best_practice(self, best_practice: str) -> None:
        """
        Add an item to the list of best practices.

        Params:
            best_practice (str): The best practice item to be added.
        """
        # 检查最佳实践项是否已经存在于最佳实践列表中，如果不存在则添加
        if best_practice not in self.best_practices:
            self.best_practices.append(best_practice)
```