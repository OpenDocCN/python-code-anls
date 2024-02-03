# `.\AutoGPT\autogpts\forge\forge\actions\registry.py`

```py
# 导入所需的模块
import glob
import importlib
import inspect
import os
from typing import Any, Callable, List

import pydantic

# 定义一个参数类，表示一个动作的参数
class ActionParameter(pydantic.BaseModel):
    """
    This class represents a parameter for an action.

    Attributes:
        name (str): The name of the parameter.
        description (str): A brief description of what the parameter does.
        type (str): The type of the parameter.
        required (bool): A flag indicating whether the parameter is required or optional.
    """

    name: str
    description: str
    type: str
    required: bool

# 定义一个动作类，表示系统中的一个动作
class Action(pydantic.BaseModel):
    """
    This class represents an action in the system.

    Attributes:
        name (str): The name of the action.
        description (str): A brief description of what the action does.
        method (Callable): The method that implements the action.
        parameters (List[ActionParameter]): A list of parameters that the action requires.
        output_type (str): The type of the output that the action returns.
    """

    name: str
    description: str
    method: Callable
    parameters: List[ActionParameter]
    output_type: str
    category: str | None = None

    # 定义一个特殊方法，使类实例可以像函数一样被调用
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        This method allows the class instance to be called as a function.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the method call.
        """
        return self.method(*args, **kwds)
    # 定义一个特殊方法，用于返回类实例的字符串表示形式
    def __str__(self) -> str:
        """
        This method returns a string representation of the class instance.

        Returns:
            str: A string representation of the class instance.
        """
        # 初始化函数摘要字符串，包含函数名
        func_summary = f"{self.name}("
        # 遍历函数参数列表，将参数名和类型添加到函数摘要字符串中
        for param in self.parameters:
            func_summary += f"{param.name}: {param.type}, "
        # 去除最后一个逗号和空格
        func_summary = func_summary[:-2] + ")"
        # 添加函数的输出类型和用法描述到函数摘要字符串中
        func_summary += f" -> {self.output_type}. Usage: {self.description},"
        # 返回函数摘要字符串
        return func_summary
# 定义一个装饰器函数，用于为函数添加动作注解
def action(
    name: str, description: str, parameters: List[ActionParameter], output_type: str
):
    # 定义装饰器函数，接受一个函数作为参数
    def decorator(func):
        # 获取函数的参数签名
        func_params = inspect.signature(func).parameters
        # 解析参数列表中的参数名，并添加到集合中
        param_names = set(
            [ActionParameter.parse_obj(param).name for param in parameters]
        )
        # 添加固定的参数名到集合中
        param_names.add("agent")
        param_names.add("task_id")
        # 获取函数参数的名称集合
        func_param_names = set(func_params.keys())
        # 检查参数名集合是否一致，如果不一致则抛出异常
        if param_names != func_param_names:
            raise ValueError(
                f"Mismatch in parameter names. Action Annotation includes {param_names}, but function actually takes {func_param_names} in function {func.__name__} signature"
            )
        # 为函数添加动作注解
        func.action = Action(
            name=name,
            description=description,
            parameters=parameters,
            method=func,
            output_type=output_type,
        )
        # 返回装饰后的函数
        return func

    # 返回装饰器函数
    return decorator

# 定义一个动作注册类
class ActionRegister:
    # 初始化方法，接受一个代理对象作为参数
    def __init__(self, agent) -> None:
        # 初始化能力字典
        self.abilities = {}
        # 注册能力
        self.register_abilities()
        # 设置代理对象
        self.agent = agent
    # 注册能力函数，遍历指定目录下的所有.py文件
    def register_abilities(self) -> None:
        for action_path in glob.glob(
            os.path.join(os.path.dirname(__file__), "**/*.py"), recursive=True
        ):
            # 排除特定文件
            if not os.path.basename(action_path) in [
                "__init__.py",
                "registry.py",
            ]:
                # 获取相对路径并替换斜杠为点
                action = os.path.relpath(
                    action_path, os.path.dirname(__file__)
                ).replace("/", ".")
                try:
                    # 动态导入模块
                    module = importlib.import_module(
                        f".{action[:-3]}", package="forge.actions"
                    )
                    # 遍历模块中的属性
                    for attr in dir(module):
                        func = getattr(module, attr)
                        # 判断属性是否为一个能力函数
                        if hasattr(func, "action"):
                            ab = func.action

                            # 设置能力的分类
                            ab.category = (
                                action.split(".")[0].lower().replace("_", " ")
                                if len(action.split(".")) > 1
                                else "general"
                            )
                            # 将能力添加到能力字典中
                            self.abilities[func.action.name] = func.action
                except Exception as e:
                    # 捕获异常并打印错误信息
                    print(f"Error occurred while registering abilities: {str(e)}")

    # 返回所有能力的列表
    def list_abilities(self) -> List[Action]:
        return self.abilities

    # 返回用于提示的能力列表
    def list_abilities_for_prompt(self) -> List[str]:
        return [str(action) for action in self.abilities.values()]
    # 返回代理的能力描述
    def abilities_description(self) -> str:
        # 创建一个空字典，用于按类别存储能力
        abilities_by_category = {}
        # 遍历代理的所有能力
        for action in self.abilities.values():
            # 如果能力的类别不在 abilities_by_category 中，则添加一个空列表
            if action.category not in abilities_by_category:
                abilities_by_category[action.category] = []
            # 将能力添加到对应类别的列表中
            abilities_by_category[action.category].append(str(action))

        # 初始化能力描述字符串
        abilities_description = ""
        # 遍历 abilities_by_category 中的类别和对应的能力列表
        for category, abilities in abilities_by_category.items():
            # 如果 abilities_description 不为空，则添加换行符
            if abilities_description != "":
                abilities_description += "\n"
            # 添加类别到能力描述字符串
            abilities_description += f"{category}:"
            # 遍历类别下的所有能力，添加到能力描述字符串中
            for action in abilities:
                abilities_description += f"  {action}"

        # 返回能力描述字符串
        return abilities_description

    # 异步运行指定的动作
    async def run_action(
        self, task_id: str, action_name: str, *args: Any, **kwds: Any
    ) -> Any:
        """
        This method runs a specified action with the provided arguments and keyword arguments.

        The agent is passed as the first argument to the action. This allows the action to access and manipulate
        the agent's state as needed.

        Args:
            task_id (str): The ID of the task that the action is being run for.
            action_name (str): The name of the action to run.
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the action execution.

        Raises:
            Exception: If there is an error in running the action.
        """
        try:
            # 获取指定名称的动作
            action = self.abilities[action_name]
            # 调用动作并传入代理、任务ID、参数和关键字参数
            return await action(self.agent, task_id, *args, **kwds)
        except Exception:
            # 如果出现异常，则抛出异常
            raise
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 导入 sys 模块
    import sys

    # 将指定路径添加到 sys.path 中，以便导入模块
    sys.path.append("/Users/swifty/dev/forge/forge")
    
    # 创建一个 ActionRegister 对象，agent 参数为 None
    register = ActionRegister(agent=None)
    
    # 打印注册对象的能力描述
    print(register.abilities_description())
    
    # 运行指定动作，输出结果
    print(register.run_action("abc", "list_files", "/Users/swifty/dev/forge/forge"))
```