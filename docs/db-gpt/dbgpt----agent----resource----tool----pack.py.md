# `.\DB-GPT-src\dbgpt\agent\resource\tool\pack.py`

```py
"""
Tool resource pack module.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from ..base import ResourceType, T
from ..pack import Resource, ResourcePack
from .base import DB_GPT_TOOL_IDENTIFIER, BaseTool, FunctionTool, ToolFunc
from .exceptions import ToolExecutionException, ToolNotFoundException

ToolResourceType = Union[BaseTool, List[BaseTool], ToolFunc, List[ToolFunc]]


def _is_function_tool(resources: Any) -> bool:
    """
    Check if the given resource is a function tool.

    Args:
        resources: Any object to check.

    Returns:
        bool: True if the object is a function tool, False otherwise.
    """
    return (
        callable(resources)
        and hasattr(resources, DB_GPT_TOOL_IDENTIFIER)
        and getattr(resources, DB_GPT_TOOL_IDENTIFIER)
        and hasattr(resources, "_tool")
        and isinstance(getattr(resources, "_tool"), BaseTool)
    )


def _to_tool_list(resources: ToolResourceType) -> List[BaseTool]:
    """
    Convert various types of resources into a list of BaseTool objects.

    Args:
        resources: The input resources to convert.

    Returns:
        List[BaseTool]: List of BaseTool objects extracted from the input.

    Raises:
        ValueError: If the input resources cannot be converted to BaseTool objects.
    """
    if isinstance(resources, BaseTool):
        return [resources]
    elif isinstance(resources, list) and all(
        isinstance(r, BaseTool) for r in resources
    ):
        return cast(List[BaseTool], resources)
    elif isinstance(resources, list) and all(_is_function_tool(r) for r in resources):
        return [cast(FunctionTool, getattr(r, "_tool")) for r in resources]
    elif _is_function_tool(resources):
        function_tool = cast(FunctionTool, getattr(resources, "_tool"))
        return [function_tool]
    raise ValueError("Invalid tool resource type")


class ToolPack(ResourcePack):
    """
    Tool resource pack class.

    Inherits from ResourcePack and manages a collection of tools.
    """

    def __init__(
        self, resources: ToolResourceType, name: str = "Tool Resource Pack", **kwargs
    ):
        """
        Initialize the tool resource pack.

        Args:
            resources (ToolResourceType): The resources to include in the pack.
            name (str, optional): Name of the resource pack. Defaults to "Tool Resource Pack".
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        tools = cast(List[Resource], _to_tool_list(resources))
        super().__init__(resources=tools, name=name, **kwargs)

    @classmethod
    def from_resource(
        cls: Type[T],
        resource: Optional[Resource],
        expected_type: Optional[ResourceType] = None,
    ) -> List[T]:
        """
        Create a list of ToolPack instances from a given resource.

        Args:
            resource (Optional[Resource]): The resource to create ToolPack instances from.
            expected_type (Optional[ResourceType], optional): Expected resource type. Defaults to None.

        Returns:
            List[T]: List of ToolPack instances created from the resource.

        Notes:
            If the resource is not of the expected type or does not contain tools, an empty list is returned.
        """
        if not resource:
            return []
        if isinstance(resource, ToolPack):
            return [cast(T, resource)]
        tools = super().from_resource(resource, ResourceType.Tool)
        if not tools:
            return []
        typed_tools = [cast(BaseTool, t) for t in tools]
        return [ToolPack(typed_tools)]  # type: ignore

    async def get_prompt(
        self,
        *,
        lang: str = "en",
        prompt_type: str = "default",
        question: Optional[str] = None,
        resource_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Asynchronously retrieve a prompt from the tool pack.

        Args:
            lang (str, optional): Language code for the prompt. Defaults to "en".
            prompt_type (str, optional): Type of prompt to retrieve. Defaults to "default".
            question (Optional[str], optional): Optional question text. Defaults to None.
            resource_name (Optional[str], optional): Optional name of the resource. Defaults to None.
            **kwargs: Additional keyword arguments for customization.

        Notes:
            This method is intended to be overridden by subclasses to provide specific prompt retrieval logic.
        """
        pass  # Placeholder for asynchronous prompt retrieval logic
    ) -> str:
        """获取资源的提示信息。"""
        # 初始化一个空列表，用于存储每个资源的提示信息
        prompt_list = []
        # 遍历资源字典中的每个资源
        for name, resource in self._resources.items():
            # 调用每个资源的获取提示信息方法，并将返回的提示信息添加到列表中
            prompt = await resource.get_prompt(
                lang=lang,
                prompt_type=prompt_type,
                question=question,
                resource_name=resource_name,
                **kwargs,
            )
            prompt_list.append(prompt)
        # 根据语言选择头部提示信息
        head_prompt = (
            "You have access to the following APIs:" if lang == "en" else "您可以使用以下API："
        )
        # 返回头部提示信息和所有资源的提示信息拼接而成的字符串
        return head_prompt + "\n" + self._prompt_separator.join(prompt_list)

    def add_command(
        self,
        command_label: str,
        command_name: str,
        args: Optional[Dict[str, Any]] = None,
        function: Optional[Callable] = None,
    ) -> None:
        """向命令列表中添加命令。

        兼容Auto-GPT旧插件系统。

        向命令列表中添加带有标签、名称和可选参数的命令。

        Args:
            command_label (str): 命令的标签。
            command_name (str): 命令的名称。
            args (dict, optional): 包含参数名及其值的字典。默认为None。
            function (callable, optional): 执行命令时要调用的可调用函数。默认为None。
        """
        # 如果有提供参数，则创建一个空字典用于存储工具参数
        if args is not None:
            tool_args = {}
            # 遍历参数字典，为每个参数创建一个描述性条目
            for name, value in args.items():
                tool_args[name] = {
                    "name": name,
                    "type": "str",
                    "description": value,
                }
        else:
            tool_args = {}
        # 如果未提供函数，则抛出值错误异常
        if not function:
            raise ValueError("Function must be provided")

        # 创建一个FunctionTool对象，用于表示命令及其相关信息
        ft = FunctionTool(
            name=command_name,
            func=function,
            args=tool_args,
            description=command_label,
        )
        # 将创建的FunctionTool对象添加到当前对象的命令列表中
        self.append(ft)

    def _get_execution_tool(
        self,
        name: Optional[str] = None,
    ) -> BaseTool:
        # 如果未提供名称或名称不在资源列表中，则抛出工具未找到异常
        if not name or name not in self._resources:
            raise ToolNotFoundException("No tool found for execution")
        # 返回指定名称的资源对象
        return cast(BaseTool, self._resources[name])

    def _get_call_args(self, arguments: Dict[str, Any], tl: BaseTool) -> Dict[str, Any]:
        """获取调用参数。"""
        # 找出参数字典中与工具参数不匹配的参数，并从参数字典中删除这些参数
        diff_args = list(set(arguments.keys()).difference(set(tl.args.keys())))
        for arg_name in diff_args:
            del arguments[arg_name]
        # 返回经过筛选后的参数字典
        return arguments

    def execute(
        self,
        *args,
        resource_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Execute the tool.

        Args:
            *args: The positional arguments.
            resource_name (str, optional): The tool name to be executed.
            **kwargs: The keyword arguments.

        Returns:
            Any: The result of the tool execution.
        """
        # 获取执行工具对象
        tl = self._get_execution_tool(resource_name)
        try:
            # 将关键字参数转换为字典形式
            arguments = {k: v for k, v in kwargs.items()}
            # 获取调用参数
            arguments = self._get_call_args(arguments, tl)
            # 检查工具是否支持异步执行
            if tl.is_async:
                # 异步执行不支持，抛出异常
                raise ToolExecutionException("Async execution is not supported")
            else:
                # 同步执行工具并返回结果
                return tl.execute(**arguments)
        except Exception as e:
            # 捕获异常并抛出工具执行异常
            raise ToolExecutionException(f"Execution error: {str(e)}")

    async def async_execute(
        self,
        *args,
        resource_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Execute the tool asynchronously.

        Args:
            *args: The positional arguments.
            resource_name (str, optional): The tool name to be executed.
            **kwargs: The keyword arguments.

        Returns:
            Any: The result of the tool execution.
        """
        # 获取执行工具对象
        tl = self._get_execution_tool(resource_name)
        try:
            # 将关键字参数转换为字典形式
            arguments = {k: v for k, v in kwargs.items()}
            # 获取调用参数
            arguments = self._get_call_args(arguments, tl)
            # 检查工具是否支持异步执行
            if tl.is_async:
                # 异步执行工具并返回结果
                return await tl.async_execute(**arguments)
            else:
                # 同步执行工具（此处为待实现功能的提醒）
                # TODO: Execute in a separate executor
                return tl.execute(**arguments)
        except Exception as e:
            # 捕获异常并抛出工具执行异常
            raise ToolExecutionException(f"Execution error: {str(e)}")
class AutoGPTPluginToolPack(ToolPack):
    """Auto-GPT plugin tool pack class."""

    def __init__(self, plugin_path: Union[str, List[str]], **kwargs):
        """Create an Auto-GPT plugin tool pack."""
        # 调用父类的初始化方法，传入空列表和其他关键字参数
        super().__init__([], **kwargs)
        # 设置插件路径属性
        self._plugin_path = plugin_path
        # 初始化加载状态为未加载
        self._loaded = False

    def preload_resource(self):
        """Preload the resource."""
        # 导入必要的工具函数，用于扫描插件文件和目录
        from .autogpt.plugins_util import scan_plugin_file, scan_plugins
        
        # 如果已经加载过资源，直接返回
        if self._loaded:
            return
        
        # 根据插件路径类型进行处理，确保路径为列表形式
        paths = (
            [self._plugin_path]
            if isinstance(self._plugin_path, str)
            else self._plugin_path
        )
        
        # 初始化插件列表
        plugins = []
        
        # 遍历处理每个路径
        for path in paths:
            # 如果路径是绝对路径且不存在，抛出异常
            if os.path.isabs(path):
                if not os.path.exists(path):
                    raise ValueError(f"Wrong plugin path configured {path}!")
                # 如果是文件，扫描文件并添加到插件列表
                if os.path.isfile(path):
                    plugins.extend(scan_plugin_file(path))
                else:
                    # 如果是目录，扫描目录下所有插件并添加到插件列表
                    plugins.extend(scan_plugins(path))
        
        # 对于每个插件，检查是否可以处理后提示，如果可以则执行后提示方法
        for plugin in plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(self)
        
        # 设置加载状态为已加载
        self._loaded = True
```