# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\builtins\file_operations.py`

```py
# 导入 logging 模块
import logging
# 导入 os 模块
import os
# 导入 ClassVar 类型提示
from typing import ClassVar

# 导入 Ability、AbilityConfiguration 类
from autogpt.core.ability.base import Ability, AbilityConfiguration
# 导入 AbilityResult、ContentType、Knowledge 类
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
# 导入 PluginLocation、PluginStorageFormat 类
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
# 导入 JSONSchema 类
from autogpt.core.utils.json_schema import JSONSchema
# 导入 Workspace 类
from autogpt.core.workspace import Workspace

# 定义 ReadFile 类，继承自 Ability 类
class ReadFile(Ability):
    # 默认配置
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.ReadFile",
        ),
        packages_required=["unstructured"],
        workspace_required=True,
    )

    # 初始化方法
    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        # 初始化 logger 属性
        self._logger = logger
        # 初始化 workspace 属性
        self._workspace = workspace

    # 描述信息
    description: ClassVar[str] = "Read and parse all text from a file."

    # 参数定义
    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to read.",
        ),
    }

    # 检查前提条件的私有方法
    def _check_preconditions(self, filename: str) -> AbilityResult | None:
        # 初始化消息
        message = ""
        try:
            pass
        except ImportError:
            message = "Package charset_normalizer is not installed."

        try:
            # 获取文件路径
            file_path = self._workspace.get_path(filename)
            # 检查文件是否存在
            if not file_path.exists():
                message = f"File {filename} does not exist."
            # 检查是否为文件
            if not file_path.is_file():
                message = f"{filename} is not a file."
        except ValueError as e:
            message = str(e)

        # 如果存在消息，则返回 AbilityResult 对象
        if message:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"filename": filename},
                success=False,
                message=message,
                data=None,
            )
    # 定义一个方法，接受文件名作为参数，并返回 AbilityResult 对象
    def __call__(self, filename: str) -> AbilityResult:
        # 检查前提条件，如果满足则返回结果
        if result := self._check_preconditions(filename):
            return result

        # 导入 partition 方法
        from unstructured.partition.auto import partition

        # 获取文件在工作空间中的路径
        file_path = self._workspace.get_path(filename)
        try:
            # 对文件进行分区，返回元素列表
            elements = partition(str(file_path))
            # 创建新的 Knowledge 对象，包含分区文件中所有文本内容
            new_knowledge = Knowledge(
                content="\n\n".join([element.text for element in elements]),
                content_type=ContentType.TEXT,
                content_metadata={"filename": filename},
            )
            success = True
            message = f"File {file_path} read successfully."
        except IOError as e:
            # 如果出现 IOError 异常，将新知识对象设为 None，标记操作失败，并记录异常信息
            new_knowledge = None
            success = False
            message = str(e)

        # 返回 AbilityResult 对象，包含操作结果信息
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
            new_knowledge=new_knowledge,
        )
class WriteFile(Ability):
    # 写文件的能力类，继承自 Ability 类
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.WriteFile",
        ),
        packages_required=["unstructured"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        # 初始化方法，接受 logger 和 workspace 作为参数
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = "Write text to a file."
    # 描述写文件功能的类变量

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to write.",
        ),
        "contents": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The contents of the file to write.",
        ),
    }
    # 参数定义，包括文件名和文件内容

    def _check_preconditions(
        self, filename: str, contents: str
    ) -> AbilityResult | None:
        # 检查前置条件的方法，接受文件名和文件内容作为参数，返回 AbilityResult 或 None
        message = ""
        # 初始化消息为空字符串
        try:
            file_path = self._workspace.get_path(filename)
            # 获取文件路径
            if file_path.exists():
                message = f"File {filename} already exists."
            # 如果文件已经存在，则更新消息
            if len(contents):
                message = f"File {filename} was not given any content."
            # 如果内容长度为 0，则更新消息
        except ValueError as e:
            message = str(e)
            # 捕获异常并更新消息

        if message:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"filename": filename, "contents": contents},
                success=False,
                message=message,
                data=None,
            )
            # 如果消息不为空，则返回 AbilityResult 对象
    # 定义一个方法，接受文件名和内容作为参数，并返回一个 AbilityResult 对象
    def __call__(self, filename: str, contents: str) -> AbilityResult:
        # 检查前置条件，如果有结果则直接返回
        if result := self._check_preconditions(filename, contents):
            return result

        # 获取文件在工作空间中的路径
        file_path = self._workspace.get_path(filename)
        try:
            # 获取文件所在目录，并创建目录
            directory = os.path.dirname(file_path)
            os.makedirs(directory)
            # 打开文件并写入内容
            with open(filename, "w", encoding="utf-8") as f:
                f.write(contents)
            success = True
            message = f"File {file_path} written successfully."
        except IOError as e:
            success = False
            message = str(e)

        # 返回 AbilityResult 对象，包含能力名称、参数、成功标志和消息
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
        )
```