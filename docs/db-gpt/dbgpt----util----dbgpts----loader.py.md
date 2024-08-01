# `.\DB-GPT-src\dbgpt\util\dbgpts\loader.py`

```py
# 导入模块 inspect 用于获取有关对象的信息
import inspect
# 导入 logging 模块用于记录日志
import logging
# 导入 os 模块提供与操作系统相关的功能
import os
# 导入 sys 模块提供与解释器交互的功能
import sys
# 从 pathlib 模块导入 Path 类
from pathlib import Path
# 导入 typing 模块中的泛型类和函数
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, cast

# 导入 schedule 模块用于任务调度
import schedule
# 导入 tomlkit 模块用于处理 TOML 格式的文件
import tomlkit

# 导入 BaseModel, ConfigDict, Field 和 model_validator 类和函数
# 从 dbgpt._private.pydantic 模块
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_validator
# 从 dbgpt.component 模块导入 BaseComponent 和 SystemApp 类
from dbgpt.component import BaseComponent, SystemApp
# 从 dbgpt.core.awel 中导入 DAG 类
from dbgpt.core.awel import DAG
# 从 dbgpt.core.awel.flow.flow_factory 中导入 FlowPanel 类
from dbgpt.core.awel.flow.flow_factory import FlowPanel
# 从 dbgpt.util.dbgpts.base 中导入常量 DBGPTS_METADATA_FILE, INSTALL_DIR 和 INSTALL_METADATA_FILE
from dbgpt.util.dbgpts.base import (
    DBGPTS_METADATA_FILE,
    INSTALL_DIR,
    INSTALL_METADATA_FILE,
)

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 定义一个泛型类型 T
T = TypeVar("T")


# 定义一个 BasePackage 类，继承自 BaseModel 类
class BasePackage(BaseModel):
    # 定义一个 ConfigDict 类型的 model_config 属性
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 定义 name 属性，表示包的名称，必须提供
    name: str = Field(..., description="The name of the package")
    # 定义 label 属性，表示包的标签，必须提供
    label: str = Field(..., description="The label of the package")
    # 定义 package_type 属性，表示包的类型，必须提供
    package_type: str = Field(..., description="The type of the package")
    # 定义 version 属性，表示包的版本，必须提供
    version: str = Field(..., description="The version of the package")
    # 定义 description 属性，表示包的描述，必须提供
    description: str = Field(..., description="The description of the package")
    # 定义 path 属性，表示包的路径，必须提供
    path: str = Field(..., description="The path of the package")
    # 定义 authors 属性，表示包的作者列表，默认为空列表
    authors: List[str] = Field(
        default_factory=list, description="The authors of the package"
    )
    # 定义 definition_type 属性，表示定义文件的类型，默认为 "python"
    definition_type: str = Field(
        default="python", description="The type of the package"
    )
    # 定义 definition_file 属性，表示定义文件的路径，默认为 None
    definition_file: Optional[str] = Field(
        default=None, description="The definition " "file of the package"
    )
    # 定义 root 属性，表示包的根路径，必须提供
    root: str = Field(..., description="The root of the package")
    # 定义 repo 属性，表示包的仓库地址，必须提供
    repo: str = Field(..., description="The repository of the package")
    # 定义 package 属性，表示包的名称（例如在 PyPI 中的名称）
    package: str = Field(..., description="The package name(like name in pypi)")

    # 定义一个类方法 build_from，根据给定的值创建 BasePackage 对象
    @classmethod
    def build_from(cls, values: Dict[str, Any], ext_dict: Dict[str, Any]):
        return cls(**values)

    # 定义一个模型验证器方法 pre_fill，在对象构建之前填充 definition_file 属性
    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-fill the definition_file"""
        # 如果 values 不是字典类型，则直接返回 values
        if not isinstance(values, dict):
            return values
        # 导入 importlib.resources 模块作为 pkg_resources
        import importlib.resources as pkg_resources

        # 获取 name 和 root 属性的值
        name = values.get("name")
        root = values.get("root")
        # 如果 name 不存在，抛出 ValueError 异常
        if not name:
            raise ValueError("The name is required")
        # 如果 root 不存在，抛出 ValueError 异常
        if not root:
            raise ValueError("The root is required")
        # 如果 root 不在 sys.path 中，将其添加到 sys.path 中
        if root not in sys.path:
            sys.path.append(root)
        # 使用 pkg_resources.path 获取指定包的 __init__.py 文件的路径
        with pkg_resources.path(name, "__init__.py") as path:
            # 将路径的目录部分作为 path 属性值保存到 values 中
            values["path"] = os.path.dirname(os.path.abspath(path))
        return values

    # 定义一个方法 abs_definition_file，返回完整的定义文件路径
    def abs_definition_file(self) -> str:
        return str(Path(self.path) / self.definition_file)

    # 定义一个类方法 load_module_class，加载指定类模块并返回其实例
    @classmethod
    def load_module_class(
        cls,
        values: Dict[str, Any],
        expected_cls: Type[T],
        predicates: Optional[List[Callable[..., bool]]] = None,
    # 定义函数签名，该函数接受一些参数并返回三个列表的元组
    def _load_modules(values: Dict[str, Any], predicates: List[Callable[..., bool]], expected_cls: Type[T]) -> Tuple[List[Type[T]], List[Any], List[Any]]:
        # 导入包资源模块
        import importlib.resources as pkg_resources
        # 导入自定义模块，从中加载模块的函数
        from dbgpt.core.awel.dag.loader import _load_modules_from_file
    
        # 从参数中获取名称和根目录
        name = values.get("name")
        root = values.get("root")
        
        # 如果名称为空，抛出数值错误异常
        if not name:
            raise ValueError("The name is required")
        # 如果根目录为空，抛出数值错误异常
        if not root:
            raise ValueError("The root is required")
        
        # 如果根目录不在系统路径中，则添加到系统路径中
        if root not in sys.path:
            sys.path.append(root)
        
        # 使用包资源模块获取初始化文件的路径
        with pkg_resources.path(name, "__init__.py") as path:
            # 调用自定义函数加载模块并返回结果
            mods = _load_modules_from_file(str(path), name, show_log=False)
            
            # 从加载的每个模块中获取所有类
            all_cls = [_get_classes_from_module(m) for m in mods]
            
            # 初始化一个空列表来存储所有的断言结果
            all_predicate_results = []
            # 遍历每个模块，并扩展所有的断言结果列表
            for m in mods:
                all_predicate_results.extend(_get_from_module(m, predicates))
            
            # 初始化一个空列表来存储符合预期类的模块类
            module_cls = []
            # 遍历所有类的列表，将符合预期类的类添加到模块类列表中
            for list_cls in all_cls:
                for c in list_cls:
                    if issubclass(c, expected_cls):
                        module_cls.append(c)
            
            # 返回符合预期的模块类列表，所有的断言结果列表以及加载的模块列表
            return module_cls, all_predicate_results, mods
# 定义 FlowPackage 类，继承自 BasePackage 类
class FlowPackage(BasePackage):
    # 定义 package_type 属性为字符串 "flow"
    package_type: str = "flow"

    # 定义类方法 build_from，接受 values 和 ext_dict 两个字典参数，返回 FlowPackage 对象
    @classmethod
    def build_from(
        cls, values: Dict[str, Any], ext_dict: Dict[str, Any]
    ) -> "FlowPackage":
        # 如果 values 字典中的 definition_type 键的值为 "json"，则调用 FlowJsonPackage 类的 build_from 方法
        if values["definition_type"] == "json":
            return FlowJsonPackage.build_from(values, ext_dict)
        # 否则调用 FlowPythonPackage 类的 build_from 方法
        return FlowPythonPackage.build_from(values, ext_dict)


# 定义 FlowPythonPackage 类，继承自 FlowPackage 类
class FlowPythonPackage(FlowPackage):
    # 定义 dag 属性为 DAG 类型，描述为 "The DAG of the package"
    dag: DAG = Field(..., description="The DAG of the package")

    # 定义类方法 build_from，接受 values 和 ext_dict 两个字典参数
    @classmethod
    def build_from(cls, values: Dict[str, Any], ext_dict: Dict[str, Any]):
        # 导入 _process_modules 函数
        from dbgpt.core.awel.dag.loader import _process_modules

        # 调用 load_module_class 方法加载模块类，获取模块信息
        _, _, mods = cls.load_module_class(values, DAG)

        # 处理模块信息，获取 DAG 对象列表
        dags = _process_modules(mods, show_log=False)
        # 如果没有找到 DAG 对象，则抛出 ValueError 异常
        if not dags:
            raise ValueError("No DAGs found in the package")
        # 如果找到多个 DAG 对象，则抛出 ValueError 异常
        if len(dags) > 1:
            raise ValueError("Only support one DAG in the package")
        # 将第一个 DAG 对象添加到 values 字典中的 "dag" 键下，并返回 FlowPythonPackage 对象
        values["dag"] = dags[0]
        return cls(**values)


# 定义 FlowJsonPackage 类，继承自 FlowPackage 类
class FlowJsonPackage(FlowPackage):
    # 定义类方法 build_from，接受 values 和 ext_dict 两个字典参数
    @classmethod
    def build_from(cls, values: Dict[str, Any], ext_dict: Dict[str, Any]):
        # 如果 ext_dict 中不包含 "json_config" 键，则抛出 ValueError 异常
        if "json_config" not in ext_dict:
            raise ValueError("The json_config is required")
        # 如果 "json_config" 中不包含 "file_path" 键，则抛出 ValueError 异常
        if "file_path" not in ext_dict["json_config"]:
            raise ValueError("The file_path is required")
        # 将 "definition_file" 键的值设为 "json_config" 中 "file_path" 键的值，并返回 FlowJsonPackage 对象
        values["definition_file"] = ext_dict["json_config"]["file_path"]
        return cls(**values)

    # 定义 read_definition_json 方法，返回值为字典类型
    def read_definition_json(self) -> Dict[str, Any]:
        # 导入 json 模块
        import json

        # 打开定义文件，读取内容并解析为 JSON 格式，返回解析后的字典
        with open(self.abs_definition_file(), "r", encoding="utf-8") as f:
            return json.loads(f.read())


# 定义 OperatorPackage 类，继承自 BasePackage 类
class OperatorPackage(BasePackage):
    # 定义 package_type 属性为字符串 "operator"
    package_type: str = "operator"

    # 定义 operators 属性为类型列表，默认为空列表，描述为 "The operators of the package"
    operators: List[type] = Field(
        default_factory=list, description="The operators of the package"
    )

    # 定义类方法 build_from，接受 values 和 ext_dict 两个字典参数
    @classmethod
    def build_from(cls, values: Dict[str, Any], ext_dict: Dict[str, Any]):
        # 导入 BaseOperator 类
        from dbgpt.core.awel import BaseOperator

        # 调用 load_module_class 方法加载模块类，获取模块信息
        values["operators"], _, _ = cls.load_module_class(values, BaseOperator)
        # 返回 OperatorPackage 对象
        return cls(**values)


# 定义 AgentPackage 类，继承自 BasePackage 类
class AgentPackage(BasePackage):
    # 定义 package_type 属性为字符串 "agent"
    package_type: str = "agent"

    # 定义 agents 属性为类型列表，默认为空列表，描述为 "The agents of the package"
    agents: List[type] = Field(
        default_factory=list, description="The agents of the package"
    )

    # 定义类方法 build_from，接受 values 和 ext_dict 两个字典参数
    @classmethod
    def build_from(cls, values: Dict[str, Any], ext_dict: Dict[str, Any]):
        # 导入 ConversableAgent 类
        from dbgpt.agent import ConversableAgent

        # 调用 load_module_class 方法加载模块类，获取模块信息
        values["agents"], _, _ = cls.load_module_class(values, ConversableAgent)
        # 返回 AgentPackage 对象
        return cls(**values)


# 定义 ResourcePackage 类，继承自 BasePackage 类
class ResourcePackage(BasePackage):
    # 定义 package_type 属性为字符串 "resource"
    package_type: str = "resource"

    # 定义 resources 属性为类型列表，默认为空列表，描述为 "The resources of the package"
    resources: List[type] = Field(
        default_factory=list, description="The resources of the package"
    )
    # 定义 resource_instances 属性为任意类型列表，默认为空列表，描述为 "The resource instances of the package"
    resource_instances: List[Any] = Field(
        default_factory=list, description="The resource instances of the package"
    )

    # 定义类方法 build_from，接受 values 和 ext_dict 两个字典参数
    @classmethod
    # 定义一个类方法 build_from，接受三个参数：cls（类本身）、values（字典类型，包含各种属性值）、ext_dict（另一个字典类型参数）
    def build_from(cls, values: Dict[str, Any], ext_dict: Dict[str, Any]):
        # 导入必要的模块和函数
        from dbgpt.agent.resource import Resource
        from dbgpt.agent.resource.tool.pack import _is_function_tool

        # 定义内部函数 _predicate，用于判断对象是否符合特定条件
        def _predicate(obj):
            # 如果对象为空，则返回 False
            if not obj:
                return False
            # 如果对象是函数工具，则返回 True
            elif _is_function_tool(obj):
                return True
            # 如果对象是 Resource 类的实例，则返回 True
            elif isinstance(obj, Resource):
                return True
            # 如果对象是 Resource 类或其子类的类型，则返回 True
            elif isinstance(obj, type) and issubclass(obj, Resource):
                return True
            else:
                return False

        # 调用类方法 load_module_class，加载指定的类并返回结果
        _, predicted_cls, _ = cls.load_module_class(values, Resource, [_predicate])
        
        # 初始化两个空列表，用于存储资源实例和资源类
        resource_instances = []
        resources = []
        
        # 遍历 predicted_cls 列表中的每个对象 o
        for o in predicted_cls:
            # 如果 o 是函数工具或者是 Resource 类的实例，则将其添加到 resource_instances 列表中
            if _is_function_tool(o) or isinstance(o, Resource):
                resource_instances.append(o)
            # 如果 o 是 Resource 类或其子类的类型，则将其添加到 resources 列表中
            elif isinstance(o, type) and issubclass(o, Resource):
                resources.append(o)
        
        # 将 resource_instances 和 resources 列表赋值给 values 字典的对应键
        values["resource_instances"] = resource_instances
        values["resources"] = resources
        
        # 使用类的构造函数（__init__ 方法）以 values 字典中的参数初始化 cls 类的新实例，并返回该实例
        return cls(**values)
class InstalledPackage(BaseModel):
    name: str = Field(..., description="The name of the package")
    repo: str = Field(..., description="The repository of the package")
    root: str = Field(..., description="The root of the package")
    package: str = Field(..., description="The package name(like name in pypi)")


def _get_classes_from_module(module):
    # 获取指定模块中的所有类对象并返回列表
    classes = [
        obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
    ]
    return classes


def _get_from_module(module, predicates: Optional[List[str]] = None):
    # 根据给定的谓词列表，获取指定模块中符合条件的对象列表
    if not predicates:
        return []
    results = []
    for predicate in predicates:
        for name, obj in inspect.getmembers(module, predicate):
            if obj.__module__ == module.__name__:
                results.append(obj)
    return results


def _parse_package_metadata(package: InstalledPackage) -> BasePackage:
    # 解析安装包的元数据，并返回相应的 BasePackage 对象
    with open(
        Path(package.root) / DBGPTS_METADATA_FILE, mode="r+", encoding="utf-8"
    ) as f:
        metadata = tomlkit.loads(f.read())
    ext_metadata = {}
    pkg_dict = {}
    for key, value in metadata.items():
        if key == "flow":
            pkg_dict = {k: v for k, v in value.items()}
            pkg_dict["package_type"] = "flow"
        elif key == "operator":
            pkg_dict = {k: v for k, v in value.items()}
            pkg_dict["package_type"] = "operator"
        elif key == "agent":
            pkg_dict = {k: v for k, v in value.items()}
            pkg_dict["package_type"] = "agent"
        elif key == "resource":
            pkg_dict = {k: v for k, v in value.items()}
            pkg_dict["package_type"] = "resource"
        else:
            ext_metadata[key] = value
    pkg_dict["root"] = package.root
    pkg_dict["repo"] = package.repo
    pkg_dict["package"] = package.package
    if pkg_dict["package_type"] == "flow":
        return FlowPackage.build_from(pkg_dict, ext_metadata)
    elif pkg_dict["package_type"] == "operator":
        return OperatorPackage.build_from(pkg_dict, ext_metadata)
    elif pkg_dict["package_type"] == "agent":
        return AgentPackage.build_from(pkg_dict, ext_metadata)
    elif pkg_dict["package_type"] == "resource":
        return ResourcePackage.build_from(pkg_dict, ext_metadata)
    else:
        raise ValueError(
            f"Unsupported package package_type: {pkg_dict['package_type']}"
        )


def _load_installed_package(path: str) -> List[InstalledPackage]:
    # 加载安装包的信息，并返回一个包含 InstalledPackage 对象的列表
    packages = []
    # 遍历指定路径下的所有文件和文件夹
    for package in os.listdir(path):
        # 构建完整的文件或文件夹路径
        full_path = Path(path) / package
        # 构建安装元数据文件的路径
        install_metadata_file = full_path / INSTALL_METADATA_FILE
        # 构建调试点元数据文件的路径
        dbgpts_metadata_file = full_path / DBGPTS_METADATA_FILE
        
        # 检查路径是否是一个目录，并且安装元数据文件和调试点元数据文件是否存在
        if (
            full_path.is_dir()
            and install_metadata_file.exists()
            and dbgpts_metadata_file.exists()
        ):
            # 打开安装元数据文件并加载其中的内容为 TOML 格式
            with open(install_metadata_file) as f:
                metadata = tomlkit.loads(f.read())
                # 从元数据中获取名称和仓库信息
                name = metadata["name"]
                repo = metadata["repo"]
                # 将已安装的包信息添加到列表中
                packages.append(
                    InstalledPackage(
                        name=name, repo=repo, root=str(full_path), package=package
                    )
                )
    
    # 返回包含已安装包信息的列表
    return packages
# 从指定路径加载包
def _load_package_from_path(path: str):
    # 调用函数_load_installed_package加载安装在指定路径下的所有包
    packages = _load_installed_package(path)
    # 对每个加载的包，解析其元数据并存储在parsed_packages列表中
    parsed_packages = []
    for package in packages:
        parsed_packages.append(_parse_package_metadata(package))
    # 返回解析后的包列表
    return parsed_packages


# 从指定路径加载流程包，并返回FlowPackage对象
def _load_flow_package_from_path(name: str, path: str = INSTALL_DIR) -> FlowPackage:
    # 加载安装在指定路径下的所有包
    raw_packages = _load_installed_package(path)
    # 将name中的下划线替换为破折号
    new_name = name.replace("_", "-")
    # 筛选出与name或new_name匹配的包
    packages = [p for p in raw_packages if p.package == name or p.name == name]
    # 如果未找到匹配的包，则再次尝试匹配new_name
    if not packages:
        packages = [
            p for p in raw_packages if p.package == new_name or p.name == new_name
        ]
    # 如果仍未找到匹配的包，则抛出异常
    if not packages:
        raise ValueError(f"Can't find the package {name} or {new_name}")
    # 解析找到的第一个包的元数据
    flow_package = _parse_package_metadata(packages[0])
    # 如果包的类型不是"flow"，则抛出异常
    if flow_package.package_type != "flow":
        raise ValueError(f"Unsupported package type: {flow_package.package_type}")
    # 返回解析后的FlowPackage对象
    return cast(FlowPackage, flow_package)


# 将FlowPackage对象转换为FlowPanel对象
def _flow_package_to_flow_panel(package: FlowPackage) -> FlowPanel:
    # 创建包含package信息的字典dict_value
    dict_value = {
        "name": package.name,
        "label": package.label,
        "version": package.version,
        "editable": False,
        "description": package.description,
        "source": package.repo,
        "define_type": "json",
    }
    # 根据package类型添加额外的信息到dict_value
    if isinstance(package, FlowJsonPackage):
        dict_value["flow_data"] = package.read_definition_json()
    elif isinstance(package, FlowPythonPackage):
        dict_value["flow_data"] = {
            "nodes": [],
            "edges": [],
            "viewport": {
                "x": 213,
                "y": 269,
                "zoom": 0,
            },
        }
        dict_value["flow_dag"] = package.dag
        dict_value["define_type"] = "python"
    else:
        # 如果package类型不支持，则抛出异常
        raise ValueError(f"Unsupported package type: {package}")
    # 使用dict_value创建并返回FlowPanel对象
    return FlowPanel(**dict_value)


class DBGPTsLoader(BaseComponent):
    """The loader of the dbgpts packages"""

    name: str = "dbgpt_dbgpts_loader"

    def __init__(
        self,
        system_app: Optional[SystemApp] = None,
        install_dir: Optional[str] = None,
        load_dbgpts_interval: int = 10,
    ):
        """Initialize the DBGPTsLoader."""
        # 初始化DBGPTsLoader对象
        self._system_app = None
        self._install_dir = install_dir or INSTALL_DIR
        self._packages: Dict[str, BasePackage] = {}
        self._load_dbgpts_interval = load_dbgpts_interval
        # 调用父类BaseComponent的构造函数
        super().__init__(system_app)

    def init_app(self, system_app: SystemApp):
        """Initialize the DBGPTsLoader."""
        # 初始化应用程序相关的参数
        self._system_app = system_app

    def before_start(self):
        """Execute after the application starts."""
        # 在应用程序启动后加载包，第一次加载时标记为is_first=True
        self.load_package(is_first=True)

        # 使用schedule模块定期调用self.load_package加载包
        schedule.every(self._load_dbgpts_interval).seconds.do(self.load_package)
    # 加载给定目录下的包
    def load_package(self, is_first: bool = False) -> None:
        """Load the package by name."""
        try:
            # 从指定路径加载包列表
            packages = _load_package_from_path(self._install_dir)
            # 如果是第一次加载，记录找到的包数量和目录路径
            if is_first:
                logger.info(
                    f"Found {len(packages)} dbgpts packages from {self._install_dir}"
                )
            # 将加载的每个包添加到实例的包字典中，并注册这些包
            for package in packages:
                self._packages[package.name] = package
                self._register_packages(package)
        except Exception as e:
            # 如果加载包过程中出现异常，记录警告信息
            logger.warning(f"Load dbgpts package error: {e}")

    # 获取所有流程面板
    def get_flows(self) -> List[FlowPanel]:
        """Get the flows.

        Returns:
            List[FlowPanel]: The list of the flows
        """
        panels = []
        # 遍历实例中的每个包
        for package in self._packages.values():
            # 如果包的类型不是流程类型，则跳过
            if package.package_type != "flow":
                continue
            # 将包强制转换为流程包类型
            package = cast(FlowPackage, package)
            # 将流程包转换为流程面板对象，并添加到结果列表中
            flow_panel = _flow_package_to_flow_panel(package)
            panels.append(flow_panel)
        return panels

    # 注册给定包的各种组件
    def _register_packages(self, package: BasePackage):
        # 如果包的类型是代理类型
        if package.package_type == "agent":
            # 导入必要的代理相关模块
            from dbgpt.agent import ConversableAgent, get_agent_manager
            
            # 获取代理管理器实例
            agent_manager = get_agent_manager(self._system_app)
            # 将包强制转换为代理包类型
            pkg = cast(AgentPackage, package)
            # 遍历代理包中的每个代理类
            for agent_cls in pkg.agents:
                # 如果代理类是 ConversableAgent 的子类
                if issubclass(agent_cls, ConversableAgent):
                    try:
                        # 注册代理类到代理管理器中，忽略重复注册的异常
                        agent_manager.register_agent(agent_cls, ignore_duplicate=True)
                    except ValueError as e:
                        # 记录注册代理出错的警告信息
                        logger.warning(f"Register agent {agent_cls} error: {e}")
        # 如果包的类型是资源类型
        elif package.package_type == "resource":
            # 导入必要的资源相关模块
            from dbgpt.agent.resource import Resource
            from dbgpt.agent.resource.manage import get_resource_manager
            
            # 将包强制转换为资源包类型
            pkg = cast(ResourcePackage, package)
            # 获取资源管理器实例
            rm = get_resource_manager(self._system_app)
            # 遍历资源包中的每个资源实例，注册到资源管理器中
            for inst in pkg.resource_instances:
                try:
                    rm.register_resource(resource_instance=inst, ignore_duplicate=True)
                except ValueError as e:
                    logger.warning(f"Register resource {inst} error: {e}")
            # 遍历资源包中的每个资源类，如果是 Resource 的子类，则注册到资源管理器中
            for res in pkg.resources:
                try:
                    if issubclass(res, Resource):
                        rm.register_resource(res, ignore_duplicate=True)
                except ValueError as e:
                    logger.warning(f"Register resource {res} error: {e}")
```