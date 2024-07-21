# `.\pytorch\torch\package\package_exporter.py`

```py
# mypy: allow-untyped-defs
# 导入所需的模块
import collections                       # 导入collections模块，提供额外的数据结构和操作
import importlib.machinery               # 导入importlib.machinery模块，用于导入机制相关的操作
import io                                # 导入io模块，提供对I/O操作的支持
import linecache                         # 导入linecache模块，用于缓存和随机读取文本行
import pickletools                       # 导入pickletools模块，用于分析Python的pickle二进制数据格式
import platform                          # 导入platform模块，提供对平台相关信息的访问
import types                             # 导入types模块，支持动态创建和操作Python类型
from collections import defaultdict, OrderedDict  # 导入defaultdict和OrderedDict类，提供额外的数据结构
from dataclasses import dataclass        # 导入dataclass装饰器，用于快速定义数据类
from enum import Enum                    # 导入Enum类，支持枚举类型
from importlib.machinery import SourceFileLoader  # 导入SourceFileLoader类，用于动态加载源文件
from pathlib import Path                 # 导入Path类，提供处理文件路径的类
from typing import (                     # 导入多种类型定义，包括基本类型、序列和字典等
    Any,
    BinaryIO,
    Callable,
    cast,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import torch                             # 导入torch模块，深度学习框架PyTorch的主要模块
from torch.serialization import location_tag, normalize_storage_type  # 导入serialization子模块的函数
from torch.types import Storage          # 导入Storage类型，用于表示PyTorch张量的存储类型
from torch.utils.hooks import RemovableHandle  # 导入RemovableHandle类，用于注册和移除钩子函数

from ._digraph import DiGraph            # 从当前包导入_DiGraph模块的DiGraph类
from ._importlib import _normalize_path  # 从当前包导入_importlib模块的_normalize_path函数
from ._mangling import demangle, is_mangled  # 从当前包导入_mangling模块的demangle和is_mangled函数
from ._package_pickler import create_pickler  # 从当前包导入_package_pickler模块的create_pickler函数
from ._stdlib import is_stdlib_module    # 从当前包导入_stdlib模块的is_stdlib_module函数
from .find_file_dependencies import find_files_source_depends_on  # 从当前包导入find_file_dependencies模块的函数
from .glob_group import GlobGroup, GlobPattern  # 从当前包导入glob_group模块的GlobGroup和GlobPattern类
from .importer import Importer, OrderedImporter, sys_importer  # 从当前包导入importer模块的Importer、OrderedImporter和sys_importer对象

__all__ = [                              # 设置模块的公开接口列表
    "PackagingErrorReason",
    "EmptyMatchError",
    "PackagingError",
    "PackageExporter",
]

_gate_torchscript_serialization = True   # 设置torchscript序列化门限为True

ActionHook = Callable[["PackageExporter", str], None]  # 定义ActionHook类型别名，表示接受PackageExporter和str参数的回调函数


class _ModuleProviderAction(Enum):       # 定义_ModuleProviderAction枚举类，表示PackageExporter对模块可能执行的操作
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """
    INTERN = 1                           # 内部化模块
    EXTERN = 2                           # 外部化模块
    MOCK = 3                             # 模拟模块
    DENY = 4                             # 拒绝模块
    # Special case: when a module is mocked, PackageExporter writes out a
    # `_mock` module that implements our mocking stubs. If we re-package code,
    # we may encounter a `_mock` module from the original package. If we do,
    # just ignore it and write a `_mock` module once.
    REPACKAGED_MOCK_MODULE = 5           # 重新打包的模拟模块
    # Special case: PackageImporter adds a fake module
    # (`torch_package_importer`) that allows packaged code to access it. Don't
    # re-export this.
    SKIP = 6                             # 跳过模块


class PackagingErrorReason(Enum):        # 定义PackagingErrorReason枚举类，描述打包失败的不同原因
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """

    def __repr__(self):                  # 定义__repr__方法，返回枚举实例的字符串表示形式
        return f"<{self.__class__.__name__}.{self.name}>"

    IS_EXTENSION_MODULE = (              # 模块是C扩展模块，torch.package仅支持Python模块
        "Module is a C extension module. torch.package supports Python modules only."
    )
    NO_DUNDER_FILE = "Module had no __file__ defined."  # 模块未定义__file__属性
    SOURCE_FILE_NOT_FOUND = (            # 找不到模块的源文件
        "Module had a __file__, but we could not find it in your filesystem."
    )
    DEPENDENCY_RESOLUTION_FAILED = "Dependency resolution failed."  # 依赖解析失败
    NO_ACTION = (                        # 模块未匹配任何动作模式
        "Module did not match against any action pattern. Extern, mock, or intern it."
    )
    DENIED = "Module was denied by a pattern."  # 模块被模式拒绝
    # 定义一个常量字符串，指出模块被模拟但仍在包中使用的情况。
    # 提示开发人员应当考虑将被模拟的模块内部化或外部化，确保对象在包中的正确位置。
    MOCKED_BUT_STILL_USED = (
        "Module was mocked out, but is still being used in the package. "
        "Please intern or extern the mocked modules if objects are supposed to be in "
        "the package."
    )
# 定义一个数据类 `_PatternInfo`，用于存储关于如何执行匹配的具体信息，这些信息是针对 `PackageExporter` 的特定设置

@dataclass
class _PatternInfo:
    """Holds :class:`PackageExporter`-specific info about how to execute matches against"""

    # 描述对于匹配此模式的模块应该执行的操作
    action: _ModuleProviderAction
    # 用户在指定模式时提供的 `allow_empty` 值
    allow_empty: bool
    # 表示在打包过程中是否已匹配到此模式
    was_matched: bool

    def __init__(self, action, allow_empty):
        # 初始化 `_PatternInfo` 实例时设置初始值
        self.action = action
        self.allow_empty = allow_empty
        self.was_matched = False


class EmptyMatchError(Exception):
    """This is an exception that is thrown when a mock or extern is marked as
    ``allow_empty=False``, and is not matched with any module during packaging.
    """

    pass


class PackagingError(Exception):
    """This exception is raised when there is an issue with exporting a package.
    ``PackageExporter`` will attempt to gather up all the errors and present
    them to you at once.
    """
    # 初始化函数，接受一个依赖图和一个调试标志（默认为 False）
    def __init__(self, dependency_graph: DiGraph, debug=False):
        # 使用 defaultdict 创建一个字典，将错误按原因分组
        broken: Dict[PackagingErrorReason, List[str]] = defaultdict(list)
        
        # 遍历依赖图中的每个模块及其属性
        for module_name, attrs in dependency_graph.nodes.items():
            # 获取模块可能存在的错误信息
            error = attrs.get("error")
            # 如果模块没有错误信息，跳过当前模块
            if error is None:
                continue
            # 如果错误为 NO_ACTION，则检查模块属性中不应该存在动作信息
            if error == PackagingErrorReason.NO_ACTION:
                assert "action" not in attrs
            # 将模块按其错误原因分组存储
            broken[error].append(module_name)

        # 创建一个 StringIO 对象来构建错误消息
        message = io.StringIO()
        message.write("\n")

        # 遍历所有的错误原因及其对应的模块列表
        for reason, module_names in broken.items():
            # 将错误原因作为标题写入消息
            message.write(f"* {reason.value}\n")
            # 遍历每个错误原因下的模块名称，写入消息
            for module_name in module_names:
                message.write(f"    {module_name}\n")

                # 如果模块存在错误上下文信息，则将其添加到消息中
                error_context = dependency_graph.nodes[module_name].get("error_context")
                if error_context is not None:
                    message.write(f"      Context: {error_context}\n")
                
                # 如果模块名称在 _DISALLOWED_MODULES 中，写入关于安全风险的建议信息
                if module_name in _DISALLOWED_MODULES:
                    message.write(
                        "      Note: While we usually use modules in the python standard library "
                        f"from the local environment, `{module_name}` has a lot of system "
                        "level access and therefore can pose a security risk. We heavily "
                        f"recommend removing `{module_name}` from your packaged code. However, if that "
                        "is not possible, add it to the extern list by calling "
                        f'PackageExporter.extern("`{module_name}`")\n'
                    )
                
                # 如果启用了调试模式，获取模块的第一个路径，并将其添加到消息中
                if debug:
                    module_path = dependency_graph.first_path(module_name)
                    message.write(
                        f"      A path to {module_name}: {' -> '.join(module_path)}\n"
                    )
        
        # 如果未启用调试模式，向消息中添加一条提示信息
        if not debug:
            message.write("\n")
            message.write(
                "Set debug=True when invoking PackageExporter for a visualization of where "
                "broken modules are coming from!\n"
            )
        
        # 将依赖图保存到当前对象中，以便工具可以访问它
        self.dependency_graph = dependency_graph
        
        # 调用父类的初始化函数，将消息内容作为异常信息传递
        super().__init__(message.getvalue())
# PackageExporter 类定义了一个导出器，用于将代码包、序列化的 Python 数据、二进制和文本资源写入自包含的包中。
class PackageExporter:

    """Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    """

    """A importer that will be searched in order to find the modules referenced by other modules or by
    pickled objects. The default module environment just uses sys_importer, which searches the Python environment.
    """
    # importer 属性用于指定导入器对象，以便在导入模块或反序列化对象时进行查找。
    importer: Importer

    # 初始化方法，用于创建 PackageExporter 的实例。
    def __init__(
        self,
        f: Union[str, Path, BinaryIO],
        importer: Union[Importer, Sequence[Importer]] = sys_importer,
        debug: bool = False,
        """
        Create an exporter.

        Args:
            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename
                or a binary I/O object.
            importer: If a single Importer is passed, use that to search for modules.
                If a sequence of importers are passed, an ``OrderedImporter`` will be constructed out of them.
            debug: If set to True, add path of broken modules to PackagingErrors.
        """
        # 记录 API 使用日志
        torch._C._log_api_usage_once("torch.package.PackageExporter")
        self.debug = debug  # 设置调试模式

        # 根据 f 的类型进行处理，如果是字符串或 Path 对象则转换为字符串，buffer 初始化为 None
        if isinstance(f, (Path, str)):
            f = str(f)
            self.buffer: Optional[BinaryIO] = None
        else:  # 如果是字节流对象
            self.buffer = f  # 直接使用传入的字节流对象

        # 创建 PyTorchFileWriter 对象来写入文件
        self.zip_file = torch._C.PyTorchFileWriter(f)
        self.zip_file.set_min_version(6)  # 设置写入文件的最小版本号
        self._written_files: Set[str] = set()  # 初始化已写入文件的集合

        self.serialized_reduces: Dict[int, Any] = {}  # 初始化用于序列化的减少数据结构

        # 用于跟踪所有添加到包中的模块和 pickle 对象及它们之间的依赖关系的图形
        # - 每个节点是一个模块名称（或看起来像 '<foo.obj.pkl>' 的 pickle 名称）
        # - 每个有向边 (u, v) 表示 u 依赖于 v
        # - 节点可以包含描述如何将内容写入 zipfile 的元数据
        self.dependency_graph = DiGraph()

        # 创建 ScriptModuleSerializer 对象来序列化 ScriptModule 到 zipfile
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file)
        self.storage_context = self.script_module_serializer.storage_context()

        # 下面三个 OrderedDict 用于兼容 RemovableHandle
        # 在 Python 3.6 中没有通用的 OrderedDict 类型注释，实际类型是 OrderedDict[int, Callable[[PackageExporter, str], None]]
        self._extern_hooks: OrderedDict = OrderedDict()  # 外部钩子函数的 OrderedDict
        self._mock_hooks: OrderedDict = OrderedDict()  # 模拟钩子函数的 OrderedDict
        self._intern_hooks: OrderedDict = OrderedDict()  # 内部钩子函数的 OrderedDict

        # 根据 importer 的类型初始化 self.importer
        if isinstance(importer, Importer):
            self.importer = importer  # 如果是单个 Importer 对象，则直接使用
        else:
            if not isinstance(importer, collections.abc.Sequence):
                raise TypeError(
                    "importer arg should be an Importer or a sequence of Importers, "
                    f"got {type(importer)} instead."
                )
            self.importer = OrderedImporter(*importer)  # 如果是 Importer 序列，则构建 OrderedImporter 对象

        self.patterns: Dict[GlobGroup, _PatternInfo] = {}  # 用于存储 glob 模式和对应信息的字典
        self._unique_id = 0  # 初始化唯一 ID
    ):
        """
        将本地文件系统中的文件或目录 `file_or_directory` 添加到源代码包中，以提供 `module_name` 的代码。

        Args:
            module_name (str): 例如 `"my_package.my_subpackage"`，将保存代码以提供此包的代码。
            file_or_directory (str): 文件或代码目录的路径。如果是目录，则使用 `save_source_file` 递归复制目录中的所有 Python 文件。
                如果文件名为 `"/__init__.py"`，则视为包。
            dependencies (bool, optional): 如果为 `True`，则扫描源代码以获取依赖项。

        """
        # 将路径转换为 Path 对象
        path = Path(file_or_directory)
        
        # 如果路径是目录
        if path.is_dir():
            to_save = []  # 保存 save_source_string 参数元组的列表
            module_path = module_name.replace(".", "/")
            
            # 遍历目录中的所有 Python 文件
            for filename in path.glob("**/*.py"):
                relative_path = filename.relative_to(path).as_posix()
                archivename = module_path + "/" + relative_path
                submodule_name = None
                
                # 判断是否是包的 __init__.py 文件
                if filename.name == "__init__.py":
                    submodule_name = archivename[: -len("/__init__.py")].replace(
                        "/", "."
                    )
                    is_package = True
                else:
                    submodule_name = archivename[: -len(".py")].replace("/", ".")
                    is_package = False

                # 延迟调用 save_source_string，以便记录此目录结构提供的所有源文件，
                # 在尝试解析源代码依赖之前。这确保我们不会尝试复制会被此目录 blob 覆盖的模块。
                to_save.append(
                    (
                        submodule_name,
                        _read_file(str(filename)),
                        is_package,
                        dependencies,
                    )
                )

            # 遍历保存的参数元组，调用 save_source_string 方法
            for item in to_save:
                self.save_source_string(*item)
        
        else:
            # 如果路径是文件而不是目录，则直接调用 save_source_string 方法
            is_package = path.name == "__init__.py"
            self.save_source_string(
                module_name,
                _read_file(file_or_directory),
                is_package,
                dependencies,
            )

    def get_unique_id(self) -> str:
        """
        获取一个唯一的 ID。此 ID 保证在此包中只被分配一次。
        """
        ret = str(self._unique_id)
        self._unique_id += 1
        return ret

    def _get_dependencies(
        self, src: str, module_name: str, is_package: bool
    ) -> List[str]:
        """
        Return all modules that this source code depends on.

        Dependencies are found by scanning the source code for import-like statements.

        Arguments:
            src: The Python source code to analyze for dependencies.
            module_name: The name of the module that ``src`` corresponds to.
            is_package: Whether this module should be treated as a package.
                See :py:meth:`save_source_string` for more info.

        Returns:
            A list containing modules detected as direct dependencies in
            ``src``.  The items in the list are guaranteed to be unique.
        """
        # Determine the package name based on whether the module is treated as a package or not
        package_name = (
            module_name if is_package else module_name.rsplit(".", maxsplit=1)[0]
        )
        
        try:
            # Attempt to find dependencies by calling an external function
            dep_pairs = find_files_source_depends_on(src, package_name)
        except Exception as e:
            # Handle exceptions by adding the module to the dependency graph with an error context
            self.dependency_graph.add_node(
                module_name,
                error=PackagingErrorReason.DEPENDENCY_RESOLUTION_FAILED,
                error_context=str(e),
            )
            # Return an empty list in case of failure to find dependencies
            return []

        # Use a dictionary to store dependencies for uniqueness and deterministic order
        dependencies = {}
        for dep_module_name, dep_module_obj in dep_pairs:
            # Check if dep_module_obj is not None, implying submodule situation
            if dep_module_obj is not None:
                possible_submodule = f"{dep_module_name}.{dep_module_obj}"
                # If the submodule exists, add it to dependencies and continue without saving the parent module
                if self._module_exists(possible_submodule):
                    dependencies[possible_submodule] = True
                    continue
            # If the module exists, add it to dependencies
            if self._module_exists(dep_module_name):
                dependencies[dep_module_name] = True

        # Convert dictionary keys to a list and return as the final list of dependencies
        return list(dependencies.keys())
    ):
        """
        将 ``src`` 作为 ``module_name`` 的源代码添加到导出包中。

        Args:
            module_name (str): 例如 ``my_package.my_subpackage``，代码将被保存以提供该包的代码。
            src (str): 要保存到此包的 Python 源代码。
            is_package (bool, optional): 如果为 ``True``，则此模块将被视为包。包可以有子模块（例如 ``my_package.my_subpackage.my_subsubpackage``），并且可以在其中保存资源。默认为 ``False``。
            dependencies (bool, optional): 如果为 ``True``，我们将扫描源代码以查找依赖项。
        """
        self.dependency_graph.add_node(
            module_name,
            source=src,
            is_package=is_package,
            provided=True,
            action=_ModuleProviderAction.INTERN,
        )

        if dependencies:
            deps = self._get_dependencies(src, module_name, is_package)

            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def _write_source_string(
        self,
        module_name: str,
        src: str,
        is_package: bool = False,
    ):
        """
        将 ``src`` 作为 ``module_name`` 的源代码写入 zip 存档中。

        参数与 :meth:`save_source_string` 相同。
        """
        extension = "/__init__.py" if is_package else ".py"
        filename = module_name.replace(".", "/") + extension

        self._write(filename, src)

    def _import_module(self, module_name: str):
        try:
            return self.importer.import_module(module_name)
        except ModuleNotFoundError as e:
            if not is_mangled(module_name):
                raise
            msg = (
                f"Module not found: '{module_name}'. Make sure the PackageImporter that "
                "created this module is present in `self.importer`"
            )
            raise ModuleNotFoundError(msg) from None

    def _module_exists(self, module_name: str) -> bool:
        try:
            self._import_module(module_name)
            return True
        except Exception:
            return False

    def _get_source_of_module(self, module: types.ModuleType) -> Optional[str]:
        filename = None
        spec = getattr(module, "__spec__", None)
        if spec is not None:
            loader = getattr(spec, "loader", None)
            if loader is not None and isinstance(loader, SourceFileLoader):
                try:
                    filename = loader.get_filename(module.__name__)
                except ImportError:
                    pass
        if filename is None:
            filename = getattr(module, "__file__", None)
        if isinstance(filename, str) and filename.endswith(".py"):
            return "".join(linecache.getlines(filename, module.__dict__))
        return None
    # 给定一个模块名，根据用户指定的模式将其添加到依赖图中。
    def add_dependency(self, module_name: str, dependencies=True):
        """Given a module, add it to the dependency graph according to patterns
        specified by the user.
        """
        # 如果模块名已存在于依赖图中，并且已经标记为“provided”，则直接返回，不做任何操作。
        if (
            module_name in self.dependency_graph
            and self.dependency_graph.nodes[module_name].get("provided") is True
        ):
            return

        # 特殊情况：PackageImporter 提供了一个名为 `torch_package_importer` 的特殊模块，
        # 允许打包的模块引用其 PackageImporter。我们不希望重新导出这个模块。
        if module_name == "torch_package_importer":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.SKIP,
                provided=True,
            )
            return

        # 特殊情况：如果模块名为 "_mock"，则将其添加到依赖图中，并标记为 REPACKAGED_MOCK_MODULE。
        if module_name == "_mock":
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.REPACKAGED_MOCK_MODULE,
                provided=True,
            )
            return

        # 如果可以隐式外部化该模块名，则将其添加到依赖图中，并标记为 EXTERN。
        if self._can_implicitly_extern(module_name):
            self.dependency_graph.add_node(
                module_name, action=_ModuleProviderAction.EXTERN, provided=True
            )
            return

        # 遍历用户定义的模式及其信息，尝试匹配模块名到相应的模式。
        for pattern, pattern_info in self.patterns.items():
            if pattern.matches(module_name):
                # 标记该模块名已匹配到某个模式。
                pattern_info.was_matched = True
                # 将该模块名添加到依赖图中，并使用模式指定的操作。
                self.dependency_graph.add_node(
                    module_name, action=pattern_info.action, provided=True
                )

                # 如果模式指定的操作是 DENY，则表示拒绝该模块，将其作为错误添加到依赖图中。
                if pattern_info.action == _ModuleProviderAction.DENY:
                    self.dependency_graph.add_node(
                        module_name, error=PackagingErrorReason.DENIED
                    )

                # 如果模式指定的操作是 INTERN，需要获取其依赖并一并打包。
                if pattern_info.action == _ModuleProviderAction.INTERN:
                    self._intern_module(module_name, dependencies)
                return

        # 如果没有任何模式匹配到该模块名，将其明确标记为错误。
        self.dependency_graph.add_node(
            module_name, error=PackagingErrorReason.NO_ACTION
        )
    def save_module(self, module_name: str, dependencies=True):
        """
        Save the code for ``module`` into the package. Code for the module is resolved using the ``importers`` path to find the
        module object, and then using its ``__file__`` attribute to find the source code.

        Args:
            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code
                for this package.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
        # 检查 module_name 是否为字符串，若不是则抛出类型错误
        if not isinstance(module_name, str):
            raise TypeError(
                "save_module() expects a string input, did you perhaps mean to pass `__name__`?"
            )

        # 调用 _intern_module 方法来内部保存模块
        self._intern_module(module_name, dependencies)

    def _intern_module(
        self,
        module_name: str,
        dependencies: bool,
    ):
        """
        Internal method to save the code for a specified module into the package.

        Args:
            module_name (str): Name of the module to be saved.
            dependencies (bool): Flag indicating whether to include dependencies.
        """
        # 在这里实现保存模块代码到包中的逻辑
    ):
        """Adds the module to the dependency graph as an interned module,
        along with any metadata needed to write it out to the zipfile at serialization time.
        """
        # 导入指定名称的模块并获取其对象
        module_obj = self._import_module(module_name)
        # 如果成功导入模块，将模块名称反混淆，确保在序列化时不保存混淆后的名称
        module_name = demangle(module_name)

        # 判断模块是否为包
        is_package = hasattr(module_obj, "__path__")
        # 获取模块的源代码
        source = self._get_source_of_module(module_obj)
        if source is None:
            # 若无法找到源代码，则将模块添加到依赖图中，并标记为错误
            filename = getattr(module_obj, "__file__", None)
            error_context = None
            if filename is None:
                packaging_error = PackagingErrorReason.NO_DUNDER_FILE
            elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                packaging_error = PackagingErrorReason.IS_EXTENSION_MODULE
            else:
                packaging_error = PackagingErrorReason.SOURCE_FILE_NOT_FOUND
                error_context = f"filename: {filename}"
            # 将模块添加到依赖图中，并返回
            self.dependency_graph.add_node(
                module_name,
                action=_ModuleProviderAction.INTERN,
                is_package=is_package,
                error=packaging_error,
                error_context=error_context,
                provided=True,
            )
            return

        # 将模块添加到依赖图中
        self.dependency_graph.add_node(
            module_name,
            action=_ModuleProviderAction.INTERN,
            is_package=is_package,
            source=source,
            provided=True,
        )

        # 若需要添加依赖模块
        if dependencies:
            # 获取模块的依赖关系并添加到依赖图中
            deps = self._get_dependencies(source, module_name, is_package)
            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def save_pickle(
        self,
        package: str,
        resource: str,
        obj: Any,
        dependencies: bool = True,
        pickle_protocol: int = 3,
    ):
        """Serialize and save an object using pickle to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            obj (Any): The object to serialize and save.
            dependencies (bool, optional): Whether to save dependencies as well. Defaults to True.
            pickle_protocol (int, optional): Pickle protocol version. Defaults to 3.
        """

    def save_text(self, package: str, resource: str, text: str):
        """Save text data to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            text (str): The contents to save.
        """
        # 将文本数据编码为 UTF-8 后保存到包中
        return self.save_binary(package, resource, text.encode("utf-8"))
    def save_binary(self, package, resource, binary: bytes):
        """Save raw bytes to the package.

        Args:
            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            binary (str): The data to save.
        """
        # 构建保存的文件名
        filename = self._filename(package, resource)
        # 调用内部方法进行写入操作
        self._write(filename, binary)

    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an extern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`extern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        # 创建一个可移除的处理句柄
        handle = RemovableHandle(self._extern_hooks)
        # 将外部钩子函数添加到钩子字典中
        self._extern_hooks[handle.id] = hook
        return handle

    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers a mock hook on the exporter.

        The hook will be called each time a module matches against a :meth:`mock` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        # 创建一个可移除的处理句柄
        handle = RemovableHandle(self._mock_hooks)
        # 将模拟钩子函数添加到钩子字典中
        self._mock_hooks[handle.id] = hook
        return handle

    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle:
        """Registers an intern hook on the exporter.

        The hook will be called each time a module matches against an :meth:`intern` pattern.
        It should have the following signature::

            hook(exporter: PackageExporter, module_name: str) -> None

        Hooks will be called in order of registration.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                A handle that can be used to remove the added hook by calling
                ``handle.remove()``.
        """
        # 创建一个可移除的处理句柄
        handle = RemovableHandle(self._intern_hooks)
        # 将内部钩子函数添加到钩子字典中
        self._intern_hooks[handle.id] = hook
        return handle

    def intern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
        recursive: bool = True
    ):
        """Registers an intern pattern on the exporter.

        Args:
            include (GlobPattern): Glob pattern or list of patterns to include.
            exclude (GlobPattern, optional): Glob pattern or list of patterns to exclude.
            allow_empty (bool, optional): Whether an empty directory is allowed.
            recursive (bool, optional): Whether to recursively include files from subdirectories.
        """
        # 省略了函数体，因此无需注释内部逻辑
    ):
        """
        指定应打包的模块。模块必须与某些 `intern` 模式匹配，以便包含在包中并递归处理其依赖关系。

        Args:
            include (Union[List[str], str]): 字符串，例如 "my_package.my_subpackage"，或模块名称列表，
                用于指定应打包的模块。也可以是一种类似于 glob 样式的模式，如 :meth:`mock` 中描述的那样。

            exclude (Union[List[str], str]): 可选参数，排除匹配 include 字符串的某些模式。

            allow_empty (bool): 可选标志，指定此调用到 `intern` 方法的内部模块是否在打包期间必须与某些模块匹配。
                如果使用 `allow_empty=False` 添加了一个 `intern` 模块 glob 模式，并且在匹配该模式之前调用了 :meth:`close`
                （显式调用或通过 `__exit__` 调用），则会抛出异常。如果 `allow_empty=True`，则不会抛出这样的异常。

        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.INTERN, allow_empty
        )

    def mock(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,
        """
        Replace some required modules with a mock implementation.  Mocked modules will return a fake
        object for any attribute accessed from it. Because we copy file-by-file, the dependency resolution will sometimes
        find files that are imported by model files but whose functionality is never used
        (e.g. custom serialization code or training helpers).
        Use this function to mock this functionality out without having to modify the original code.

        Args:
            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings
                for the names of the modules to be mocked out. Strings can also be a glob-style pattern
                string that may match multiple modules. Any required dependencies that match this pattern
                string will be mocked out automatically.

                Examples :
                    ``'torch.**'`` -- matches ``torch`` and all submodules of torch, e.g. ``'torch.nn'``
                    and ``'torch.nn.functional'``

                    ``'torch.*'`` -- matches ``'torch.nn'`` or ``'torch.functional'``, but not
                    ``'torch.nn.functional'``

            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.
                e.g. ``include='torch.**', exclude='torch.foo'`` will mock all torch packages except ``'torch.foo'``,
                Default: is ``[]``.

            allow_empty (bool): An optional flag that specifies whether the mock implementation(s) specified by this call
                to the :meth:`mock` method must be matched to some module during packaging. If a mock is added with
                ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``) and the mock has
                not been matched to a module used by the package being exported, an exception is thrown.
                If ``allow_empty=True``, no such exception is thrown.
        """
        # 将指定的模块替换为模拟实现，用于测试或移除未使用的功能
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.MOCK, allow_empty
        )



    def extern(
        self,
        include: "GlobPattern",
        *,
        exclude: "GlobPattern" = (),
        allow_empty: bool = True,


注释：
    ):
        """
        将模块 ``module`` 加入到包可以导入的外部模块列表中。
        这将阻止依赖发现将其保存在包中。导入程序将直接从标准导入系统中加载外部模块。
        外部模块的代码也必须存在于加载包的进程中。

        Args:
            include (Union[List[str], str]): 字符串，例如 ``"my_package.my_subpackage"``，或字符串列表，表示要外部化的模块名称。
                这也可以是一个像在 :meth:`mock` 中描述的 glob 样式模式。

            exclude (Union[List[str], str]): 可选模式，用于排除与包含字符串匹配的某些模式。

            allow_empty (bool): 可选标志，指定此调用到 ``extern`` 方法的外部模块是否必须在打包过程中与某些模块匹配。
                如果以 ``allow_empty=False`` 添加外部模块 glob 模式，并且在任何模块匹配该模式之前调用了 :meth:`close`
                （无论是显式调用还是通过 ``__exit__``），都会引发异常。如果 ``allow_empty=True``，则不会引发这样的异常。
        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.EXTERN, allow_empty
        )

    def deny(self, include: "GlobPattern", *, exclude: "GlobPattern" = ()):
        """
        阻止模块名称与给定的 glob 模式匹配的模块从包可以导入的模块列表中。
        如果发现对任何匹配包的依赖关系，则引发 :class:`PackagingError`。

        Args:
            include (Union[List[str], str]): 字符串，例如 ``"my_package.my_subpackage"``，或字符串列表，表示要阻止导入的模块名称。
                这也可以是一个像在 :meth:`mock` 中描述的 glob 样式模式。

            exclude (Union[List[str], str]): 可选模式，用于排除与包含字符串匹配的某些模式。
        """
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(
            _ModuleProviderAction.DENY, allow_empty=True
        )
    # 定义一个方法，用于持久化标识化对象
    def _persistent_id(self, obj):
        # 检查对象是否为 Torch 的存储对象或其子类型
        if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
            # 声明一个存储对象
            storage: Storage
            # 如果对象是 TypedStorage 类型
            if isinstance(obj, torch.storage.TypedStorage):
                # 获取未标记类型的存储，获取存储类型字符串并转换为对应类型
                untyped_storage = obj._untyped_storage
                storage_type_str = obj.pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage = cast(Storage, untyped_storage)
                # 获取存储中元素的数量
                storage_numel = obj.size()

            # 如果对象是 UntypedStorage 类型
            elif isinstance(obj, torch.UntypedStorage):
                untyped_storage = obj
                storage = cast(Storage, untyped_storage)
                # 规范化存储类型并获取存储的字节数
                storage_type = normalize_storage_type(type(storage))
                storage_numel = storage.nbytes()

            else:
                # 抛出运行时错误，说明存储类型未识别
                raise RuntimeError(f"storage type not recognized: {type(obj)}")

            # 获取存储的位置信息标签
            location = location_tag(storage)

            # 检查存储是否已存在
            storage_present = self.storage_context.has_storage(storage)
            # 获取或添加存储的唯一标识符
            storage_id = self.storage_context.get_or_add_storage(storage)
            # 如果存储不存在，则进行序列化
            if not storage_present:
                # 如果存储设备不是 CPU，则将存储移到 CPU 上
                if storage.device.type != "cpu":
                    storage = storage.cpu()
                # 获取存储的字节数并写入 ZIP 文件
                num_bytes = storage.nbytes()
                self.zip_file.write_record(
                    f".data/{storage_id}.storage", storage, num_bytes
                )
            # 返回存储的标识化信息
            return ("storage", storage_type, storage_id, location, storage_numel)

        # 如果对象具有 "__reduce_package__" 属性
        if hasattr(obj, "__reduce_package__"):
            # 如果开启了 TorchScript 序列化，并且对象是 RecursiveScriptModule 类型
            if _gate_torchscript_serialization and isinstance(
                obj, torch.jit.RecursiveScriptModule
            ):
                # 抛出异常，说明直接序列化 ScriptModules 是试验性功能
                raise Exception(
                    "Serializing ScriptModules directly into a package is a beta feature. "
                    "To use, set global "
                    "`torch.package.package_exporter._gate_torchscript_serialization` to `False`."
                )
            # 如果对象的 ID 尚未被序列化过，则将其添加到序列化字典中
            if self.serialized_reduces.get(id(obj)) is None:
                self.serialized_reduces[id(obj)] = (
                    "reduce_package",
                    id(obj),
                    *obj.__reduce_package__(self),
                )
            # 返回对象的序列化信息
            return self.serialized_reduces[id(obj)]

        # 如果对象不符合以上条件，则返回空值
        return None

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        return self

    # 退出上下文管理器时调用的方法
    def __exit__(self, exc_type, exc_value, traceback):
        # 如果 __exit__ 被调用是因为有异常被抛出，则不尝试完成打包过程
        if exc_type is not None:
            # 尽量保证开放的缓冲区处于有效状态
            self._finalize_zip()
            return

        # 如果没有异常被抛出，则正常关闭打包过程
        self.close()
    # 将文件名和字符串或字节写入压缩文件中
    def _write(self, filename, str_or_bytes):
        # 检查文件名是否已经存在于已写入文件集合中，如果是则引发断言错误
        if filename in self._written_files:
            raise AssertionError(
                f"Tried to write file '{filename}', but it already exists in this archive. "
                "Please file a bug."
            )
        # 将文件名添加到已写入文件集合中
        self._written_files.add(filename)

        # 如果文件名被篡改（mangled），则抛出断言错误
        if is_mangled(filename):
            raise AssertionError(
                f"Tried to save a torch.package'd module as '{filename}'. "
                "Directly saving torch.package'd modules is not allowed."
            )
        
        # 如果输入的数据是字符串，则将其编码为 UTF-8 字节
        if isinstance(str_or_bytes, str):
            str_or_bytes = str_or_bytes.encode("utf-8")
        
        # 将文件名和对应的字节数据写入到压缩文件中
        self.zip_file.write_record(filename, str_or_bytes, len(str_or_bytes))

    # 验证依赖图的有效性
    def _validate_dependency_graph(self):
        # 1. 检查依赖图中是否存在通过依赖分析插入的错误
        for attrs in self.dependency_graph.nodes.values():
            if "error" in attrs:
                raise PackagingError(self.dependency_graph, debug=self.debug)

        # 2. 检查所有标记为 allow_empty=False 的模式是否至少匹配了一次
        for pattern, pattern_info in self.patterns.items():
            if not pattern_info.allow_empty and not pattern_info.was_matched:
                raise EmptyMatchError(
                    f"Exporter did not match any modules to {pattern}, which was marked as allow_empty=False"
                )

    # 写入模拟文件（mock file）
    def _write_mock_file(self):
        # 如果 "_mock.py" 文件尚未写入到压缩文件中
        if "_mock.py" not in self._written_files:
            # 构造 "_mock.py" 文件的路径
            mock_file = str(Path(__file__).parent / "_mock.py")
            # 读取 "_mock.py" 文件内容并写入压缩文件，标记为非包文件
            self._write_source_string("_mock", _read_file(mock_file), is_package=False)
    def _execute_dependency_graph(self):
        """执行依赖图，描述如何打包所有模块并写入到 ZIP 归档中。"""
        # 验证依赖图的有效性
        self._validate_dependency_graph()

        # 存储外部模块的列表
        extern_modules = []
        for module_name, attrs in self.dependency_graph.nodes.items():
            action = attrs["action"]

            if action == _ModuleProviderAction.EXTERN:
                # 执行所有外部模块的钩子函数
                for hook in self._extern_hooks.values():
                    hook(self, module_name)

                # 将当前模块名加入到外部模块列表中
                extern_modules.append(module_name)

            elif action == _ModuleProviderAction.MOCK:
                # 执行所有模拟模块的钩子函数
                for hook in self._mock_hooks.values():
                    hook(self, module_name)

                # 写入模拟文件
                self._write_mock_file()

                # 检查当前模块是否是一个包
                is_package = hasattr(self._import_module(module_name), "__path__")
                # 写入源代码字符串到 ZIP 中
                self._write_source_string(module_name, _MOCK_IMPL, is_package)

            elif action == _ModuleProviderAction.INTERN:
                # 执行所有内部模块的钩子函数
                for hook in self._intern_hooks.values():
                    hook(self, module_name)

                # 依赖图节点包含的元数据告诉我们如何内部化该模块
                if "provided" not in attrs:
                    raise AssertionError(
                        f"Module was marked `intern` but not provided: {module_name}"
                    )

                # 如果模块标记为 pickle，则无需为其编写任何源代码
                if attrs.get("is_pickle") is True:
                    continue

                # 检查当前模块是否是一个包
                is_package = attrs["is_package"]
                # 写入源代码字符串到 ZIP 中
                source = attrs["source"]
                self._write_source_string(module_name, source, is_package)

            elif action == _ModuleProviderAction.REPACKAGED_MOCK_MODULE:
                # 写入模拟文件
                self._write_mock_file()
            elif action == _ModuleProviderAction.SKIP:
                # 跳过当前模块的处理
                continue
            else:
                # 抛出异常，指示无效的操作
                raise AssertionError(
                    f"Invalid action: {module_name}, {action}. Please report a bug to PyTorch."
                )

        # 将外部模块列表转换成字符串并写入到 ZIP 中
        extern_file_contents = "\n".join(extern_modules) + "\n"
        self._write(".data/extern_modules", extern_file_contents)

    def _write_python_version(self):
        """将创建该包的 Python 版本写入到 .data/python_version 文件中。"""
        # 写入 Python 版本信息到 ZIP 中
        self._write(".data/python_version", platform.python_version())

    def close(self):
        """将打包好的内容写入到文件系统中。在调用 :meth:`close` 之后的任何调用都将无效。
        建议使用资源管理语法进行代替::

            with PackageExporter("file.zip") as e:
                ...
        """
        # 执行依赖图的打包操作
        self._execute_dependency_graph()
        # 写入 Python 版本信息
        self._write_python_version()

        # 写入脚本模块的文件
        self.script_module_serializer.write_files()
        # 最终化 ZIP 文件
        self._finalize_zip()
    def _finalize_zip(self):
        """
        Called at the very end of packaging to leave the zipfile in a closed but valid state.
        """
        # 删除 self.zip_file 属性，关闭 zip 文件
        del self.zip_file
        # 如果存在缓冲区，刷新缓冲区
        if self.buffer:
            self.buffer.flush()

    def _filename(self, package, resource):
        """
        Constructs a filename from package and resource.

        Args:
            package (str): The package name.
            resource (str): The resource path.

        Returns:
            str: The constructed filename in the format "<package_path>/<resource>".
        """
        # 将包名中的点替换为斜杠，构造资源的标准路径
        package_path = package.replace(".", "/")
        # 规范化资源路径
        resource = _normalize_path(resource)
        # 返回构造的完整文件名
        return f"{package_path}/{resource}"

    def _can_implicitly_extern(self, module_name: str):
        """
        Determines if a module can be implicitly externed.

        Args:
            module_name (str): The name of the module.

        Returns:
            bool: True if the module can be implicitly externed, False otherwise.
        """
        # 获取顶级包的名称
        top_level_package_name = module_name.partition(".")[0]
        # 判断顶级包是否为 "torch" 或者在允许的模块中，并且是标准库模块
        return top_level_package_name == "torch" or (
            top_level_package_name not in _DISALLOWED_MODULES
            and is_stdlib_module(top_level_package_name)
        )

    def dependency_graph_string(self) -> str:
        """
        Returns digraph string representation of dependencies in package.

        Returns:
            str: A string representation of dependencies in the package.
        """
        # 返回依赖图的 DOT 格式字符串表示
        return self.dependency_graph.to_dot()

    def _nodes_with_action_type(
        self, action: Optional[_ModuleProviderAction]
    ) -> List[str]:
        """
        Returns nodes (modules) that match the given action type.

        Args:
            action (_ModuleProviderAction): The action type to filter nodes by.

        Returns:
            List[str]: A sorted list of module names that match the action type and are not pickled.
        """
        # 初始化结果列表
        result = []
        # 遍历依赖图中的所有节点
        for name, node_dict in self.dependency_graph.nodes.items():
            # 获取节点的动作类型
            node_action = node_dict.get("action", None)
            # 如果节点的动作类型与指定的 action 相同，并且不是 pickled 类型的节点，则加入结果列表
            if node_action == action and "is_pickle" not in node_dict:
                result.append(name)
        # 对结果列表进行排序
        result.sort()
        # 返回结果列表
        return result

    def externed_modules(self) -> List[str]:
        """
        Return all modules that are currently externed.

        Returns:
            List[str]: A list containing the names of modules which will be externed in this package.
        """
        # 返回所有当前被 externed 的模块列表
        return self._nodes_with_action_type(_ModuleProviderAction.EXTERN)

    def interned_modules(self) -> List[str]:
        """
        Return all modules that are currently interned.

        Returns:
            List[str]: A list containing the names of modules which will be interned in this package.
        """
        # 返回所有当前被 interned 的模块列表
        return self._nodes_with_action_type(_ModuleProviderAction.INTERN)

    def mocked_modules(self) -> List[str]:
        """
        Return all modules that are currently mocked.

        Returns:
            List[str]: A list containing the names of modules which will be mocked in this package.
        """
        # 返回所有当前被 mocked 的模块列表
        return self._nodes_with_action_type(_ModuleProviderAction.MOCK)

    def denied_modules(self) -> List[str]:
        """
        Return all modules that are currently denied.

        Returns:
            List[str]: A list containing the names of modules which will be denied in this package.
        """
        # 返回所有当前被 denied 的模块列表
        return self._nodes_with_action_type(_ModuleProviderAction.DENY)
    # 返回依赖于指定模块名的所有模块列表
    def get_rdeps(self, module_name: str) -> List[str]:
        """Return a list of all modules which depend on the module ``module_name``.

        Returns:
            A list containing the names of modules which depend on ``module_name``.
        """
        # 检查模块名是否存在于依赖图的前向关系键中
        if module_name in self.dependency_graph._pred.keys():
            # 返回依赖于给定模块名的所有模块的列表
            return list(self.dependency_graph._pred[module_name].keys())
        else:
            # 如果模块名不存在依赖图中，则返回空列表
            return []

    # 返回从源模块到目标模块的所有路径的 DOT 表示
    def all_paths(self, src: str, dst: str) -> str:
        """Return a dot representation of the subgraph
           that has all paths from src to dst.

        Returns:
            A dot representation containing all paths from src to dst.
            (https://graphviz.org/doc/info/lang.html)
        """
        # 调用依赖图对象的方法，获取从源模块到目标模块的所有路径的 DOT 表示
        return self.dependency_graph.all_paths(src, dst)
# 禁止自动导出的模块列表，这些模块提供了系统级别的访问权限
_DISALLOWED_MODULES = ["sys", "io"]

# 模拟实现的代码字符串，用于动态创建属性访问的模拟对象
# 使用 _mock 模块的 MockedObject 类进行属性访问的模拟
_MOCK_IMPL = """\
from _mock import MockedObject
def __getattr__(attr: str):
    return MockedObject(__name__ + '.' + attr, _suppress_err=True)
"""

# 读取文件内容并返回其解码后的字符串表示
def _read_file(filename: str) -> str:
    # 使用二进制模式打开文件
    with open(filename, "rb") as f:
        # 读取文件内容为字节串
        b = f.read()
        # 将字节串解码为 UTF-8 格式的字符串并返回
        return b.decode("utf-8")
```