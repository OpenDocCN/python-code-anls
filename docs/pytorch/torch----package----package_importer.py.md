# `.\pytorch\torch\package\package_importer.py`

```
# mypy: allow-untyped-defs
# 引入标准库模块和第三方库模块
import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os
import types
# 从 contextlib 模块中引入 contextmanager 装饰器
from contextlib import contextmanager
# 从 typing 模块中引入各种类型相关的类和装饰器
from typing import (
    Any,
    BinaryIO,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)
# 从 weakref 模块中引入 WeakValueDictionary 类
from weakref import WeakValueDictionary

# 引入 torch 库
import torch
# 从 torch.serialization 模块中引入 _get_restore_location 和 _maybe_decode_ascii 函数
from torch.serialization import _get_restore_location, _maybe_decode_ascii

# 从本地导入模块
from ._directory_reader import DirectoryReader
from ._importlib import (
    _calc___package__,
    _normalize_line_endings,
    _normalize_path,
    _resolve_name,
    _sanity_check,
)
from ._mangling import demangle, PackageMangler
from ._package_unpickler import PackageUnpickler
# 从 file_structure_representation 模块中导入 _create_directory_from_file_list 和 Directory 类
from .file_structure_representation import _create_directory_from_file_list, Directory
# 从 importer 模块中导入 Importer 类
from .importer import Importer

# 如果 TYPE_CHECKING 为真，则从 glob_group 模块中导入 GlobPattern 类
if TYPE_CHECKING:
    from .glob_group import GlobPattern

# 定义一个公开的列表，列出了一些隐式允许的导入项，即使它们没有被标记为 extern
# 这是为了解决 Torch 隐式依赖于 numpy 的问题，而 package 不能跟踪它们
IMPLICIT_IMPORT_ALLOWLIST: Iterable[str] = [
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    # FX GraphModule 可能依赖于 builtins 模块，通常用户不会将其标记为 extern，因此默认在此处导入它
    "builtins",
]


class PackageImporter(Importer):
    """Importers allow you to load code written to packages by :class:`PackageExporter`.
    Code is loaded in a hermetic way, using files from the package
    rather than the normal python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external during export.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.
    """

    """The dictionary of already loaded modules from this package, equivalent to ``sys.modules`` but
    local to this importer.
    """
    # 已从该包中加载的模块的字典，类似于 sys.modules，但是局限于此导入器内部
    modules: Dict[str, types.ModuleType]

    def __init__(
        self,
        file_or_buffer: Union[str, torch._C.PyTorchFileReader, os.PathLike, BinaryIO],
        # module_allowed 是一个函数，用于确定是否允许加载指定模块
        module_allowed: Callable[[str], bool] = lambda module_name: True,
    ):
        """
        打开用于导入的 ``file_or_buffer``。检查导入的包是否仅需要 ``module_allowed`` 允许的模块。

        Args:
            file_or_buffer: 文件样对象（必须实现 :meth:`read`, :meth:`readline`, :meth:`tell`, 和 :meth:`seek`），
                字符串，或包含文件名的 ``os.PathLike`` 对象。
            module_allowed (Callable[[str], bool], optional): 确定是否允许外部提供的模块的方法。可以用于确保加载的包不依赖服务器不支持的模块。默认允许任何模块。

        Raises:
            ImportError: 如果包使用了不允许的模块。
        """
        torch._C._log_api_usage_once("torch.package.PackageImporter")

        self.zip_reader: Any
        if isinstance(file_or_buffer, torch._C.PyTorchFileReader):
            self.filename = "<pytorch_file_reader>"
            self.zip_reader = file_or_buffer
        elif isinstance(file_or_buffer, (os.PathLike, str)):
            self.filename = os.fspath(file_or_buffer)
            if not os.path.isdir(self.filename):
                self.zip_reader = torch._C.PyTorchFileReader(self.filename)
            else:
                self.zip_reader = DirectoryReader(self.filename)
        else:
            self.filename = "<binary>"
            self.zip_reader = torch._C.PyTorchFileReader(file_or_buffer)

        torch._C._log_api_usage_metadata(
            "torch.package.PackageImporter.metadata",
            {
                "serialization_id": self.zip_reader.serialization_id(),
                "file_name": self.filename,
            },
        )

        self.root = _PackageNode(None)
        self.modules = {}
        self.extern_modules = self._read_extern()

        for extern_module in self.extern_modules:
            if not module_allowed(extern_module):
                raise ImportError(
                    f"package '{file_or_buffer}' needs the external module '{extern_module}' "
                    f"but that module has been disallowed"
                )
            self._add_extern(extern_module)

        for fname in self.zip_reader.get_all_records():
            self._add_file(fname)

        self.patched_builtins = builtins.__dict__.copy()
        self.patched_builtins["__import__"] = self.__import__
        # 允许打包的模块引用其 PackageImporter
        self.modules["torch_package_importer"] = self  # type: ignore[assignment]

        self._mangler = PackageMangler()

        # 用于 reduce 反序列化
        self.storage_context: Any = None
        self.last_map_location = None

        # 用于 torch.serialization._load
        self.Unpickler = lambda *args, **kwargs: PackageUnpickler(self, *args, **kwargs)
    def import_module(self, name: str, package=None):
        """Load a module from the package if it hasn't already been loaded, and then return
        the module. Modules are loaded locally
        to the importer and will appear in ``self.modules`` rather than ``sys.modules``.

        Args:
            name (str): Fully qualified name of the module to load.
            package ([type], optional): Unused, but present to match the signature of importlib.import_module. Defaults to ``None``.

        Returns:
            types.ModuleType: The (possibly already) loaded module.
        """
        # 将模块名进行解码，支持模块名的解码（例如解决模块名重命名问题）
        name = self._mangler.demangle(name)

        # 调用_gcd_import方法加载模块并返回
        return self._gcd_import(name)

    def load_binary(self, package: str, resource: str) -> bytes:
        """Load raw bytes.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.

        Returns:
            bytes: The loaded data.
        """

        # 获取资源在 ZIP 文件中的路径
        path = self._zipfile_path(package, resource)
        # 从 zip_reader 中读取指定路径的资源数据
        return self.zip_reader.get_record(path)

    def load_text(
        self,
        package: str,
        resource: str,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        """Load a string.

        Args:
            package (str): The name of module package (e.g. ``"my_package.my_subpackage"``).
            resource (str): The unique name for the resource.
            encoding (str, optional): Passed to ``decode``. Defaults to ``'utf-8'``.
            errors (str, optional): Passed to ``decode``. Defaults to ``'strict'``.

        Returns:
            str: The loaded text.
        """
        # 调用 load_binary 方法获取资源的原始字节数据
        data = self.load_binary(package, resource)
        # 将字节数据按指定编码解码为字符串并返回
        return data.decode(encoding, errors)

    def id(self):
        """
        Returns internal identifier that torch.package uses to distinguish :class:`PackageImporter` instances.
        Looks like::

            <torch_package_0>
        """
        # 返回包含实例标识符的字符串，使用父模块名标识
        return self._mangler.parent_name()

    def file_structure(
        self, *, include: "GlobPattern" = "**", exclude: "GlobPattern" = ()
    ) -> Directory:
        """
        Returns a file structure representation of package's zipfile.

        Args:
            include (Union[List[str], str]): An optional string e.g. ``"my_package.my_subpackage"``, or optional list of strings
                for the names of the files to be included in the zipfile representation. This can also be
                a glob-style pattern, as described in :meth:`PackageExporter.mock`
            exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.

        Returns:
            :class:`Directory`
        """
        # 调用内部方法，基于当前实例的文件名和所有记录创建目录结构
        return _create_directory_from_file_list(
            self.filename, self.zip_reader.get_all_records(), include, exclude
        )

    def python_version(self):
        """
        Returns the version of python that was used to create this package.

        Note: this function is experimental and not Forward Compatible. The plan is to move this into a lock
        file later on.

        Returns:
            :class:`Optional[str]` a python version e.g. 3.8.9 or None if no version was stored with this package
        """
        # 定义存储Python版本信息的路径
        python_version_path = ".data/python_version"
        # 如果存储路径存在记录，则解码并返回Python版本号
        return (
            self.zip_reader.get_record(python_version_path).decode("utf-8").strip()
            if self.zip_reader.has_record(python_version_path)
            else None
        )

    def _read_extern(self):
        # 读取存储在ZIP文件中的外部模块信息，以列表形式返回
        return (
            self.zip_reader.get_record(".data/extern_modules")
            .decode("utf-8")
            .splitlines(keepends=False)
        )

    def _make_module(
        self, name: str, filename: Optional[str], is_package: bool, parent: str
    ):
        # 内部方法，用于创建模块对象，接受模块名称、文件名、是否为包、父模块名称作为参数
    ):
        # 如果有提供文件名，则对文件名进行名称混淆处理，否则设置为 None
        mangled_filename = self._mangler.mangle(filename) if filename else None
        
        # 创建一个 ModuleSpec 对象，用于描述模块的规范信息
        spec = importlib.machinery.ModuleSpec(
            name,
            self,  # type: ignore[arg-type]
            origin="<package_importer>",
            is_package=is_package,
        )
        
        # 根据 ModuleSpec 对象创建一个新的模块对象
        module = importlib.util.module_from_spec(spec)
        
        # 将新创建的模块对象存储在 self.modules 字典中
        self.modules[name] = module
        
        # 设置模块对象的名称为经过名称混淆处理后的名称
        module.__name__ = self._mangler.mangle(name)
        
        # 获取模块对象的命名空间字典
        ns = module.__dict__
        
        # 设置模块的 __spec__ 属性为当前的 ModuleSpec 对象
        ns["__spec__"] = spec
        
        # 设置模块的 __loader__ 属性为当前对象（即 self）
        ns["__loader__"] = self
        
        # 设置模块的 __file__ 属性为经过名称混淆处理后的文件名
        ns["__file__"] = mangled_filename
        
        # 设置模块的 __cached__ 属性为 None
        ns["__cached__"] = None
        
        # 设置模块的 __builtins__ 属性为经过补丁处理后的内建模块
        ns["__builtins__"] = self.patched_builtins
        
        # 设置模块的 __torch_package__ 属性为 True
        ns["__torch_package__"] = True
        
        # 将当前模块添加到私有全局注册表 _package_imported_modules 中
        assert module.__name__ not in _package_imported_modules
        _package_imported_modules[module.__name__] = module
        
        # 在父模块上预先安装当前模块，以防止 IMPORT_FROM 尝试访问 sys.modules
        self._install_on_parent(parent, name, module)
        
        # 如果提供了文件名，则继续处理以下步骤
        if filename is not None:
            assert mangled_filename is not None
            
            # 预先将源代码安装到 `linecache` 中，以便堆栈跟踪和 `inspect` 等工作正常
            assert filename not in linecache.cache  # type: ignore[attr-defined]
            linecache.lazycache(mangled_filename, ns)
            
            # 编译源代码，生成可执行的代码对象
            code = self._compile_source(filename, mangled_filename)
            
            # 在模块的命名空间中执行编译后的代码
            exec(code, ns)
        
        # 返回创建或加载的模块对象
        return module

    # 加载模块的方法，根据模块名和父模块名加载模块
    def _load_module(self, name: str, parent: str):
        # 从根节点开始逐级查找模块
        cur: _PathNode = self.root
        for atom in name.split("."):
            # 如果当前节点不是包节点或者当前节点下找不到对应模块，则抛出模块未找到的异常
            if not isinstance(cur, _PackageNode) or atom not in cur.children:
                # 如果模块名在 IMPLICIT_IMPORT_ALLOWLIST 中，则允许隐式导入
                if name in IMPLICIT_IMPORT_ALLOWLIST:
                    # 将模块对象存储在 self.modules 中，并返回该模块对象
                    module = self.modules[name] = importlib.import_module(name)
                    return module
                # 抛出模块未找到的异常，显示详细信息
                raise ModuleNotFoundError(
                    f'No module named "{name}" in self-contained archive "{self.filename}"'
                    f" and the module is also not in the list of allowed external modules: {self.extern_modules}",
                    name=name,
                )
            # 如果当前节点是外部节点，则直接导入模块
            cur = cur.children[atom]
            if isinstance(cur, _ExternNode):
                # 将模块对象存储在 self.modules 中，并返回该模块对象
                module = self.modules[name] = importlib.import_module(name)
                return module
        # 根据节点创建模块，并返回该模块对象
        return self._make_module(name, cur.source_file, isinstance(cur, _PackageNode), parent)  # type: ignore[attr-defined]

    # 编译源代码的方法，接受源文件路径和混淆后的文件名作为参数
    def _compile_source(self, fullpath: str, mangled_filename: str):
        # 从 zip_reader 中获取源代码记录
        source = self.zip_reader.get_record(fullpath)
        
        # 规范化源代码中的换行符
        source = _normalize_line_endings(source)
        
        # 编译源代码，生成可执行的代码对象
        return compile(source, mangled_filename, "exec", dont_inherit=True)

    # 名称为 `get_source`，以便 linecache 可以找到源代码，
    # 当这个方法作为模块的 __loader__ 时。
    def get_source(self, module_name) -> str:
        # linecache调用`get_source`时使用`module.__name__`作为参数，因此这里需要解开名称修饰。
        module = self.import_module(demangle(module_name))
        # 使用zip_reader从模块的记录中获取数据，并以utf-8解码返回源代码字符串。
        return self.zip_reader.get_record(demangle(module.__file__)).decode("utf-8")

    # 注意：命名为`get_resource_reader`以便让importlib.resources能够找到它。
    # 否则会被视为内部方法。
    def get_resource_reader(self, fullname):
        try:
            # 尝试获取fullname指定的包。
            package = self._get_package(fullname)
        except ImportError:
            # 如果导入失败，则返回None。
            return None
        # 如果找到的包的加载器不是自身，则返回None。
        if package.__loader__ is not self:
            return None
        # 返回一个_PackageResourceReader对象，用于处理指定fullname的资源。
        return _PackageResourceReader(self, fullname)

    def _install_on_parent(self, parent: str, name: str, module: types.ModuleType):
        # 如果parent为空，则直接返回，不进行安装。
        if not parent:
            return
        # 将module作为其父模块的属性设置。
        parent_module = self.modules[parent]
        # 如果父模块的加载器是自身，则将name的最后一部分作为属性名，将module设置为其属性。
        if parent_module.__loader__ is self:
            setattr(parent_module, name.rpartition(".")[2], module)

    # 注意：从cpython的导入代码复制而来，调用创建模块的方法替换为_make_module。
    # 定义一个方法 `_do_find_and_load`，用于查找和加载指定名称的模块
    def _do_find_and_load(self, name):
        # 初始化路径变量为 None
        path = None
        # 获取模块名称中最后一个点号之前的部分作为父模块名
        parent = name.rpartition(".")[0]
        # 获取模块名称中最后一个点号之后的部分作为无父级的模块名
        module_name_no_parent = name.rpartition(".")[-1]
        
        # 如果存在父模块名
        if parent:
            # 如果父模块名不在已加载的模块列表中，调用 `_gcd_import` 方法加载它
            if parent not in self.modules:
                self._gcd_import(parent)
            
            # 如果模块名已经在已加载的模块列表中，直接返回该模块
            if name in self.modules:
                return self.modules[name]
            
            # 获取父模块对象
            parent_module = self.modules[parent]

            try:
                # 尝试获取父模块的路径
                path = parent_module.__path__  # type: ignore[attr-defined]

            except AttributeError:
                # 处理当尝试导入仅包含 pybinded 文件的包时的异常情况
                if isinstance(
                    parent_module.__loader__,
                    importlib.machinery.ExtensionFileLoader,
                ):
                    # 如果模块名不在外部模块列表中，抛出模块未找到异常
                    if name not in self.extern_modules:
                        msg = (
                            _ERR_MSG
                            + "; {!r} is a c extension module which was not externed. C extension modules \
                            need to be externed by the PackageExporter in order to be used as we do not support interning them.}."
                        ).format(name, name)
                        raise ModuleNotFoundError(msg, name=name) from None
                    # 如果模块名对应的对象不是模块类型，抛出模块未找到异常
                    if not isinstance(
                        parent_module.__dict__.get(module_name_no_parent),
                        types.ModuleType,
                    ):
                        msg = (
                            _ERR_MSG
                            + "; {!r} is a c extension package which does not contain {!r}."
                        ).format(name, parent, name)
                        raise ModuleNotFoundError(msg, name=name) from None
                else:
                    # 抛出模块未找到异常，指示模块名不是一个包
                    msg = (_ERR_MSG + "; {!r} is not a package").format(name, parent)
                    raise ModuleNotFoundError(msg, name=name) from None

        # 调用 `_load_module` 方法加载模块
        module = self._load_module(name, parent)

        # 将加载的模块安装在父模块上
        self._install_on_parent(parent, name, module)

        # 返回加载的模块
        return module

    # 注意：这段代码摘自 CPython 的导入代码
    # 查找并加载指定名称的模块
    def _find_and_load(self, name):
        # 获取模块对象或标记为需要加载的状态
        module = self.modules.get(name, _NEEDS_LOADING)
        # 如果模块需要加载，则调用私有方法进行查找和加载
        if module is _NEEDS_LOADING:
            return self._do_find_and_load(name)

        # 如果模块为 None，说明导入被中断，抛出 ModuleNotFoundError 异常
        if module is None:
            message = f"import of {name} halted; None in sys.modules"
            raise ModuleNotFoundError(message, name=name)

        # 处理特定情况下的模块导入问题，例如处理 sys.modules 被修改的情况
        if name == "os":
            # 如果是 os 模块，则将 os.path 注册为模块的路径
            self.modules["os.path"] = cast(Any, module).path
        elif name == "typing":
            # 如果是 typing 模块，则将 typing.io 和 typing.re 分别注册为模块的 io 和 re 子模块
            self.modules["typing.io"] = cast(Any, module).io
            self.modules["typing.re"] = cast(Any, module).re

        # 返回找到的模块对象
        return module

    # 导入并返回模块的函数，基于名称、调用包和级别调整
    def _gcd_import(self, name, package=None, level=0):
        """Import and return the module based on its name, the package the call is
        being made from, and the level adjustment.

        This function represents the greatest common denominator of functionality
        between import_module and __import__. This includes setting __package__ if
        the loader did not.

        """
        # 执行参数的健全性检查
        _sanity_check(name, package, level)
        # 如果级别大于 0，则解析名称以获取最终模块名称
        if level > 0:
            name = _resolve_name(name, package, level)

        # 调用 _find_and_load 方法查找并加载模块
        return self._find_and_load(name)

    # 注意: 从 CPython 的导入代码复制而来
    def _handle_fromlist(self, module, fromlist, *, recursive=False):
        """处理 fromlist 参数，确定 __import__ 应该返回什么。

        import_ 参数是一个可调用对象，接受要导入的模块名。这是为了使函数不依赖于假设要使用 importlib 的导入实现。

        """
        # 解析模块名称，去除可能存在的名称修饰
        module_name = demangle(module.__name__)
        # 处理 fromlist 中的每个元素
        # 如果模块是一个包，尝试从 fromlist 导入内容
        if hasattr(module, "__path__"):
            for x in fromlist:
                # 检查每个元素是否为字符串
                if not isinstance(x, str):
                    if recursive:
                        where = module_name + ".__all__"
                    else:
                        where = "``from list''"
                    # 抛出类型错误，要求 fromlist 中的每一项必须是字符串而不是其他类型
                    raise TypeError(
                        f"Item in {where} must be str, not {type(x).__name__}"
                    )
                elif x == "*":
                    # 如果遇到 "*"，且非递归情况下且模块有 __all__ 属性，则递归处理 fromlist
                    if not recursive and hasattr(module, "__all__"):
                        self._handle_fromlist(module, module.__all__, recursive=True)
                elif not hasattr(module, x):
                    # 如果模块不包含名为 x 的属性，尝试导入 x
                    from_name = f"{module_name}.{x}"
                    try:
                        self._gcd_import(from_name)
                    except ModuleNotFoundError as exc:
                        # 如果模块不存在，根据向后兼容性忽略导入失败的情况
                        if (
                            exc.name == from_name
                            and self.modules.get(from_name, _NEEDS_LOADING) is not None
                        ):
                            continue
                        raise
        return module

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        # 如果 level 为 0，直接使用 _gcd_import 导入模块
        if level == 0:
            module = self._gcd_import(name)
        else:
            # 否则，计算 __package__ 并使用 _gcd_import 导入模块
            globals_ = globals if globals is not None else {}
            package = _calc___package__(globals_)
            module = self._gcd_import(name, package, level)
        if not fromlist:
            # 如果 fromlist 为空，返回 name 中第一个点之前的部分的模块
            if level == 0:
                return self._gcd_import(name.partition(".")[0])
            elif not name:
                return module
            else:
                # 计算模块名称中第一个点之前的部分，并返回对应的模块
                cut_off = len(name) - len(name.partition(".")[0])
                module_name = demangle(module.__name__)
                return self.modules[module_name[: len(module_name) - cut_off]]
        else:
            # 否则，处理 fromlist 中的导入情况
            return self._handle_fromlist(module, fromlist)
    def _get_package(self, package):
        """获取给定包名或模块对象，并返回该模块。

        如果是包名，则导入对应的模块。如果传入或导入的模块对象不是一个包，会抛出异常。
        """
        if hasattr(package, "__spec__"):
            # 检查模块对象是否具有 __spec__ 属性
            if package.__spec__.submodule_search_locations is None:
                # 如果没有子模块搜索位置，则不是一个包，抛出异常
                raise TypeError(f"{package.__spec__.name!r} is not a package")
            else:
                return package
        else:
            # 否则，导入指定的模块，并检查其是否为包
            module = self.import_module(package)
            if module.__spec__.submodule_search_locations is None:
                raise TypeError(f"{package!r} is not a package")
            else:
                return module

    def _zipfile_path(self, package, resource=None):
        """生成指定包中资源的路径。

        Args:
            package: 包名或模块对象
            resource: 资源名称（可选）

        Returns:
            生成的资源路径
        """
        package = self._get_package(package)
        assert package.__loader__ is self
        name = demangle(package.__name__)  # 将包名转换为路径形式
        if resource is not None:
            resource = _normalize_path(resource)  # 标准化资源路径
            return f"{name.replace('.', '/')}/{resource}"  # 返回资源路径
        else:
            return f"{name.replace('.', '/')}"

    def _get_or_create_package(
        self, atoms: List[str]
    ) -> "Union[_PackageNode, _ExternNode]":
        """获取或创建包节点。

        根据传入的路径信息列表，从根节点开始遍历，逐级获取或创建对应的包节点。
        """
        cur = self.root
        for i, atom in enumerate(atoms):
            node = cur.children.get(atom, None)
            if node is None:
                node = cur.children[atom] = _PackageNode(None)
            if isinstance(node, _ExternNode):
                return node
            if isinstance(node, _ModuleNode):
                name = ".".join(atoms[:i])
                raise ImportError(
                    f"inconsistent module structure. module {name} is not a package, but has submodules"
                )
            assert isinstance(node, _PackageNode)
            cur = node
        return cur

    def _add_file(self, filename: str):
        """从给定文件组装一个 Python 模块。

        如果文件在 .data 目录内，则忽略。
        如果文件名以 '__init__.py' 结尾，将其设置为包的源文件。
        如果文件名以 '.py' 结尾，将其添加为模块节点的子节点。
        """
        *prefix, last = filename.split("/")
        if len(prefix) > 1 and prefix[0] == ".data":
            return  # 忽略 .data 目录内的文件
        package = self._get_or_create_package(prefix)
        if isinstance(package, _ExternNode):
            raise ImportError(
                f"inconsistent module structure. package contains a module file {filename}"
                f" that is a subpackage of a module marked external."
            )
        if last == "__init__.py":
            package.source_file = filename  # 设置包的源文件
        elif last.endswith(".py"):
            package_name = last[: -len(".py")]
            package.children[package_name] = _ModuleNode(filename)  # 添加为模块节点的子节点
    # 定义一个方法 `_add_extern`，接受一个外部名称字符串作为参数
    def _add_extern(self, extern_name: str):
        # 使用点号分割外部名称字符串，将最后一个元素作为 `last`，其余部分作为 `prefix`
        *prefix, last = extern_name.split(".")
        # 调用内部方法 `_get_or_create_package`，传入 `prefix` 部分，获取或创建对应的包对象
        package = self._get_or_create_package(prefix)
        # 如果 `package` 是 `_ExternNode` 类型的对象，说明外部名称已经存在，直接返回
        if isinstance(package, _ExternNode):
            return  # the shorter extern covers this extern case
        # 否则，在 `package` 的 `children` 字典中创建一个新的 `_ExternNode` 对象，键为 `last`
        package.children[last] = _ExternNode()
# `_NEEDS_LOADING`被定义为一个特殊对象，用于表示需要加载的状态
_NEEDS_LOADING = object()

# `_ERR_MSG_PREFIX`定义了一个错误消息的前缀
_ERR_MSG_PREFIX = "No module named "

# `_ERR_MSG`是完整的错误消息，包括前缀和占位符{!r}
_ERR_MSG = _ERR_MSG_PREFIX + "{!r}"


class _PathNode:
    # `_PathNode`类是路径节点的基类，没有额外的属性或方法定义
    pass


class _PackageNode(_PathNode):
    def __init__(self, source_file: Optional[str]):
        self.source_file = source_file
        self.children: Dict[str, _PathNode] = {}
        # `_PackageNode`类表示一个包节点，包含源文件路径和子节点字典


class _ModuleNode(_PathNode):
    __slots__ = ["source_file"]

    def __init__(self, source_file: str):
        self.source_file = source_file
        # `_ModuleNode`类表示一个模块节点，包含源文件路径


class _ExternNode(_PathNode):
    # `_ExternNode`类表示一个外部节点，是`_PathNode`的子类但没有额外的属性或方法定义
    pass


# `_package_imported_modules`是一个弱引用字典，用于存储已经被包导入的所有模块
_package_imported_modules: WeakValueDictionary = WeakValueDictionary()

# `inspect`默认只在`sys.modules`中查找类的源文件，此处进行了修改以检查私有的包导入模块注册表
_orig_getfile = inspect.getfile


def _patched_getfile(object):
    if inspect.isclass(object):
        if object.__module__ in _package_imported_modules:
            return _package_imported_modules[object.__module__].__file__
    return _orig_getfile(object)
# `_patched_getfile`函数替换了`inspect.getfile`，使其在查找类的源文件时也检查私有的包导入模块注册表

inspect.getfile = _patched_getfile


class _PackageResourceReader:
    """Private class used to support PackageImporter.get_resource_reader().

    Confirms to the importlib.abc.ResourceReader interface. Allowed to access
    the innards of PackageImporter.
    """

    def __init__(self, importer, fullname):
        self.importer = importer
        self.fullname = fullname
        # `_PackageResourceReader`类用于支持`PackageImporter.get_resource_reader()`，存储导入器和完整名称

    def open_resource(self, resource):
        from io import BytesIO

        return BytesIO(self.importer.load_binary(self.fullname, resource))
    # `open_resource`方法打开资源，返回一个BytesIO对象，使用导入器加载二进制数据

    def resource_path(self, resource):
        # `resource_path`方法根据资源名称返回具体的文件系统路径或者抛出FileNotFoundError
        if isinstance(
            self.importer.zip_reader, DirectoryReader
        ) and self.importer.zip_reader.has_record(
            os.path.join(self.fullname, resource)
        ):
            return os.path.join(
                self.importer.zip_reader.directory, self.fullname, resource
            )
        raise FileNotFoundError
    # `resource_path`方法根据资源名称返回具体的文件系统路径或者抛出FileNotFoundError

    def is_resource(self, name):
        path = self.importer._zipfile_path(self.fullname, name)
        return self.importer.zip_reader.has_record(path)
    # `is_resource`方法检查给定名称的资源是否存在于导入器的ZIP文件中
    # 导入路径处理工具 Path 从 pathlib 模块
    from pathlib import Path

    # 将 fullname 中的点替换为斜杠，构造文件名路径
    filename = self.fullname.replace(".", "/")

    # 获取 fullname 在 ZIP 文件中的完整路径，并转换为 Path 对象
    fullname_path = Path(self.importer._zipfile_path(self.fullname))

    # 获取 ZIP 文件读取器中的所有记录
    files = self.importer.zip_reader.get_all_records()

    # 用于记录已经处理过的子目录名集合
    subdirs_seen = set()

    # 遍历每个文件名
    for filename in files:
        try:
            # 尝试将 filename 相对于 fullname_path 解析成相对路径
            relative = Path(filename).relative_to(fullname_path)
        except ValueError:
            # 如果解析失败，继续下一个文件名的处理
            continue
        
        # 如果相对路径的父目录名为空，则说明是在顶层目录，直接 yield 文件名
        parent_name = relative.parent.name
        if len(parent_name) == 0:
            yield relative.name
        # 如果父目录名不在 subdirs_seen 中，说明是新的子目录，记录并 yield 父目录名
        elif parent_name not in subdirs_seen:
            subdirs_seen.add(parent_name)
            yield parent_name
```