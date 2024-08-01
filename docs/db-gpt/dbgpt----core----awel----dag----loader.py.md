# `.\DB-GPT-src\dbgpt\core\awel\dag\loader.py`

```py
"""DAG loader.

DAGLoader will load DAGs from dag_dirs or other sources.
Now only support load DAGs from local files.
"""

import hashlib  # 导入哈希算法模块
import logging  # 导入日志记录模块
import os  # 导入操作系统功能模块
import sys  # 导入系统相关功能模块
import traceback  # 导入异常追踪模块
from abc import ABC, abstractmethod  # 从abc模块导入抽象基类和抽象方法
from typing import List  # 导入类型提示模块中的List类型

from .base import DAG  # 从当前包中的base模块导入DAG类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class DAGLoader(ABC):
    """Abstract base class representing a loader for loading DAGs."""

    @abstractmethod
    def load_dags(self) -> List[DAG]:
        """Load dags."""


class LocalFileDAGLoader(DAGLoader):
    """DAG loader for loading DAGs from local files."""

    def __init__(self, dag_dirs: List[str]) -> None:
        """Initialize a LocalFileDAGLoader.

        Args:
            dag_dirs (List[str]): The directories to load DAGs.
        """
        self._dag_dirs = dag_dirs  # 初始化实例属性_dag_dirs，用于存储DAG文件目录列表

    def load_dags(self) -> List[DAG]:
        """Load dags from local files."""
        dags = []  # 初始化一个空列表，用于存储加载的DAG对象
        for filepath in self._dag_dirs:  # 遍历DAG文件目录列表
            if not os.path.exists(filepath):  # 如果文件路径不存在
                continue  # 继续下一次循环
            if os.path.isdir(filepath):  # 如果文件路径是一个目录
                dags += _process_directory(filepath)  # 加载该目录下的所有DAG文件
            else:
                dags += _process_file(filepath)  # 加载指定的单个DAG文件
        return dags  # 返回加载的所有DAG对象列表


def _process_directory(directory: str) -> List[DAG]:
    """Process a directory to find and process Python files containing DAGs.

    Args:
        directory (str): Directory path to process.

    Returns:
        List[DAG]: List of DAG objects found and processed.
    """
    dags = []  # 初始化一个空列表，用于存储找到并处理的DAG对象
    for file in os.listdir(directory):  # 遍历目录下的所有文件和文件夹
        if file.endswith(".py"):  # 如果文件是以.py结尾的Python文件
            filepath = os.path.join(directory, file)  # 构建文件的完整路径
            dags += _process_file(filepath)  # 处理该Python文件，获取其中的DAG对象
    return dags  # 返回处理后的所有DAG对象列表


def _process_file(filepath) -> List[DAG]:
    """Process a Python file to load modules and extract DAG objects.

    Args:
        filepath: Path to the Python file.

    Returns:
        List[DAG]: List of DAG objects extracted from the file.
    """
    mods = _load_modules_from_file(filepath)  # 加载Python文件中的模块
    results = _process_modules(mods)  # 处理加载的模块，提取其中的DAG对象
    return results  # 返回提取的DAG对象列表


def _load_modules_from_file(
    filepath: str, mod_name: str | None = None, show_log: bool = True
):
    """Load Python modules from a file.

    Args:
        filepath (str): Path to the Python file.
        mod_name (str | None, optional): Name of the module to load. Defaults to None.
        show_log (bool, optional): Whether to log the import process. Defaults to True.

    Returns:
        modules: List of loaded module objects.
    """
    import importlib  # 导入动态加载模块的标准库
    import importlib.machinery  # 导入加载机制模块
    import importlib.util  # 导入模块规范工具模块

    if show_log:  # 如果需要记录导入过程
        logger.info(f"Importing {filepath}")  # 记录导入文件的信息

    org_mod_name, _ = os.path.splitext(os.path.split(filepath)[-1])  # 获取原始模块名
    path_hash = hashlib.sha1(filepath.encode("utf-8")).hexdigest()  # 计算文件路径的SHA-1哈希值
    if mod_name is None:  # 如果未指定模块名
        mod_name = f"unusual_prefix_{path_hash}_{org_mod_name}"  # 使用特定前缀和文件路径哈希构建模块名

        if mod_name in sys.modules:  # 如果该模块名已经在sys.modules中存在
            del sys.modules[mod_name]  # 从sys.modules中删除已有的同名模块

    def parse(mod_name, filepath):
        """Parse and load a module from a file.

        Args:
            mod_name (str): Name of the module.
            filepath (str): Path to the Python file.

        Returns:
            List: List containing the loaded module object.
        """
        try:
            loader = importlib.machinery.SourceFileLoader(mod_name, filepath)  # 使用源文件加载器加载模块
            spec = importlib.util.spec_from_loader(mod_name, loader)  # 根据加载器创建模块规范
            new_module = importlib.util.module_from_spec(spec)  # 创建新的模块对象
            sys.modules[spec.name] = new_module  # 将新模块对象添加到sys.modules中
            loader.exec_module(new_module)  # 执行模块代码
            return [new_module]  # 返回包含新模块对象的列表
        except Exception:
            msg = traceback.format_exc()  # 获取异常的详细信息
            logger.error(f"Failed to import: {filepath}, error message: {msg}")  # 记录导入失败的错误信息
            # TODO save error message  # TODO注释，保存错误消息的功能尚未实现
            return []  # 返回空列表，表示未成功加载模块

    return parse(mod_name, filepath)  # 返回解析和加载模块后的结果


def _process_modules(mods, show_log: bool = True) -> List[DAG]:
    """Process loaded modules to find and extract DAG objects.

    Args:
        mods: List of loaded module objects.
        show_log (bool, optional): Whether to log the processing. Defaults to True.

    Returns:
        List[DAG]: List of DAG objects found in the modules.
    """
    top_level_dags = (
        (o, m) for m in mods for o in m.__dict__.values() if isinstance(o, DAG)
    )  # 生成器表达式，遍历模块中的所有对象，筛选出属于DAG类的对象
    found_dags = []  # 初始化一个空列表，用于存储找到的DAG对象
    for dag_obj, module in top_level_dags:  # 遍历筛选出的每个DAG对象及其所属的模块
        found_dags.append(dag_obj)  # 将找到的DAG对象添加到列表中
    return found_dags  # 返回找到的所有DAG对象列表
    # 对每个顶级 DAG 和模块进行迭代
    for dag, mod in top_level_dags:
        try:
            # TODO validate dag params  # TODO: 验证 DAG 参数（待完成）
            
            # 如果 show_log 为 True，则记录信息到日志中，显示找到的 DAG 名称和其来源模块以及模块文件路径
            if show_log:
                logger.info(
                    f"Found dag {dag} from mod {mod} and model file {mod.__file__}"
                )
                
            # 将找到的 DAG 添加到 found_dags 列表中
            found_dags.append(dag)
        
        # 捕获任何异常并记录详细的错误信息到日志中
        except Exception:
            msg = traceback.format_exc()
            logger.error(f"Failed to dag file, error message: {msg}")
    
    # 返回包含所有找到 DAG 名称的列表
    return found_dags
```