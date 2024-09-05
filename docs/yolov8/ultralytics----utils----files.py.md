# `.\yolov8\ultralytics\utils\files.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import contextlib                   # 导入上下文管理模块
import glob                         # 导入文件路径模块
import os                           # 导入操作系统接口模块
import shutil                       # 导入文件操作模块
import tempfile                     # 导入临时文件和目录模块
from contextlib import contextmanager  # 导入上下文管理器装饰器
from datetime import datetime       # 导入日期时间模块
from pathlib import Path            # 导入路径操作模块


class WorkingDirectory(contextlib.ContextDecorator):
    """Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager."""

    def __init__(self, new_dir):
        """Sets the working directory to 'new_dir' upon instantiation."""
        self.dir = new_dir          # 设置新的工作目录
        self.cwd = Path.cwd().resolve()  # 获取当前工作目录的绝对路径

    def __enter__(self):
        """Changes the current directory to the specified directory."""
        os.chdir(self.dir)          # 切换当前工作目录到指定目录

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        """Restore the current working directory on context exit."""
        os.chdir(self.cwd)          # 在上下文退出时恢复原始工作目录


@contextmanager
def spaces_in_path(path):
    """
    Context manager to handle paths with spaces in their names. If a path contains spaces, it replaces them with
    underscores, copies the file/directory to the new path, executes the context code block, then copies the
    file/directory back to its original location.

    Args:
        path (str | Path): The original path.

    Yields:
        (Path): Temporary path with spaces replaced by underscores if spaces were present, otherwise the original path.

    Example:
        ```py
        with ultralytics.utils.files import spaces_in_path

        with spaces_in_path('/path/with spaces') as new_path:
            # Your code here
        ```py
    """

    # If path has spaces, replace them with underscores
    if " " in str(path):
        string = isinstance(path, str)  # 判断输入路径类型
        path = Path(path)

        # Create a temporary directory and construct the new path
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / path.name.replace(" ", "_")  # 构造替换空格后的临时路径

            # Copy file/directory
            if path.is_dir():
                # 如果是目录，则复制整个目录结构
                # tmp_path.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, tmp_path)
            elif path.is_file():
                # 如果是文件，则复制文件
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, tmp_path)

            try:
                # Yield the temporary path
                yield str(tmp_path) if string else tmp_path  # 生成临时路径并传递给上下文

            finally:
                # Copy file/directory back
                # 将文件/目录复制回原始位置
                if tmp_path.is_dir():
                    shutil.copytree(tmp_path, path, dirs_exist_ok=True)
                elif tmp_path.is_file():
                    shutil.copy2(tmp_path, path)  # 复制文件回原始位置

    else:
        # If there are no spaces, just yield the original path
        yield path  # 如果路径中没有空格，则直接传递原始路径


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    """
    # 根据参数path创建一个Path对象，确保在不同操作系统上路径兼容性
    path = Path(path)  
    
    # 检查路径是否存在且exist_ok参数为False时，执行路径增量操作
    if path.exists() and not exist_ok:
        # 如果path是文件，则保留文件扩展名(suffix)，否则suffix为空字符串
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
    
        # 方法1：从2开始尝试递增直到9999，形成新的路径p
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # 增加路径序号
            # 如果新路径p不存在，则中断循环
            if not os.path.exists(p):
                break
        # 更新path为新路径的Path对象
        path = Path(p)
    
    # 如果设置了mkdir为True，则创建路径作为目录（包括创建中间目录）
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # 创建目录
    
    # 返回增加处理后的Path对象
    return path
def file_age(path=__file__):
    """Return days since last file update."""
    # 获取当前时间与文件最后修改时间的时间差
    dt = datetime.now() - datetime.fromtimestamp(Path(path).stat().st_mtime)  # delta
    # 返回时间差的天数部分，表示文件自上次更新以来经过的天数
    return dt.days  # + dt.seconds / 86400  # fractional days


def file_date(path=__file__):
    """Return human-readable file modification date, i.e. '2021-3-26'."""
    # 获取文件最后修改时间
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    # 返回文件最后修改时间的年、月、日组成的格式化字符串
    return f"{t.year}-{t.month}-{t.day}"


def file_size(path):
    """Return file/dir size (MB)."""
    if isinstance(path, (str, Path)):
        mb = 1 << 20  # bytes to MiB (1024 ** 2)
        path = Path(path)
        if path.is_file():
            # 如果路径是文件，则返回文件大小（MB）
            return path.stat().st_size / mb
        elif path.is_dir():
            # 如果路径是目录，则返回目录中所有文件大小的总和（MB）
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    # 默认情况下返回 0.0 表示大小为 0 MB
    return 0.0


def get_latest_run(search_dir="."):
    """Return path to most recent 'last.pt' in /runs (i.e. to --resume from)."""
    # 在指定目录及其子目录中搜索所有符合条件的文件路径列表
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    # 返回最新的文件路径，即创建时间最晚的文件路径，如果列表为空则返回空字符串
    return max(last_list, key=os.path.getctime) if last_list else ""


def update_models(model_names=("yolov8n.pt",), source_dir=Path("."), update_names=False):
    """
    Updates and re-saves specified YOLO models in an 'updated_models' subdirectory.

    Args:
        model_names (tuple, optional): Model filenames to update, defaults to ("yolov8n.pt").
        source_dir (Path, optional): Directory containing models and target subdirectory, defaults to current directory.
        update_names (bool, optional): Update model names from a data YAML.

    Example:
        ```py
        from ultralytics.utils.files import update_models

        model_names = (f"rtdetr-{size}.pt" for size in "lx")
        update_models(model_names)
        ```py
    """
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names

    # 设置目标目录为当前目录下的 updated_models 子目录，如果不存在则创建
    target_dir = source_dir / "updated_models"
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists

    for model_name in model_names:
        model_path = source_dir / model_name
        print(f"Loading model from {model_path}")

        # 加载模型
        model = YOLO(model_path)
        model.half()  # 使用半精度浮点数进行模型运算，加速模型计算速度

        if update_names:  # 根据数据 YAML 更新模型的类别名称
            model.model.names = default_class_names("coco8.yaml")

        # 定义新的保存路径
        save_path = target_dir / model_name

        # 使用 model.save() 方法重新保存模型
        print(f"Re-saving {model_name} model to {save_path}")
        model.save(save_path, use_dill=False)
```