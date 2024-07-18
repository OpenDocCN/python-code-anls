# `.\graphrag\graphrag\index\progress\rich.py`

```
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Rich-based progress reporter for CLI use."""

# 引入 asyncio 库，用于异步操作
import asyncio

# 引入 Progress 类型别名 ProgressReporter，定义于 .types 模块中
from datashaper import Progress as DSProgress

# 引入 Console、Group、Live、Progress、TaskID、TimeElapsedColumn 类
# 以及 Spinner、Tree 组件，用于丰富的命令行界面显示
from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, TaskID, TimeElapsedColumn
from rich.spinner import Spinner
from rich.tree import Tree

# 从当前包的 types 模块中引入 ProgressReporter 类型别名
from .types import ProgressReporter


# https://stackoverflow.com/a/34325723
# 定义 RichProgressReporter 类，实现 ProgressReporter 接口
class RichProgressReporter(ProgressReporter):
    """A rich-based progress reporter for CLI use."""

    _console: Console  # 命令行控制台对象
    _group: Group  # Rich 组件 Group 对象
    _tree: Tree  # Rich 组件 Tree 对象
    _live: Live  # Rich 组件 Live 对象
    _task: TaskID | None = None  # 任务 ID，可空
    _prefix: str  # 进度报告的前缀字符串
    _transient: bool  # 是否是瞬态的
    _disposing: bool = False  # 是否正在释放资源
    _progressbar: Progress  # Rich 进度条对象
    _last_refresh: float = 0  # 上次刷新的时间戳

    def dispose(self) -> None:
        """Dispose of the progress reporter."""
        self._disposing = True  # 标记正在释放资源
        self._live.stop()  # 停止 Live 组件更新

    @property
    def console(self) -> Console:
        """Get the console."""
        return self._console  # 返回命令行控制台对象

    @property
    def group(self) -> Group:
        """Get the group."""
        return self._group  # 返回 Rich 组件 Group 对象

    @property
    def tree(self) -> Tree:
        """Get the tree."""
        return self._tree  # 返回 Rich 组件 Tree 对象

    @property
    def live(self) -> Live:
        """Get the live."""
        return self._live  # 返回 Rich 组件 Live 对象

    def __init__(
        self,
        prefix: str,
        parent: "RichProgressReporter | None" = None,
        transient: bool = True,
    ) -> None:
        """Create a new rich-based progress reporter."""
        self._prefix = prefix  # 设置进度报告的前缀字符串

        if parent is None:
            # 如果没有父进度报告者，创建新的 Rich 控制台、Group、Tree 和 Live 组件
            console = Console()
            group = Group(Spinner("dots", prefix), fit=True)
            tree = Tree(group)
            live = Live(
                tree, console=console, refresh_per_second=1, vertical_overflow="crop"
            )
            live.start()  # 启动 Live 组件

            self._console = console  # 初始化命令行控制台对象
            self._group = group  # 初始化 Rich 组件 Group 对象
            self._tree = tree  # 初始化 Rich 组件 Tree 对象
            self._live = live  # 初始化 Rich 组件 Live 对象
            self._transient = False  # 标记为非瞬态
        else:
            # 如果有父进度报告者，复用其控制台和 Group，创建新的 Progress 组件
            self._console = parent.console  # 继承父进度报告者的命令行控制台对象
            self._group = parent.group  # 继承父进度报告者的 Rich 组件 Group 对象
            progress_columns = [*Progress.get_default_columns(), TimeElapsedColumn()]
            self._progressbar = Progress(
                *progress_columns, console=self._console, transient=transient
            )  # 创建新的 Rich 进度条对象

            tree = Tree(prefix)  # 创建新的 Rich 组件 Tree 对象，并设置前缀
            tree.add(self._progressbar)  # 将 Rich 进度条对象添加到 Tree 中
            tree.hide_root = True

            if parent is not None:
                parent_tree = parent.tree
                parent_tree.hide_root = False
                parent_tree.add(tree)

            self._tree = tree  # 初始化 Rich 组件 Tree 对象
            self._live = parent.live  # 继承父进度报告者的 Rich 组件 Live 对象
            self._transient = transient  # 设置是否是瞬态

        self.refresh()  # 刷新界面显示
    def refresh(self) -> None:
        """Perform a debounced refresh."""
        # 获取当前事件循环时间
        now = asyncio.get_event_loop().time()
        # 计算距上次刷新的时间间隔
        duration = now - self._last_refresh
        # 如果时间间隔大于0.1秒，则进行刷新操作
        if duration > 0.1:
            # 更新上次刷新时间为当前时间
            self._last_refresh = now
            # 强制执行刷新操作
            self.force_refresh()

    def force_refresh(self) -> None:
        """Force a refresh."""
        # 调用live对象的refresh方法执行刷新
        self.live.refresh()

    def stop(self) -> None:
        """Stop the progress reporter."""
        # 停止进度报告器的运行
        self._live.stop()

    def child(self, prefix: str, transient: bool = True) -> ProgressReporter:
        """Create a child progress bar."""
        # 创建一个子进度条对象，并返回
        return RichProgressReporter(parent=self, prefix=prefix, transient=transient)

    def error(self, message: str) -> None:
        """Report an error."""
        # 打印红色错误信息
        self._console.print(f"❌ [red]{message}[/red]")

    def warning(self, message: str) -> None:
        """Report a warning."""
        # 打印黄色警告信息
        self._console.print(f"⚠️ [yellow]{message}[/yellow]")

    def success(self, message: str) -> None:
        """Report success."""
        # 打印绿色成功信息
        self._console.print(f"🚀 [green]{message}[/green]")

    def info(self, message: str) -> None:
        """Report information."""
        # 打印普通信息
        self._console.print(message)

    def __call__(self, progress_update: DSProgress) -> None:
        """Update progress."""
        # 如果正在销毁，则直接返回
        if self._disposing:
            return
        # 获取进度条对象
        progressbar = self._progressbar

        # 如果任务为空，则添加一个新任务
        if self._task is None:
            self._task = progressbar.add_task(self._prefix)

        # 设置进度更新的描述信息
        progress_description = ""
        if progress_update.description is not None:
            progress_description = f" - {progress_update.description}"

        # 计算已完成和总数
        completed = progress_update.completed_items or progress_update.percent
        total = progress_update.total_items or 1

        # 更新进度条
        progressbar.update(
            self._task,
            completed=completed,
            total=total,
            description=f"{self._prefix}{progress_description}",
        )

        # 如果完成等于总数且是瞬态的，则隐藏进度条
        if completed == total and self._transient:
            progressbar.update(self._task, visible=False)

        # 执行刷新操作
        self.refresh()
```