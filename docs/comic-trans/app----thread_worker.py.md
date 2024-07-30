# `.\comic-translate\app\thread_worker.py`

```py
from PySide6.QtCore import QRunnable, Slot, Signal, QObject, QThreadPool
import traceback, sys
import types

class WorkerSignals(QObject):
    # 定义信号，用于通知任务完成
    finished = Signal()
    # 定义信号，用于传递错误信息元组
    error = Signal(tuple)
    # 定义信号，用于传递任务结果对象
    result = Signal(object)

class GenericWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(GenericWorker, self).__init__()
        # 保存任务函数
        self.fn = fn
        # 保存任务函数的位置参数
        self.args = args
        # 保存任务函数的关键字参数
        self.kwargs = kwargs
        # 创建信号实例，用于任务状态的通知
        self.signals = WorkerSignals()
        # 标志位，用于标识是否取消任务
        self.is_cancelled = False

    def run(self):
        try:
            # 如果任务函数是生成器类型，则逐步处理生成器结果
            if isinstance(self.fn, types.GeneratorType):
                result = None
                for r in self.fn:
                    # 如果取消标志位被设置，中断处理
                    if self.is_cancelled:
                        break
                    result = r
                # 如果未取消，则发送最后一个生成器结果
                if not self.is_cancelled:
                    self.signals.result.emit(result)
            else:
                # 否则，执行一般的函数调用并发送结果
                result = self.fn(*self.args, **self.kwargs)
                if not self.is_cancelled:
                    self.signals.result.emit(result)
        except:
            # 捕获任何异常并发送错误信号
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            # 无论是否发生异常，最终都发送完成信号
            self.signals.finished.emit()

    def cancel(self):
        # 设置取消标志位，用于中断任务执行
        self.is_cancelled = True
```