# `.\DB-GPT-src\dbgpt\util\tracer\span_storage.py`

```py
import datetime
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import List, Optional

from dbgpt.component import SystemApp
from dbgpt.util.tracer.base import Span, SpanStorage

logger = logging.getLogger(__name__)


class MemorySpanStorage(SpanStorage):
    def __init__(self, system_app: SystemApp | None = None):
        super().__init__(system_app)
        self.spans = []  # 初始化一个空列表用于存储 Span 对象
        self._lock = threading.Lock()  # 创建一个线程锁对象

    def append_span(self, span: Span):
        with self._lock:
            self.spans.append(span)  # 添加一个 Span 对象到 spans 列表中


class SpanStorageContainer(SpanStorage):
    def __init__(
        self,
        system_app: SystemApp | None = None,
        batch_size=10,
        flush_interval=10,
        executor: Executor = None,
    ):
        super().__init__(system_app)
        if not executor:
            executor = ThreadPoolExecutor(thread_name_prefix="trace_storage_sync_")  # 如果没有指定 executor，则创建一个线程池执行器
        self.executor = executor  # 存储传入的 executor 对象
        self.storages: List[SpanStorage] = []  # 初始化一个空列表用于存储 SpanStorage 对象
        self.last_date = (
            datetime.datetime.now().date()
        )  # 存储当前日期对象，用于检查日期变化
        self.queue = queue.Queue()  # 创建一个队列对象
        self.batch_size = batch_size  # 存储批处理大小参数
        self.flush_interval = flush_interval  # 存储刷新间隔参数
        self.last_flush_time = time.time()  # 存储上次刷新时间戳
        self.flush_signal_queue = queue.Queue()  # 创建一个信号队列对象
        self.flush_thread = threading.Thread(
            target=self._flush_to_storages, daemon=True
        )  # 创建一个线程对象用于后台刷新存储
        self._stop_event = threading.Event()  # 创建一个线程事件对象
        self.flush_thread.start()  # 启动刷新线程
        self._stop_event.clear()  # 清除停止事件的状态

    def append_storage(self, storage: SpanStorage):
        """Append sotrage to container

        Args:
            storage ([`SpanStorage`]): The storage to be append to current container
        """
        self.storages.append(storage)  # 将传入的 SpanStorage 对象添加到 storages 列表中

    def append_span(self, span: Span):
        self.queue.put(span)  # 将传入的 Span 对象放入队列中
        if self.queue.qsize() >= self.batch_size:
            try:
                self.flush_signal_queue.put_nowait(True)  # 如果队列大小达到批处理大小，则向刷新信号队列发送刷新信号
            except queue.Full:
                pass  # 如果信号队列已满，则忽略，刷新线程会处理这种情况。
    # 当不停止事件未设置时，持续执行以下操作
    def _flush_to_storages(self):
        while not self._stop_event.is_set():
            # 计算距上次刷新时间的间隔
            interval = time.time() - self.last_flush_time
            # 若间隔小于刷新间隔，则尝试从刷新信号队列获取信号
            if interval < self.flush_interval:
                try:
                    self.flush_signal_queue.get(
                        block=True, timeout=self.flush_interval - interval
                    )
                except Exception:
                    # 捕获超时异常
                    pass

            # 初始化用于存储待写入数据的列表
            spans_to_write = []
            # 将队列中的所有数据取出并添加到待写入数据列表中
            while not self.queue.empty():
                spans_to_write.append(self.queue.get())
            # 遍历所有存储对象
            for s in self.storages:

                # 定义函数：向存储对象追加数据并忽略错误
                def append_and_ignore_error(
                    storage: SpanStorage, spans_to_write: List[SpanStorage]
                ):
                    try:
                        storage.append_span_batch(spans_to_write)
                    except Exception as e:
                        # 记录警告日志，指出追加数据到存储对象失败的原因
                        logger.warning(
                            f"Append spans to storage {str(storage)} failed: {str(e)}, span_data: {spans_to_write}"
                        )

                try:
                    # 提交任务到执行器以异步执行追加数据操作
                    self.executor.submit(append_and_ignore_error, s, spans_to_write)
                except RuntimeError:
                    # 若无法提交任务，则直接调用函数执行追加数据操作
                    append_and_ignore_error(s, spans_to_write)
            # 更新最后刷新时间为当前时间
            self.last_flush_time = time.time()

    # 在停止前的清理工作
    def before_stop(self):
        try:
            # 向刷新信号队列中放入True信号
            self.flush_signal_queue.put(True)
            # 设置停止事件
            self._stop_event.set()
            # 等待刷新线程结束
            self.flush_thread.join()
        except Exception:
            # 捕获所有异常，以确保方法的安全执行
            pass
class FileSpanStorage(SpanStorage):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        # 将文件名拆分为前缀和后缀
        self.filename_prefix, self.filename_suffix = os.path.splitext(filename)
        if not self.filename_suffix:
            self.filename_suffix = ".log"
        self.last_date = (
            datetime.datetime.now().date()
        )  # 存储当前日期以检查日期变化
        self.queue = queue.Queue()

        if not os.path.exists(filename):
            # 如果文件不存在则创建新文件
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "a"):
                pass

    def append_span(self, span: Span):
        self._write_to_file([span])

    def append_span_batch(self, spans: List[Span]):
        self._write_to_file(spans)

    def _get_dated_filename(self, date: datetime.date) -> str:
        """根据特定日期返回文件名。"""
        date_str = date.strftime("%Y-%m-%d")
        return f"{self.filename_prefix}_{date_str}{self.filename_suffix}"

    def _roll_over_if_needed(self):
        """检查自上次写入以来是否已经过了一天，如果是，则重命名当前文件。"""
        current_date = datetime.datetime.now().date()
        if current_date != self.last_date:
            if os.path.exists(self.filename):
                os.rename(self.filename, self._get_dated_filename(self.last_date))
            self.last_date = current_date

    def _write_to_file(self, spans: List[Span]):
        self._roll_over_if_needed()

        with open(self.filename, "a", encoding="utf8") as file:
            for span in spans:
                span_data = span.to_dict()
                try:
                    file.write(json.dumps(span_data, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.warning(
                        f"将 span 写入文件失败: {str(e)}, span_data: {span_data}"
                    )
```