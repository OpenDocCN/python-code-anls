# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_gil.py`

```
```python`
# 导入必要的库和模块
import itertools  # 提供迭代工具的函数
import threading  # 提供线程支持的模块
import time  # 时间相关的功能

import numpy as np  # 用于数值计算的库
from numpy.testing import assert_equal  # NumPy 的测试工具函数
import pytest  # Python 的单元测试框架
import scipy.interpolate  # SciPy 的插值模块


class TestGIL:
    """检查 scipy.interpolate 函数是否正确释放全局解释器锁（GIL）。"""

    def setup_method(self):
        # 初始化一个空列表，用于记录日志消息
        self.messages = []

    def log(self, message):
        # 将传入的消息添加到日志消息列表中
        self.messages.append(message)

    def make_worker_thread(self, target, args):
        # 内部函数，创建并返回一个工作线程对象
        log = self.log

        class WorkerThread(threading.Thread):
            def run(self):
                log('interpolation started')  # 记录线程开始插值的消息
                target(*args)  # 调用目标函数，并传入参数
                log('interpolation complete')  # 记录线程完成插值的消息

        return WorkerThread()

    @pytest.mark.xslow  # 标记为慢速测试
    @pytest.mark.xfail(reason='race conditions, may depend on system load')  # 标记为预期失败，原因是竞争条件可能依赖于系统负载
    def test_rectbivariatespline(self):
        def generate_params(n_points):
            # 生成用于插值的参数：x、y均匀分布在0到1000之间，生成网格并计算z
            x = y = np.linspace(0, 1000, n_points)
            x_grid, y_grid = np.meshgrid(x, y)
            z = x_grid * y_grid
            return x, y, z

        def calibrate_delay(requested_time):
            # 调整延迟，确保插值操作至少运行指定时间
            for n_points in itertools.count(5000, 1000):
                args = generate_params(n_points)
                time_started = time.time()
                interpolate(*args)
                if time.time() - time_started > requested_time:
                    return args

        def interpolate(x, y, z):
            # 执行矩形二元样条插值
            scipy.interpolate.RectBivariateSpline(x, y, z)

        # 调用 calibrate_delay 函数获取插值参数
        args = calibrate_delay(requested_time=3)
        # 创建工作线程，执行插值操作
        worker_thread = self.make_worker_thread(interpolate, args)
        worker_thread.start()  # 启动工作线程
        for i in range(3):
            time.sleep(0.5)
            self.log('working')  # 记录主线程工作状态
        worker_thread.join()  # 等待工作线程结束
        # 断言日志消息的顺序是否符合预期
        assert_equal(self.messages, [
            'interpolation started',
            'working',
            'working',
            'working',
            'interpolation complete',
        ])
```