# `D:\src\scipysrc\matplotlib\galleries\examples\misc\multiprocess_sgskip.py`

```py
"""
===============
Multiprocessing
===============

Demo of using multiprocessing for generating data in one process and
plotting in another.

Written by Robert Cimrman
"""

import multiprocessing as mp  # 导入 multiprocessing 模块
import time  # 导入 time 模块

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块
import numpy as np  # 导入 numpy 模块

# Fixing random state for reproducibility
np.random.seed(19680801)  # 设定随机数种子以便结果可复现

# %%
#
# Processing Class
# ================
#
# This class plots data it receives from a pipe.
#

class ProcessPlotter:
    def __init__(self):
        self.x = []  # 初始化 x 坐标列表
        self.y = []  # 初始化 y 坐标列表

    def terminate(self):
        plt.close('all')  # 关闭所有 matplotlib 窗口

    def call_back(self):
        while self.pipe.poll():  # 当管道有数据时循环
            command = self.pipe.recv()  # 接收管道数据
            if command is None:  # 如果接收到 None 命令
                self.terminate()  # 调用终止方法
                return False  # 返回 False 终止循环
            else:
                self.x.append(command[0])  # 将接收到的 x 数据添加到列表
                self.y.append(command[1])  # 将接收到的 y 数据添加到列表
                self.ax.plot(self.x, self.y, 'ro')  # 在图上绘制红色圆点
        self.fig.canvas.draw()  # 绘制更新后的画布
        return True  # 返回 True 继续循环

    def __call__(self, pipe):
        print('starting plotter...')  # 输出启动信息

        self.pipe = pipe  # 将管道对象保存到实例中
        self.fig, self.ax = plt.subplots()  # 创建图形和坐标轴对象
        timer = self.fig.canvas.new_timer(interval=1000)  # 创建定时器对象
        timer.add_callback(self.call_back)  # 将回调函数添加到定时器
        timer.start()  # 启动定时器

        print('...done')  # 输出完成信息
        plt.show()  # 显示图形界面

# %%
#
# Plotting class
# ==============
#
# This class uses multiprocessing to spawn a process to run code from the
# class above. When initialized, it creates a pipe and an instance of
# ``ProcessPlotter`` which will be run in a separate process.
#
# When run from the command line, the parent process sends data to the spawned
# process which is then plotted via the callback function specified in
# ``ProcessPlotter:__call__``.
#

class NBPlot:
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()  # 创建管道对象
        self.plotter = ProcessPlotter()  # 创建 ProcessPlotter 实例
        self.plot_process = mp.Process(  # 创建子进程
            target=self.plotter, args=(plotter_pipe,), daemon=True)  # 指定进程目标和参数
        self.plot_process.start()  # 启动子进程

    def plot(self, finished=False):
        send = self.plot_pipe.send  # 获取管道发送函数
        if finished:
            send(None)  # 发送 None 表示终止
        else:
            data = np.random.random(2)  # 生成随机数据
            send(data)  # 发送随机数据

def main():
    pl = NBPlot()  # 创建 NBPlot 实例
    for _ in range(10):
        pl.plot()  # 循环调用 plot 方法发送数据
        time.sleep(0.5)  # 等待0.5秒
    pl.plot(finished=True)  # 最后发送结束信号

if __name__ == '__main__':
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")  # 设置 macOS 下的进程启动方式
    main()  # 调用主函数
```