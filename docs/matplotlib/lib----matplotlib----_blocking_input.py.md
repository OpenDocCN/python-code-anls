# `D:\src\scipysrc\matplotlib\lib\matplotlib\_blocking_input.py`

```py
# 定义一个函数，用于阻塞式地运行图形对象的事件循环，并监听交互事件。
# 
# 参数：
# figure : `~matplotlib.figure.Figure`
#     要运行事件循环的图形对象。
# event_names : list of str
#     需要传递给 *handler* 的事件名称列表。
# timeout : float
#     如果为正数，则在 *timeout* 秒后停止事件循环。
# handler : Callable[[Event], Any]
#     每个事件调用的函数；可以通过调用 ``canvas.stop_event_loop()`` 强制提前退出事件循环。

def blocking_input_loop(figure, event_names, timeout, handler):
    # 如果图形对象有 canvas 管理，则确保显示该图形。
    if figure.canvas.manager:
        figure.show()  # 确保图形被显示，如果正在管理它。
    
    # 将事件连接到 on_event 函数调用。
    cids = [figure.canvas.mpl_connect(name, handler) for name in event_names]
    
    try:
        figure.canvas.start_event_loop(timeout)  # 开始事件循环。
    finally:  # 即使出现异常（如 ctrl-c），也会执行此代码块。
        # 断开回调连接。
        for cid in cids:
            figure.canvas.mpl_disconnect(cid)
```