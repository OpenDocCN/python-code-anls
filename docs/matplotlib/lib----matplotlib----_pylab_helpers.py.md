# `D:\src\scipysrc\matplotlib\lib\matplotlib\_pylab_helpers.py`

```py
    def set_active(cls, manager):
        """
        Set *manager* as the active figure manager.

        This method updates the order of managers in `cls.figs` so that
        *manager* is moved to the end, making it the active one.
        """
        if manager.num in cls.figs:
            # Move the manager to the end of OrderedDict to mark it as active
            cls.figs.move_to_end(manager.num)
    @classmethod
    def get_num_fig_managers(cls):
        """Return the number of figures being managed."""
        # 返回当前管理的图形数量
        return len(cls.figs)

    @classmethod
    def get_active(cls):
        """Return the active manager, or *None* if there is no manager."""
        # 返回当前活跃的管理器，如果没有活跃的管理器则返回 None
        return next(reversed(cls.figs.values())) if cls.figs else None

    @classmethod
    def _set_new_active_manager(cls, manager):
        """Adopt *manager* into pyplot and make it the active manager."""
        # 如果管理器没有 "_cidgcf" 属性，则创建一个按钮事件监听
        if not hasattr(manager, "_cidgcf"):
            manager._cidgcf = manager.canvas.mpl_connect(
                "button_press_event", lambda event: cls.set_active(manager))
        # 获取管理器所在的图形对象
        fig = manager.canvas.figure
        # 设置图形对象的编号为管理器的编号
        fig.number = manager.num
        # 获取图形对象的标签
        label = fig.get_label()
        # 如果有标签，则将标签设置为窗口标题
        if label:
            manager.set_window_title(label)
        # 将当前管理器设为活跃管理器
        cls.set_active(manager)

    @classmethod
    def set_active(cls, manager):
        """Make *manager* the active manager."""
        # 将给定的管理器设为活跃管理器
        cls.figs[manager.num] = manager
        # 将该管理器移到字典的末尾，表示它是最新活跃的
        cls.figs.move_to_end(manager.num)

    @classmethod
    def draw_all(cls, force=False):
        """
        Redraw all stale managed figures, or, if *force* is True, all managed
        figures.
        """
        # 遍历所有管理的图形管理器
        for manager in cls.get_all_fig_managers():
            # 如果 force 为 True 或者图形对象需要重新绘制
            if force or manager.canvas.figure.stale:
                # 绘制图形对象
                manager.canvas.draw_idle()
# 在程序退出时注册一个函数，用于销毁所有 GCF（Garbage Collection Framework）对象
atexit.register(Gcf.destroy_all)
```