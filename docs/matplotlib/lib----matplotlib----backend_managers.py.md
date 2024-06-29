# `D:\src\scipysrc\matplotlib\lib\matplotlib\backend_managers.py`

```
from matplotlib import _api, backend_tools, cbook, widgets


class ToolEvent:
    """工具操作事件（添加/删除）。"""
    def __init__(self, name, sender, tool, data=None):
        self.name = name
        self.sender = sender
        self.tool = tool
        self.data = data


class ToolTriggerEvent(ToolEvent):
    """通知工具已被触发的事件。"""
    def __init__(self, name, sender, tool, canvasevent=None, data=None):
        super().__init__(name, sender, tool, data)
        self.canvasevent = canvasevent


class ToolManagerMessageEvent:
    """
    从工具管理器传递消息的事件。

    这些消息通常由工具栏显示给用户。
    """
    def __init__(self, name, sender, message):
        self.name = name
        self.sender = sender
        self.message = message


class ToolManager:
    """
    管理用户与图表（Figure）交互触发的操作（按键按下、工具栏点击等）。

    Attributes
    ----------
    figure : `.Figure`
        图表对象。
    keypresslock : `~matplotlib.widgets.LockDraw`
        控制是否锁定 `canvas` 的 `key_press_event` 的 `.LockDraw` 对象。
    messagelock : `~matplotlib.widgets.LockDraw`
        控制消息是否可用于写入的 `.LockDraw` 对象。
    """

    def __init__(self, figure=None):

        self._key_press_handler_id = None

        self._tools = {}
        self._keys = {}
        self._toggled = {}
        self._callbacks = cbook.CallbackRegistry()

        # 处理按键事件
        self.keypresslock = widgets.LockDraw()
        self.messagelock = widgets.LockDraw()

        self._figure = None
        self.set_figure(figure)

    @property
    def canvas(self):
        """由 FigureManager 管理的画布。"""
        if not self._figure:
            return None
        return self._figure.canvas

    @property
    def figure(self):
        """持有画布的图表对象。"""
        return self._figure

    @figure.setter
    def figure(self, figure):
        self.set_figure(figure)

    def set_figure(self, figure, update_tools=True):
        """
        将给定的图表绑定到工具。

        Parameters
        ----------
        figure : `.Figure`
            要绑定的图表对象。
        update_tools : bool, default: True
            是否强制更新工具以匹配新图表。
        """
        if self._key_press_handler_id:
            self.canvas.mpl_disconnect(self._key_press_handler_id)
        self._figure = figure
        if figure:
            self._key_press_handler_id = self.canvas.mpl_connect(
                'key_press_event', self._key_press)
        if update_tools:
            for tool in self._tools.values():
                tool.figure = figure
    def toolmanager_connect(self, s, func):
        """
        Connect event with string *s* to *func*.

        Parameters
        ----------
        s : str
            The name of the event. The following events are recognized:

            - 'tool_message_event'
            - 'tool_removed_event'
            - 'tool_added_event'

            For every tool added a new event is created

            - 'tool_trigger_TOOLNAME', where TOOLNAME is the id of the tool.

        func : callable
            Callback function for the toolmanager event with signature::

                def func(event: ToolEvent) -> Any

        Returns
        -------
        cid
            The callback id for the connection. This can be used in
            `.toolmanager_disconnect`.
        """
        # 将事件字符串 s 和对应回调函数 func 进行连接，返回连接的回调 ID
        return self._callbacks.connect(s, func)

    def toolmanager_disconnect(self, cid):
        """
        Disconnect callback id *cid*.

        Example usage::

            cid = toolmanager.toolmanager_connect('tool_trigger_zoom', onpress)
            #...later
            toolmanager.toolmanager_disconnect(cid)
        """
        # 根据给定的回调 ID 断开连接
        return self._callbacks.disconnect(cid)

    def message_event(self, message, sender=None):
        """Emit a `ToolManagerMessageEvent`."""
        if sender is None:
            sender = self

        s = 'tool_message_event'
        # 创建 ToolManagerMessageEvent 实例，并传递给回调处理
        event = ToolManagerMessageEvent(s, sender, message)
        self._callbacks.process(s, event)

    @property
    def active_toggle(self):
        """Currently toggled tools."""
        # 返回当前处于切换状态的工具列表
        return self._toggled

    def get_tool_keymap(self, name):
        """
        Return the keymap associated with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.

        Returns
        -------
        list of str
            List of keys associated with the tool.
        """
        # 根据工具名称获取与之关联的键映射列表
        keys = [k for k, i in self._keys.items() if i == name]
        return keys

    def _remove_keys(self, name):
        # 私有方法：移除与指定工具名称关联的所有键映射
        for k in self.get_tool_keymap(name):
            del self._keys[k]

    def update_keymap(self, name, key):
        """
        Set the keymap to associate with the specified tool.

        Parameters
        ----------
        name : str
            Name of the Tool.
        key : str or list of str
            Keys to associate with the tool.
        """
        if name not in self._tools:
            raise KeyError(f'{name!r} not in Tools')
        # 更新工具的键映射关系
        self._remove_keys(name)
        if isinstance(key, str):
            key = [key]
        for k in key:
            if k in self._keys:
                _api.warn_external(
                    f'Key {k} changed from {self._keys[k]} to {name}')
            self._keys[k] = name
    # 从工具管理器中移除名称为 *name* 的工具。
    def remove_tool(self, name):
        # 获取名为 *name* 的工具对象
        tool = self.get_tool(name)
        # 如果工具是一个已切换的切换工具，则取消切换状态
        if getattr(tool, 'toggled', False):
            self.trigger_tool(tool, 'toolmanager')  # 触发工具管理器中的工具
        # 从工具管理器中移除键为 *name* 的工具
        self._remove_keys(name)
        # 创建一个工具事件，表示工具被移除
        event = ToolEvent('tool_removed_event', self, tool)
        # 处理事件的回调函数
        self._callbacks.process(event.name, event)
        # 从工具字典中删除键为 *name* 的工具对象
        del self._tools[name]

    # 向工具管理器中添加工具 *tool*。
    def add_tool(self, name, tool, *args, **kwargs):
        # 根据当前画布类型查找工具类
        tool_cls = backend_tools._find_tool_class(type(self.canvas), tool)
        # 如果找不到工具类，则抛出值错误
        if not tool_cls:
            raise ValueError('Impossible to find class for %s' % str(tool))

        # 如果 *name* 已经存在于工具字典中，则发出警告并返回已存在的工具对象
        if name in self._tools:
            _api.warn_external('A "Tool class" with the same name already '
                               'exists, not added')
            return self._tools[name]

        # 使用工具类创建工具对象
        tool_obj = tool_cls(self, name, *args, **kwargs)
        # 将工具对象添加到工具字典中
        self._tools[name] = tool_obj

        # 如果工具对象有默认按键映射，则更新按键映射
        if tool_obj.default_keymap is not None:
            self.update_keymap(name, tool_obj.default_keymap)

        # 对于切换工具，初始化在 self._toggled 中的 radio_group
        if isinstance(tool_obj, backend_tools.ToolToggleBase):
            # 如果 radio_group 是 None，则表示非互斥组，使用集合来跟踪此组中的所有切换工具
            if tool_obj.radio_group is None:
                self._toggled.setdefault(None, set())
            else:
                self._toggled.setdefault(tool_obj.radio_group, None)

            # 如果初始处于切换状态
            if tool_obj.toggled:
                self._handle_toggle(tool_obj, None, None)
        
        # 将图形对象设置给工具对象
        tool_obj.set_figure(self.figure)

        # 创建一个工具事件，表示工具被添加
        event = ToolEvent('tool_added_event', self, tool_obj)
        # 处理事件的回调函数
        self._callbacks.process(event.name, event)

        # 返回添加的工具对象
        return tool_obj
    def _handle_toggle(self, tool, canvasevent, data):
        """
        Toggle tools, need to untoggle prior to using other Toggle tool.
        Called from trigger_tool.

        Parameters
        ----------
        tool : `.ToolBase`
            The tool instance to toggle.
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """

        radio_group = tool.radio_group
        # radio_group None is not mutually exclusive
        # just keep track of toggled tools in this group
        if radio_group is None:
            if tool.name in self._toggled[None]:
                self._toggled[None].remove(tool.name)
            else:
                self._toggled[None].add(tool.name)
            return

        # If the tool already has a toggled state, untoggle it
        if self._toggled[radio_group] == tool.name:
            toggled = None
        # If no tool was toggled in the radio_group
        # toggle it
        elif self._toggled[radio_group] is None:
            toggled = tool.name
        # Other tool in the radio_group is toggled
        else:
            # Untoggle previously toggled tool by triggering its untoggle action
            self.trigger_tool(self._toggled[radio_group],
                              self,
                              canvasevent,
                              data)
            toggled = tool.name

        # Keep track of the toggled tool in the radio_group
        self._toggled[radio_group] = toggled


```    
    def trigger_tool(self, name, sender=None, canvasevent=None, data=None):
        """
        Trigger a tool and emit the ``tool_trigger_{name}`` event.

        Parameters
        ----------
        name : str
            Name of the tool to trigger.
        sender : object
            Object that wishes to trigger the tool (default is self).
        canvasevent : Event
            Original Canvas event or None.
        data : object
            Extra data to pass to the tool when triggering.
        """
        # Retrieve the tool object associated with the given name
        tool = self.get_tool(name)
        if tool is None:
            return

        if sender is None:
            sender = self

        # If the tool is a toggleable tool, handle its toggling mechanism
        if isinstance(tool, backend_tools.ToolToggleBase):
            self._handle_toggle(tool, canvasevent, data)

        # Actually trigger the tool's action
        tool.trigger(sender, canvasevent, data)

        # Emit a tool trigger event
        s = 'tool_trigger_%s' % name
        event = ToolTriggerEvent(s, sender, tool, canvasevent, data)
        self._callbacks.process(s, event)


```    
    def _key_press(self, event):
        """
        Handle key press events to trigger associated tools.

        Parameters
        ----------
        event : KeyEvent
            Key press event containing information about the pressed key.
        """
        # Ignore if the key is None or if keypress is locked
        if event.key is None or self.keypresslock.locked():
            return

        # Retrieve the tool name associated with the pressed key
        name = self._keys.get(event.key, None)
        if name is None:
            return
        
        # Trigger the tool associated with the pressed key
        self.trigger_tool(name, canvasevent=event)


```    
    @property
    def tools(self):
        """A dict mapping tool name -> controlled tool."""
        # Return the dictionary of tools managed by this object
        return self._tools
    # 返回具有指定名称的工具对象。
    def get_tool(self, name, warn=True):
        """
        Return the tool object with the given name.

        For convenience, this passes tool objects through.

        Parameters
        ----------
        name : str or `.ToolBase`
            Name of the tool, or the tool itself.
        warn : bool, default: True
            Whether a warning should be emitted it no tool with the given name
            exists.

        Returns
        -------
        `.ToolBase` or None
            The tool or None if no tool with the given name exists.
        """
        # 如果名称是 `.ToolBase` 类型并且具有该名称的工具存在于 self._tools 中，则直接返回该工具对象。
        if (isinstance(name, backend_tools.ToolBase)
                and name.name in self._tools):
            return name
        # 如果名称不在 self._tools 中：
        if name not in self._tools:
            # 如果 warn 参数为 True，则发出警告，说明 ToolManager 无法控制具有给定名称的工具。
            if warn:
                _api.warn_external(
                    f"ToolManager does not control tool {name!r}")
            # 返回 None，表示未找到具有给定名称的工具。
            return None
        # 返回 self._tools 中对应名称的工具对象。
        return self._tools[name]
```