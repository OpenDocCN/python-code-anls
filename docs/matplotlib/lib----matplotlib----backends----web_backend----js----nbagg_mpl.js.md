# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\web_backend\js\nbagg_mpl.js`

```py
/* global mpl */  // 引入全局变量 mpl

var comm_websocket_adapter = function (comm) {
    // 创建一个类似 "websocket" 的对象，用于调用给定的 IPython comm 对象的相应方法。
    // 目前这是一个非二进制的 socket，因此仍然有一些性能优化的空间。
    var ws = {};

    ws.binaryType = comm.kernel.ws.binaryType;  // 设置 binaryType 属性为 comm.kernel.ws 的 binaryType 属性
    ws.readyState = comm.kernel.ws.readyState;  // 设置 readyState 属性为 comm.kernel.ws 的 readyState 属性
    function updateReadyState(_event) {
        // 更新 ws 对象的 readyState 属性，根据 comm.kernel.ws 的状态更新
        if (comm.kernel.ws) {
            ws.readyState = comm.kernel.ws.readyState;
        } else {
            ws.readyState = 3; // Closed state. 如果 comm.kernel.ws 不存在，则设置为 3，表示关闭状态。
        }
    }
    comm.kernel.ws.addEventListener('open', updateReadyState);  // 监听 comm.kernel.ws 的 'open' 事件，更新 readyState
    comm.kernel.ws.addEventListener('close', updateReadyState);  // 监听 comm.kernel.ws 的 'close' 事件，更新 readyState
    comm.kernel.ws.addEventListener('error', updateReadyState);  // 监听 comm.kernel.ws 的 'error' 事件，更新 readyState

    ws.close = function () {
        comm.close();  // 关闭 comm 对象
    };
    ws.send = function (m) {
        // 发送消息 m 给 comm 对象
        //console.log('sending', m);  // （注释掉了）打印发送的消息 m
        comm.send(m);
    };
    // 注册 on_msg 回调函数
    comm.on_msg(function (msg) {
        //console.log('receiving', msg['content']['data'], msg);  // （注释掉了）打印接收到的消息和完整消息对象
        var data = msg['content']['data'];
        if (data['blob'] !== undefined) {
            // 如果数据中包含 'blob' 属性，则创建 Blob 对象
            data = {
                data: new Blob(msg['buffers'], { type: data['blob'] }),
            };
        }
        // 将 mpl 事件传递给被 mpl 覆盖的 onmessage 函数
        ws.onmessage(data);
    });
    return ws;  // 返回创建的 websocket 对象
};

mpl.mpl_figure_comm = function (comm, msg) {
    // 当 mpl 进程通过 "matplotlib" 通道启动 IPython Comm 时调用的函数。

    var id = msg.content.data.id;  // 从消息中获取 ID
    // 获取由 Python 中 Comm socket 打开时 display 调用创建的 div 元素。
    var element = document.getElementById(id);  // 获取具有特定 ID 的元素
    var ws_proxy = comm_websocket_adapter(comm);  // 使用 comm_websocket_adapter 函数创建一个 WebSocket 代理

    function ondownload(figure, _format) {
        window.open(figure.canvas.toDataURL());  // 下载图像的回调函数，打开图像的 DataURL
    }

    var fig = new mpl.figure(id, ws_proxy, ondownload, element);  // 创建新的 mpl 图像对象

    // 立即调用 onopen - mpl 需要它，因为它假设我们传递了一个真实的 WebSocket，而不是我们的 websocket->open comm 代理。
    ws_proxy.onopen();  // 调用 ws_proxy 对象的 onopen 方法

    fig.parent_element = element;  // 设置图像的父元素
    fig.cell_info = mpl.find_output_cell("<div id='" + id + "'></div>");  // 查找输出单元格信息

    if (!fig.cell_info) {
        console.error('Failed to find cell for figure', id, fig);  // 如果未找到图像对应的单元格，则输出错误信息
        return;
    }

    // 为图像的输出区域元素注册 'cleared' 事件处理程序，以及参数为 fig 的 _remove_fig_handler 函数。
    fig.cell_info[0].output_area.element.on(
        'cleared',
        { fig: fig },
        fig._remove_fig_handler
    );
};

mpl.figure.prototype.handle_close = function (fig, msg) {
    var width = fig.canvas.width / fig.ratio;  // 计算图像的宽度

    // 取消图像输出区域元素上 'cleared' 事件中的 fig._remove_fig_handler 处理程序
    fig.cell_info[0].output_area.element.off(
        'cleared',
        fig._remove_fig_handler
    );

    fig.resizeObserverInstance.unobserve(fig.canvas_div);  // 停止观察图像的 resizeObserverInstance

    // 更新输出单元格，使用当前画布的数据。
    fig.push_to_output();

    var dataURL = fig.canvas.toDataURL();  // 获取当前画布的 DataURL
    // 重新启用 IPython 中的键盘管理器 - 没有这行，在 Firefox 中，笔记本键盘快捷键会失效。
    IPython.keyboard_manager.enable();
    // 将图表的父元素的 innerHTML 设置为包含图像数据的 img 标签，设定图像宽度为指定的 width。
    fig.parent_element.innerHTML =
        '<img src="' + dataURL + '" width="' + width + '">';
    // 调用 fig 对象的 close_ws 方法，关闭工作空间，并传递 msg 参数。
    fig.close_ws(fig, msg);
};

// 定义在 mpl.figure 的原型上的 close_ws 方法，用于发送关闭消息
mpl.figure.prototype.close_ws = function (fig, msg) {
    fig.send_message('closing', msg);
    // fig.ws.close() 这行被注释掉了，可能是原本打算关闭 WebSocket 连接的代码
};

// 定义在 mpl.figure 的原型上的 push_to_output 方法，将画布上的数据转换为输出单元格中的数据
mpl.figure.prototype.push_to_output = function (_remove_interactive) {
    // 计算画布宽度并生成画布数据的 Data URL
    var width = this.canvas.width / this.ratio;
    var dataURL = this.canvas.toDataURL();
    this.cell_info[1]['text/html'] =
        '<img src="' + dataURL + '" width="' + width + '">';
};

// 定义在 mpl.figure 的原型上的 updated_canvas_event 方法，告知 IPython 笔记本内容已经改变，并将新图像推送到 DOM
mpl.figure.prototype.updated_canvas_event = function () {
    // 设置笔记本为已修改状态
    IPython.notebook.set_dirty(true);
    this.send_message('ack', {});
    var fig = this;
    // 延迟一秒钟后将新图像推送到输出
    setTimeout(function () {
        fig.push_to_output();
    }, 1000);
};

// 定义在 mpl.figure 的原型上的 _init_toolbar 方法，初始化工具栏
mpl.figure.prototype._init_toolbar = function () {
    var fig = this;

    // 创建工具栏元素并添加样式类
    var toolbar = document.createElement('div');
    toolbar.classList = 'btn-toolbar';
    this.root.appendChild(toolbar);

    // 闭包函数，用于处理工具栏按钮的点击事件
    function on_click_closure(name) {
        return function (_event) {
            return fig.toolbar_button_onclick(name);
        };
    }

    // 闭包函数，用于处理工具栏按钮的鼠标悬停事件
    function on_mouseover_closure(tooltip) {
        return function (event) {
            if (!event.currentTarget.disabled) {
                return fig.toolbar_button_onmouseover(tooltip);
            }
        };
    }

    // 初始化按钮对象
    fig.buttons = {};
    var buttonGroup = document.createElement('div');
    buttonGroup.classList = 'btn-group';
    var button;
    // 遍历工具栏项目
    for (var toolbar_ind in mpl.toolbar_items) {
        var name = mpl.toolbar_items[toolbar_ind][0];
        var tooltip = mpl.toolbar_items[toolbar_ind][1];
        var image = mpl.toolbar_items[toolbar_ind][2];
        var method_name = mpl.toolbar_items[toolbar_ind][3];

        // 如果没有按钮名，创建新的按钮组
        if (!name) {
            /* Instead of a spacer, we start a new button group. */
            if (buttonGroup.hasChildNodes()) {
                toolbar.appendChild(buttonGroup);
            }
            buttonGroup = document.createElement('div');
            buttonGroup.classList = 'btn-group';
            continue;
        }

        // 创建按钮并设置属性
        button = fig.buttons[name] = document.createElement('button');
        button.classList = 'btn btn-default';
        button.href = '#';
        button.title = name;
        button.innerHTML = '<i class="fa ' + image + ' fa-lg"></i>';
        button.addEventListener('click', on_click_closure(method_name));
        button.addEventListener('mouseover', on_mouseover_closure(tooltip));
        buttonGroup.appendChild(button);
    }

    // 添加最后一个按钮组到工具栏
    if (buttonGroup.hasChildNodes()) {
        toolbar.appendChild(buttonGroup);
    }

    // 添加状态栏
    var status_bar = document.createElement('span');
    status_bar.classList = 'mpl-message pull-right';
    toolbar.appendChild(status_bar);
    this.message = status_bar;

    // 向窗口添加关闭按钮
    var buttongrp = document.createElement('div');
    # 设置按钮组的类属性，使其样式为按钮组并靠右对齐
    buttongrp.classList = 'btn-group inline pull-right';
    
    # 创建一个新的按钮元素
    button = document.createElement('button');
    
    # 设置按钮的类属性，使其样式为小型的蓝色主题按钮
    button.classList = 'btn btn-mini btn-primary';
    
    # 设置按钮的超链接，但这里按钮元素不应该使用 href 属性，应该使用其它适当的属性或事件绑定
    button.href = '#';
    
    # 设置按钮的标题属性，显示鼠标悬停时的文本提示为“Stop Interaction”
    button.title = 'Stop Interaction';
    
    # 设置按钮的 HTML 内容，包括一个带有图标的关闭按钮
    button.innerHTML = '<i class="fa fa-power-off icon-remove icon-large"></i>';
    
    # 添加点击事件监听器，点击按钮时执行 handle_close 方法，传入 fig 对象和空对象作为参数
    button.addEventListener('click', function (_evt) {
        fig.handle_close(fig, {});
    });
    
    # 添加鼠标悬停事件监听器，当鼠标悬停在按钮上时调用 on_mouseover_closure 函数，并传入“Stop Interaction”作为参数
    button.addEventListener(
        'mouseover',
        on_mouseover_closure('Stop Interaction')
    );
    
    # 将创建好的按钮元素添加到按钮组元素中
    buttongrp.appendChild(button);
    
    # 查找并获取对话框标题栏的元素
    var titlebar = this.root.querySelector('.ui-dialog-titlebar');
    
    # 在对话框标题栏的最前面插入按钮组元素
    titlebar.insertBefore(buttongrp, titlebar.firstChild);
};

// 定义一个函数 _remove_fig_handler，处理图形对象上的事件
mpl.figure.prototype._remove_fig_handler = function (event) {
    // 从事件对象中获取图形对象
    var fig = event.data.fig;
    // 如果事件的目标不是当前元素本身，则忽略（排除来自子元素的冒泡事件）
    if (event.target !== this) {
        return;
    }
    // 调用图形对象的 close_ws 方法，并传入空对象
    fig.close_ws(fig, {});
};

// 定义一个函数 _root_extra_style，为指定元素添加额外样式
mpl.figure.prototype._root_extra_style = function (el) {
    // 设置元素的 box-sizing 样式为 content-box，覆盖笔记本设置的 border-box
    el.style.boxSizing = 'content-box';
};

// 定义一个函数 _canvas_extra_style，为指定元素添加额外样式
mpl.figure.prototype._canvas_extra_style = function (el) {
    // 使 div 元素可聚焦
    el.setAttribute('tabindex', 0);
    // 如果 IPython 的键盘管理器存在，则注册事件监听器
    if (IPython.notebook.keyboard_manager) {
        IPython.notebook.keyboard_manager.register_events(el);
    } else {
        // 兼容 IPython 2.x 的情况下注册事件监听器
        IPython.keyboard_manager.register_events(el);
    }
};

// 定义一个函数 _key_event_extra，处理键盘事件的附加操作
mpl.figure.prototype._key_event_extra = function (event, _name) {
    // 检查是否按下了 Shift+Enter
    if (event.shiftKey && event.which === 13) {
        // 取消当前 canvas_div 的焦点
        this.canvas_div.blur();
        // 选中当前单元格之后的下一个单元格
        var index = IPython.notebook.find_cell_index(this.cell_info[0]);
        IPython.notebook.select(index + 1);
    }
};

// 定义一个函数 handle_save，处理保存操作
mpl.figure.prototype.handle_save = function (fig, _msg) {
    // 调用图形对象的 ondownload 方法，并传入空值
    fig.ondownload(fig, null);
};

// 定义一个函数 find_output_cell，查找包含特定 HTML 输出的单元格
mpl.find_output_cell = function (html_output) {
    // 获取当前笔记本中的所有单元格
    var cells = IPython.notebook.get_cells();
    var ncells = cells.length;
    // 遍历每个单元格
    for (var i = 0; i < ncells; i++) {
        var cell = cells[i];
        // 如果单元格类型是代码
        if (cell.cell_type === 'code') {
            // 遍历该单元格的输出
            for (var j = 0; j < cell.output_area.outputs.length; j++) {
                var data = cell.output_area.outputs[j];
                // 如果输出中有数据属性，则将 data 赋值为其 data 属性
                if (data.data) {
                    data = data.data;
                }
                // 如果输出的 text/html 属性与指定的 html_output 相同，则返回该单元格及相关数据
                if (data['text/html'] === html_output) {
                    return [cell, data, j];
                }
            }
        }
    }
};

// 如果 IPython 笔记本的内核不为 null，则注册 matplotlib 目标通道的通信管理器
if (IPython.notebook.kernel !== null) {
    IPython.notebook.kernel.comm_manager.register_target(
        'matplotlib',
        mpl.mpl_figure_comm
    );
}
```