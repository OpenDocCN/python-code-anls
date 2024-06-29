# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\web_backend\js\mpl.js`

```
/* Put everything inside the global mpl namespace */
/* 将所有内容放在全局的 mpl 命名空间中 */
window.mpl = {};

mpl.get_websocket_type = function () {
    /* Define a function to determine the appropriate WebSocket object type supported by the browser */
    /* 定义一个函数以确定浏览器支持的 WebSocket 对象类型 */
    if (typeof WebSocket !== 'undefined') {
        return WebSocket;
    } else if (typeof MozWebSocket !== 'undefined') {
        return MozWebSocket;
    } else {
        /* Alert the user if WebSocket support is not detected */
        /* 如果未检测到 WebSocket 支持，则提示用户 */
        alert(
            'Your browser does not have WebSocket support. ' +
                'Please try Chrome, Safari or Firefox ≥ 6. ' +
                'Firefox 4 and 5 are also supported but you ' +
                'have to enable WebSockets in about:config.'
        );
    }
};

mpl.figure = function (figure_id, websocket, ondownload, parent_element) {
    /* Constructor function for creating a figure object within mpl namespace */
    /* 用于在 mpl 命名空间内创建图形对象的构造函数 */

    this.id = figure_id;  // Initialize figure ID
    this.ws = websocket;  // Set WebSocket object for communication

    this.supports_binary = this.ws.binaryType !== undefined;  // Check if binary websocket messages are supported

    if (!this.supports_binary) {
        /* Display a warning if binary websocket messages are not supported */
        /* 如果不支持二进制 websocket 消息，则显示警告 */
        var warnings = document.getElementById('mpl-warnings');
        if (warnings) {
            warnings.style.display = 'block';
            warnings.textContent =
                'This browser does not support binary websocket messages. ' +
                'Performance may be slow.';
        }
    }

    this.imageObj = new Image();  // Create an image object for rendering

    // Initialize various properties related to rendering and interaction
    this.context = undefined;
    this.message = undefined;
    this.canvas = undefined;
    this.rubberband_canvas = undefined;
    this.rubberband_context = undefined;
    this.format_dropdown = undefined;

    this.image_mode = 'full';  // Set default image rendering mode

    // Create a root div element for the figure and append it to the parent element
    this.root = document.createElement('div');
    this.root.setAttribute('style', 'display: inline-block');
    this._root_extra_style(this.root);

    parent_element.appendChild(this.root);  // Append the root div to the specified parent element

    // Initialize header, canvas, and toolbar components of the figure
    this._init_header(this);
    this._init_canvas(this);
    this._init_toolbar(this);

    var fig = this;  // Reference to the current figure object

    this.waiting = false;  // Initialize waiting state flag

    // WebSocket event handler for when the connection is opened
    this.ws.onopen = function () {
        // Send initial messages to the server upon websocket connection
        fig.send_message('supports_binary', { value: fig.supports_binary });
        fig.send_message('send_image_mode', {});
        if (fig.ratio !== 1) {
            fig.send_message('set_device_pixel_ratio', {
                device_pixel_ratio: fig.ratio,
            });
        }
        fig.send_message('refresh', {});
    };

    // Image load event handler for the figure's image object
    this.imageObj.onload = function () {
        // Clear the canvas if rendering a full image to prevent ghosting
        if (fig.image_mode === 'full') {
            fig.context.clearRect(0, 0, fig.canvas.width, fig.canvas.height);
        }
        fig.context.drawImage(fig.imageObj, 0, 0);  // Draw the loaded image onto the canvas
    };

    // Image unload event handler for closing the WebSocket connection
    this.imageObj.onunload = function () {
        fig.ws.close();  // Close the WebSocket connection
    };

    // WebSocket message event handler using a helper function
    this.ws.onmessage = this._make_on_message_function(this);

    this.ondownload = ondownload;  // Set the ondownload callback function
};

mpl.figure.prototype._init_header = function () {
    /* Initialize the header component of the figure */
    /* 初始化图形的标题栏组件 */
    var titlebar = document.createElement('div');  // Create a new div element for the title bar
    titlebar.classList =
        'ui-dialog-titlebar ui-widget-header ui-corner-all ui-helper-clearfix';  // Set CSS classes for styling
    var titletext = document.createElement('div');  // Create a new div element for the title text
    titletext.classList = 'ui-dialog-title';  // Set CSS class for styling the title text
};
    # 设置标题文本的样式，使其宽度占满父元素，居中显示，并添加内边距
    titletext.setAttribute(
        'style',
        'width: 100%; text-align: center; padding: 3px;'
    );
    # 将标题文本添加到标题栏元素中
    titlebar.appendChild(titletext);
    # 将标题栏元素添加到当前对象的根元素中
    this.root.appendChild(titlebar);
    # 将标题文本元素设置为当前对象的头部属性
    this.header = titletext;
};

// 定义 mpl.figure 原型对象的 _canvas_extra_style 方法，用于定义画布容器的额外样式
mpl.figure.prototype._canvas_extra_style = function (_canvas_div) {};

// 定义 mpl.figure 原型对象的 _root_extra_style 方法，用于定义根元素的额外样式
mpl.figure.prototype._root_extra_style = function (_canvas_div) {};

// 定义 mpl.figure 原型对象的 _init_canvas 方法，用于初始化画布
mpl.figure.prototype._init_canvas = function () {
    var fig = this;

    // 创建一个 div 元素作为画布容器
    var canvas_div = (this.canvas_div = document.createElement('div'));
    canvas_div.setAttribute('tabindex', '0');
    canvas_div.setAttribute(
        'style',
        'border: 1px solid #ddd;' +
            'box-sizing: content-box;' +
            'clear: both;' +
            'min-height: 1px;' +
            'min-width: 1px;' +
            'outline: 0;' +
            'overflow: hidden;' +
            'position: relative;' +
            'resize: both;' +
            'z-index: 2;'
    );

    // 创建处理键盘事件的闭包函数
    function on_keyboard_event_closure(name) {
        return function (event) {
            return fig.key_event(event, name);
        };
    }

    // 添加键盘事件监听器
    canvas_div.addEventListener(
        'keydown',
        on_keyboard_event_closure('key_press')
    );
    canvas_div.addEventListener(
        'keyup',
        on_keyboard_event_closure('key_release')
    );

    // 调用 _canvas_extra_style 方法，设置画布容器的额外样式
    this._canvas_extra_style(canvas_div);
    // 将画布容器添加到根元素中
    this.root.appendChild(canvas_div);

    // 创建 canvas 元素作为实际的绘图画布
    var canvas = (this.canvas = document.createElement('canvas'));
    canvas.classList.add('mpl-canvas');
    canvas.setAttribute(
        'style',
        'box-sizing: content-box;' +
            'pointer-events: none;' +
            'position: relative;' +
            'z-index: 0;'
    );

    // 获取 2D 绘图上下文
    this.context = canvas.getContext('2d');

    // 获取设备像素比例
    var backingStore =
        this.context.backingStorePixelRatio ||
        this.context.webkitBackingStorePixelRatio ||
        this.context.mozBackingStorePixelRatio ||
        this.context.msBackingStorePixelRatio ||
        this.context.oBackingStorePixelRatio ||
        this.context.backingStorePixelRatio ||
        1;

    // 计算画布的像素比例
    this.ratio = (window.devicePixelRatio || 1) / backingStore;

    // 创建用于拖动选择的橡皮筋效果 canvas 元素
    var rubberband_canvas = (this.rubberband_canvas = document.createElement(
        'canvas'
    ));
    rubberband_canvas.setAttribute(
        'style',
        'box-sizing: content-box;' +
            'left: 0;' +
            'pointer-events: none;' +
            'position: absolute;' +
            'top: 0;' +
            'z-index: 1;'
    );

    // 如果浏览器不支持 ResizeObserver，则应用一个 ponyfill
    if (this.ResizeObserver === undefined) {
        if (window.ResizeObserver !== undefined) {
            this.ResizeObserver = window.ResizeObserver;
        } else {
            var obs = _JSXTOOLS_RESIZE_OBSERVER({});
            this.ResizeObserver = obs.ResizeObserver;
        }
    }
};
    // 创建 ResizeObserver 实例，监视元素尺寸变化，执行回调函数处理变化
    this.resizeObserverInstance = new this.ResizeObserver(function (entries) {
        // WebSocket 连接未建立时无需进行大小调整：
        // - 如果 WebSocket 正在连接中，Python 连接成功后会进行初始调整。
        // - 如果 WebSocket 已断开连接，调整大小会清除画布，并且无法重新填充，因此最好不要调整大小，保持可见性。
        if (fig.ws.readyState != 1) {
            return;
        }
        var nentries = entries.length;
        // 遍历所有尺寸变化的元素
        for (var i = 0; i < nentries; i++) {
            var entry = entries[i];
            var width, height;
            // 根据不同浏览器实现版本获取内容框尺寸
            if (entry.contentBoxSize) {
                if (Array.isArray(entry.contentBoxSize)) {
                    // Chrome 84 实现了新版规范
                    width = entry.contentBoxSize[0].inlineSize;
                    height = entry.contentBoxSize[0].blockSize;
                } else {
                    // Firefox 实现了旧版规范
                    width = entry.contentBoxSize.inlineSize;
                    height = entry.contentBoxSize.blockSize;
                }
            } else {
                // Chrome <84 实现了更早的规范
                width = entry.contentRect.width;
                height = entry.contentRect.height;
            }

            // 保持画布和橡皮筋画布与容器尺寸同步
            if (entry.devicePixelContentBoxSize) {
                // Chrome 84 实现了新版规范
                canvas.setAttribute(
                    'width',
                    entry.devicePixelContentBoxSize[0].inlineSize
                );
                canvas.setAttribute(
                    'height',
                    entry.devicePixelContentBoxSize[0].blockSize
                );
            } else {
                // 其他浏览器或旧版规范下调整画布大小
                canvas.setAttribute('width', width * fig.ratio);
                canvas.setAttribute('height', height * fig.ratio);
            }

            /* 将画布重新缩放到显示像素大小，以在 HiDPI 屏幕上显示正确 */
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';

            // 调整橡皮筋画布大小
            rubberband_canvas.setAttribute('width', width);
            rubberband_canvas.setAttribute('height', height);

            // 向 Python 更新尺寸信息。忽略初始的 0/0 尺寸，这种情况由于最小尺寸样式设置通常不会发生。
            if (width != 0 && height != 0) {
                fig.request_resize(width, height);
            }
        }
    });
    // 开始观察指定的 canvas_div 元素
    this.resizeObserverInstance.observe(canvas_div);
    /* 创建一个闭包函数，根据浏览器类型返回处理鼠标事件的函数
     * 在 WebKit 中，由于 bug，需要特殊处理以避免额外的浏览器行为
     */
    function on_mouse_event_closure(name) {
        // 获取用户代理信息
        var UA = navigator.userAgent;
        // 检查是否为 WebKit 内核且不是 Chrome 浏览器
        var isWebKit = /AppleWebKit/.test(UA) && !/Chrome/.test(UA);
        if(isWebKit) {
            // 如果是 WebKit，返回一个处理事件的函数，阻止默认行为并调用 fig.mouse_event
            return function (event) {
                /* 阻止浏览器在按钮按下时自动切换到文本插入光标状态
                 * 所有的光标设置通过 'cursor' 事件从 matplotlib 手动控制
                 */
                event.preventDefault();
                return fig.mouse_event(event, name);
            };
        } else {
            // 非 WebKit 浏览器，直接调用 fig.mouse_event 处理事件
            return function (event) {
                return fig.mouse_event(event, name);
            };
        }
    }

    // 给 canvas_div 添加鼠标按下事件监听器，使用 on_mouse_event_closure 处理 'button_press' 事件
    canvas_div.addEventListener(
        'mousedown',
        on_mouse_event_closure('button_press')
    );
    // 给 canvas_div 添加鼠标释放事件监听器，使用 on_mouse_event_closure 处理 'button_release' 事件
    canvas_div.addEventListener(
        'mouseup',
        on_mouse_event_closure('button_release')
    );
    // 给 canvas_div 添加双击事件监听器，使用 on_mouse_event_closure 处理 'dblclick' 事件
    canvas_div.addEventListener(
        'dblclick',
        on_mouse_event_closure('dblclick')
    );
    // 给 canvas_div 添加鼠标移动事件监听器，使用 on_mouse_event_closure 处理 'motion_notify' 事件
    // 将鼠标事件限流到每20毫秒一个
    canvas_div.addEventListener(
        'mousemove',
        on_mouse_event_closure('motion_notify')
    );

    // 给 canvas_div 添加鼠标进入事件监听器，使用 on_mouse_event_closure 处理 'figure_enter' 事件
    canvas_div.addEventListener(
        'mouseenter',
        on_mouse_event_closure('figure_enter')
    );
    // 给 canvas_div 添加鼠标离开事件监听器，使用 on_mouse_event_closure 处理 'figure_leave' 事件
    canvas_div.addEventListener(
        'mouseleave',
        on_mouse_event_closure('figure_leave')
    );

    // 给 canvas_div 添加滚轮事件监听器，根据滚动方向设置 event.step，使用 on_mouse_event_closure 处理 'scroll' 事件
    canvas_div.addEventListener('wheel', function (event) {
        if (event.deltaY < 0) {
            event.step = 1;
        } else {
            event.step = -1;
        }
        on_mouse_event_closure('scroll')(event);
    });

    // 将 canvas 和 rubberband_canvas 添加到 canvas_div 中
    canvas_div.appendChild(canvas);
    canvas_div.appendChild(rubberband_canvas);

    // 获取 rubberband_canvas 的 2D 渲染上下文并设置默认描边样式
    this.rubberband_context = rubberband_canvas.getContext('2d');
    this.rubberband_context.strokeStyle = '#000000';

    // 定义一个函数 _resize_canvas，根据参数调整 canvas_div 的尺寸
    this._resize_canvas = function (width, height, forward) {
        if (forward) {
            canvas_div.style.width = width + 'px';
            canvas_div.style.height = height + 'px';
        }
    };

    // 禁用右键菜单事件，阻止默认行为
    canvas_div.addEventListener('contextmenu', function (_e) {
        event.preventDefault();
        return false;
    });

    // 设置焦点到 canvas 和 canvas_div
    function set_focus() {
        canvas.focus();
        canvas_div.focus();
    }

    // 延迟100毫秒后调用 set_focus 设置焦点
    window.setTimeout(set_focus, 100);
};

mpl.figure.prototype._init_toolbar = function () {
    // 将当前对象保存到变量fig中
    var fig = this;

    // 创建一个新的div元素作为工具栏，并添加到根元素下
    var toolbar = document.createElement('div');
    toolbar.classList = 'mpl-toolbar';
    this.root.appendChild(toolbar);

    // 定义处理按钮点击事件的闭包函数
    function on_click_closure(name) {
        return function (_event) {
            return fig.toolbar_button_onclick(name);
        };
    }

    // 定义处理鼠标悬停事件的闭包函数
    function on_mouseover_closure(tooltip) {
        return function (event) {
            // 如果当前按钮没有禁用，则触发悬停事件处理函数
            if (!event.currentTarget.disabled) {
                return fig.toolbar_button_onmouseover(tooltip);
            }
        };
    }

    // 初始化按钮对象
    fig.buttons = {};
    // 创建按钮组元素
    var buttonGroup = document.createElement('div');
    buttonGroup.classList = 'mpl-button-group';
    // 遍历工具栏项目数组
    for (var toolbar_ind in mpl.toolbar_items) {
        var name = mpl.toolbar_items[toolbar_ind][0];
        var tooltip = mpl.toolbar_items[toolbar_ind][1];
        var image = mpl.toolbar_items[toolbar_ind][2];
        var method_name = mpl.toolbar_items[toolbar_ind][3];

        // 如果按钮名称为空，则创建一个新的按钮组
        if (!name) {
            // 如果当前按钮组已有子节点，则将其添加到工具栏中
            if (buttonGroup.hasChildNodes()) {
                toolbar.appendChild(buttonGroup);
            }
            // 创建新的按钮组元素，并继续下一次循环
            buttonGroup = document.createElement('div');
            buttonGroup.classList = 'mpl-button-group';
            continue;
        }

        // 创建新的按钮元素，并添加到按钮对象中
        var button = (fig.buttons[name] = document.createElement('button'));
        button.classList = 'mpl-widget';
        button.setAttribute('role', 'button');
        button.setAttribute('aria-disabled', 'false');
        button.addEventListener('click', on_click_closure(method_name));
        button.addEventListener('mouseover', on_mouseover_closure(tooltip));

        // 创建图标img元素，并设置相关属性
        var icon_img = document.createElement('img');
        icon_img.src = '_images/' + image + '.png';
        icon_img.srcset = '_images/' + image + '_large.png 2x';
        icon_img.alt = tooltip;
        button.appendChild(icon_img);

        // 将按钮添加到当前按钮组中
        buttonGroup.appendChild(button);
    }

    // 如果当前按钮组有子节点，则将其添加到工具栏中
    if (buttonGroup.hasChildNodes()) {
        toolbar.appendChild(buttonGroup);
    }

    // 创建格式选择器select元素，并添加到工具栏中，并保存到当前对象的属性中
    var fmt_picker = document.createElement('select');
    fmt_picker.classList = 'mpl-widget';
    toolbar.appendChild(fmt_picker);
    this.format_dropdown = fmt_picker;

    // 遍历扩展数组，并创建选项元素，设置选中状态，并添加到格式选择器中
    for (var ind in mpl.extensions) {
        var fmt = mpl.extensions[ind];
        var option = document.createElement('option');
        option.selected = fmt === mpl.default_extension;
        option.innerHTML = fmt;
        fmt_picker.appendChild(option);
    }

    // 创建状态栏span元素，并添加到工具栏中，并保存到当前对象的属性中
    var status_bar = document.createElement('span');
    status_bar.classList = 'mpl-message';
    toolbar.appendChild(status_bar);
    this.message = status_bar;
};

mpl.figure.prototype.request_resize = function (x_pixels, y_pixels) {
    // 请求Matplotlib调整图形大小。Matplotlib将触发客户端的大小调整，并请求图像刷新。
    this.send_message('resize', { width: x_pixels, height: y_pixels });
};
mpl.figure.prototype.send_message = function (type, properties) {
    // 将消息类型和图形 ID 添加到属性对象中
    properties['type'] = type;
    properties['figure_id'] = this.id;
    // 发送经过 JSON 序列化的属性对象到 WebSocket 连接
    this.ws.send(JSON.stringify(properties));
};

mpl.figure.prototype.send_draw_message = function () {
    // 如果当前图形不在等待状态，则设置为等待并发送绘图消息到 WebSocket 连接
    if (!this.waiting) {
        this.waiting = true;
        this.ws.send(JSON.stringify({ type: 'draw', figure_id: this.id }));
    }
};

mpl.figure.prototype.handle_save = function (fig, _msg) {
    // 处理保存操作，获取选中的格式并调用 ondownload 方法下载图形
    var format_dropdown = fig.format_dropdown;
    var format = format_dropdown.options[format_dropdown.selectedIndex].value;
    fig.ondownload(fig, format);
};

mpl.figure.prototype.handle_resize = function (fig, msg) {
    // 处理图形大小调整消息，比较新旧尺寸并进行相应调整
    var size = msg['size'];
    if (size[0] !== fig.canvas.width || size[1] !== fig.canvas.height) {
        fig._resize_canvas(size[0], size[1], msg['forward']);
        fig.send_message('refresh', {});
    }
};

mpl.figure.prototype.handle_rubberband = function (fig, msg) {
    // 处理选框绘制消息，根据消息中的坐标计算选框位置和大小，并在 canvas 上绘制选框
    var x0 = msg['x0'] / fig.ratio;
    var y0 = (fig.canvas.height - msg['y0']) / fig.ratio;
    var x1 = msg['x1'] / fig.ratio;
    var y1 = (fig.canvas.height - msg['y1']) / fig.ratio;
    x0 = Math.floor(x0) + 0.5;
    y0 = Math.floor(y0) + 0.5;
    x1 = Math.floor(x1) + 0.5;
    y1 = Math.floor(y1) + 0.5;
    var min_x = Math.min(x0, x1);
    var min_y = Math.min(y0, y1);
    var width = Math.abs(x1 - x0);
    var height = Math.abs(y1 - y0);

    // 清除之前的选框并绘制新的选框
    fig.rubberband_context.clearRect(
        0,
        0,
        fig.canvas.width / fig.ratio,
        fig.canvas.height / fig.ratio
    );
    fig.rubberband_context.strokeRect(min_x, min_y, width, height);
};

mpl.figure.prototype.handle_figure_label = function (fig, msg) {
    // 更新图形标题（header）
    fig.header.textContent = msg['label'];
};

mpl.figure.prototype.handle_cursor = function (fig, msg) {
    // 更新鼠标指针样式
    fig.canvas_div.style.cursor = msg['cursor'];
};

mpl.figure.prototype.handle_message = function (fig, msg) {
    // 更新消息文本内容
    fig.message.textContent = msg['message'];
};

mpl.figure.prototype.handle_draw = function (fig, _msg) {
    // 请求服务器发送一个新的图形
    fig.send_draw_message();
};

mpl.figure.prototype.handle_image_mode = function (fig, msg) {
    // 处理图像模式切换消息，更新图像模式
    fig.image_mode = msg['mode'];
};

mpl.figure.prototype.handle_history_buttons = function (fig, msg) {
    // 处理历史按钮消息，更新按钮的禁用状态和可访问性
    for (var key in msg) {
        if (!(key in fig.buttons)) {
            continue;
        }
        fig.buttons[key].disabled = !msg[key];
        fig.buttons[key].setAttribute('aria-disabled', !msg[key]);
    }
};

mpl.figure.prototype.handle_navigate_mode = function (fig, msg) {
    // 处理导航模式消息，根据消息中的模式更新对应的按钮状态
    if (msg['mode'] === 'PAN') {
        fig.buttons['Pan'].classList.add('active');
        fig.buttons['Zoom'].classList.remove('active');
    } else if (msg['mode'] === 'ZOOM') {
        fig.buttons['Pan'].classList.remove('active');
        fig.buttons['Zoom'].classList.add('active');
    } else {
        fig.buttons['Pan'].classList.remove('active');
        fig.buttons['Zoom'].classList.remove('active');
    }
};
// 创建一个原型方法，用于在画布更新时发送确认消息
mpl.figure.prototype.updated_canvas_event = function () {
    this.send_message('ack', {});  // 调用 send_message 方法发送空消息体的确认信息
};

// 构造一个 WebSocket 的 onmessage 处理函数的工厂函数
// 在图形对象构造函数中调用
mpl.figure.prototype._make_on_message_function = function (fig) {
    return function socket_on_message(evt) {
        // 如果事件数据是 Blob 类型
        if (evt.data instanceof Blob) {
            var img = evt.data;
            // 如果图片类型不是 'image/png'
            if (img.type !== 'image/png') {
                /* FIXME: 我们在 Chrome 上会遇到 "Resource interpreted as Image but
                 * transferred with MIME type text/plain:" 错误。但如何设置 MIME 类型呢？
                 * 它似乎不是 WebSocket 流的一部分 */
                img.type = 'image/png';  // 将 MIME 类型设置为 'image/png'
            }

            // 释放前一帧的内存
            if (fig.imageObj.src) {
                // 使用 URL 对象的 revokeObjectURL 方法释放图像对象的 URL
                (window.URL || window.webkitURL).revokeObjectURL(
                    fig.imageObj.src
                );
            }

            // 为图像对象设置新的 URL
            fig.imageObj.src = (window.URL || window.webkitURL).createObjectURL(
                img
            );
            // 调用 updated_canvas_event 方法，通知画布更新事件
            fig.updated_canvas_event();
            fig.waiting = false;  // 将等待状态设置为 false
            return;
        } else if (
            typeof evt.data === 'string' &&
            evt.data.slice(0, 21) === 'data:image/png;base64'
        ) {
            // 如果事件数据是字符串，且以 'data:image/png;base64' 开头
            fig.imageObj.src = evt.data;  // 直接设置图像对象的 URL
            fig.updated_canvas_event();  // 调用 updated_canvas_event 方法，通知画布更新事件
            fig.waiting = false;  // 将等待状态设置为 false
            return;
        }

        // 解析 JSON 格式的事件数据
        var msg = JSON.parse(evt.data);
        var msg_type = msg['type'];

        // 调用 "handle_{type}" 回调函数，该函数接受图形对象和 JSON 消息作为参数
        try {
            var callback = fig['handle_' + msg_type];
        } catch (e) {
            // 如果捕获到异常，说明没有对应的消息类型处理函数
            console.log(
                "No handler for the '" + msg_type + "' message type: ",
                msg
            );
            return;
        }

        if (callback) {
            try {
                // 调用回调函数处理消息
                // console.log("Handling '" + msg_type + "' message: ", msg);
                callback(fig, msg);
            } catch (e) {
                // 如果在回调函数中捕获到异常
                console.log(
                    "Exception inside the 'handler_" + msg_type + "' callback:",
                    e,
                    e.stack,
                    msg
                );
            }
        }
    };
};

// 获取事件对象的修饰键列表
function getModifiers(event) {
    var mods = [];
    // 如果事件对象按下了 ctrl 键
    if (event.ctrlKey) {
        mods.push('ctrl');  // 添加 'ctrl' 到修饰键列表
    }
    // 如果事件对象按下了 alt 键
    if (event.altKey) {
        mods.push('alt');  // 添加 'alt' 到修饰键列表
    }
    // 如果事件对象按下了 shift 键
    if (event.shiftKey) {
        mods.push('shift');  // 添加 'shift' 到修饰键列表
    }
    // 如果事件对象按下了 meta 键
    if (event.metaKey) {
        mods.push('meta');  // 添加 'meta' 到修饰键列表
    }
    return mods;  // 返回修饰键列表
}

/*
 * 返回一个仅包含非对象键的对象副本
 * 我们需要这个函数来避免循环引用
 * 参考：https://stackoverflow.com/a/24161582/3208463
 */
function simpleKeys(original) {
    // 使用原始对象的键数组，通过 reduce 函数生成一个新的对象
    return Object.keys(original).reduce(function (obj, key) {
        // 检查原始对象的当前键对应的值是否不是对象
        if (typeof original[key] !== 'object') {
            // 如果值不是对象，则将键值对添加到新对象中
            obj[key] = original[key];
        }
        // 返回累积的对象
        return obj;
    }, {});
}

mpl.figure.prototype.mouse_event = function (event, name) {
    // 如果事件名为 'button_press'，则将焦点设置到画布和画布容器上
    if (name === 'button_press') {
        this.canvas.focus();
        this.canvas_div.focus();
    }

    // 获取画布相对于视口的边界矩形
    var boundingRect = this.canvas.getBoundingClientRect();
    // 计算事件发生位置相对于画布左上角的坐标，并乘以缩放比例
    var x = (event.clientX - boundingRect.left) * this.ratio;
    var y = (event.clientY - boundingRect.top) * this.ratio;

    // 发送事件消息，包含事件的坐标、按钮信息、步骤、修饰键信息和简化的事件对象
    this.send_message(name, {
        x: x,
        y: y,
        button: event.button,
        step: event.step,
        modifiers: getModifiers(event),
        guiEvent: simpleKeys(event),
    });

    // 阻止事件的默认行为
    return false;
};

mpl.figure.prototype._key_event_extra = function (_event, _name) {
    // 处理与按键事件相关的任何额外行为
};

mpl.figure.prototype.key_event = function (event, name) {
    // 防止重复事件
    if (name === 'key_press') {
        if (event.key === this._key) {
            return;
        } else {
            this._key = event.key;
        }
    }
    if (name === 'key_release') {
        this._key = null;
    }

    // 构建键值字符串，包括修饰键（Ctrl、Alt、Shift）和按键值
    var value = '';
    if (event.ctrlKey && event.key !== 'Control') {
        value += 'ctrl+';
    }
    else if (event.altKey && event.key !== 'Alt') {
        value += 'alt+';
    }
    else if (event.shiftKey && event.key !== 'Shift') {
        value += 'shift+';
    }

    value += 'k' + event.key;

    // 调用额外的按键事件处理函数
    this._key_event_extra(event, name);

    // 发送按键事件消息，包含按键值和简化的事件对象
    this.send_message(name, { key: value, guiEvent: simpleKeys(event) });
    // 阻止事件的默认行为
    return false;
};

mpl.figure.prototype.toolbar_button_onclick = function (name) {
    // 如果按钮名为 'download'，则调用保存处理函数，否则发送工具栏按钮消息
    if (name === 'download') {
        this.handle_save(this, null);
    } else {
        this.send_message('toolbar_button', { name: name });
    }
};

mpl.figure.prototype.toolbar_button_onmouseover = function (tooltip) {
    // 设置消息元素的文本内容为工具提示
    this.message.textContent = tooltip;
};

///////////////// REMAINING CONTENT GENERATED BY embed_js.py /////////////////
// prettier-ignore
var _JSXTOOLS_RESIZE_OBSERVER=function(A){ // 定义全局变量 _JSXTOOLS_RESIZE_OBSERVER，接受参数 A
    var t, // 声明变量 t
        i=new WeakMap, // 创建 WeakMap 实例 i
        n=new WeakMap, // 创建 WeakMap 实例 n
        a=new WeakMap, // 创建 WeakMap 实例 a
        r=new WeakMap, // 创建 WeakMap 实例 r
        o=new Set; // 创建 Set 实例 o

    function s(e){ // 定义函数 s，接受参数 e
        if(!(this instanceof s)) // 如果不是通过 'new' 操作符调用，则抛出类型错误
            throw new TypeError("Constructor requires 'new' operator");
        i.set(this,e) // 将当前对象和参数 e 存入 WeakMap i
    }

    function h(){ // 定义函数 h
        throw new TypeError("Function is not a constructor") // 抛出类型错误，提示函数不是构造函数
    }

    function c(e,t,i,n){ // 定义函数 c，接受四个参数 e、t、i、n
        e=0 in arguments?Number(arguments[0]):0, // 如果有第一个参数，则将其转换为数字，否则默认为 0
        t=1 in arguments?Number(arguments[1]):0, // 如果有第二个参数，则将其转换为数字，否则默认为 0
        i=2 in arguments?Number(arguments[2]):0, // 如果有第三个参数，则将其转换为数字，否则默认为 0
        n=3 in arguments?Number(arguments[3]):0, // 如果有第四个参数，则将其转换为数字，否则默认为 0
        this.right=(this.x=this.left=e)+(this.width=i), // 计算右边界，设置 x 和 left，并加上宽度 i
        this.bottom=(this.y=this.top=t)+(this.height=n), // 计算底边界，设置 y 和 top，并加上高度 n
        Object.freeze(this) // 冻结当前对象，使其不可修改
    }

    function d(){ // 定义函数 d
        t=requestAnimationFrame(d); // 请求动画帧，调用函数 d
        var s=new WeakMap, // 创建局部变量 s，作为 WeakMap 实例
            p=new Set; // 创建局部变量 p，作为 Set 实例
        o.forEach((function(t){ // 遍历 Set o 中的每个元素
            r.get(t).forEach((function(i){ // 遍历 t 对应的 WeakMap r 中 t 的值的每个元素
                var r=t instanceof window.SVGElement, // 判断 t 是否为 SVGElement 元素
                    o=a.get(t), // 获取 t 的样式对象
                    d=r?0:parseFloat(o.paddingTop), // 如果是 SVGElement 则为 0，否则解析获取 paddingTop 的值
                    f=r?0:parseFloat(o.paddingRight), // 如果是 SVGElement 则为 0，否则解析获取 paddingRight 的值
                    l=r?0:parseFloat(o.paddingBottom), // 如果是 SVGElement 则为 0，否则解析获取 paddingBottom 的值
                    u=r?0:parseFloat(o.paddingLeft), // 如果是 SVGElement 则为 0，否则解析获取 paddingLeft 的值
                    g=r?0:parseFloat(o.borderTopWidth), // 如果是 SVGElement 则为 0，否则解析获取 borderTopWidth 的值
                    m=r?0:parseFloat(o.borderRightWidth), // 如果是 SVGElement 则为 0，否则解析获取 borderRightWidth 的值
                    w=r?0:parseFloat(o.borderBottomWidth), // 如果是 SVGElement 则为 0，否则解析获取 borderBottomWidth 的值
                    b=u+f, // 计算边框左右宽度和
                    F=d+l, // 计算边框上下高度和
                    v=(r?0:parseFloat(o.borderLeftWidth))+m, // 计算左边框宽度
                    W=g+w, // 计算上边框宽度
                    y=r?0:t.offsetHeight-W-t.clientHeight, // 计算高度不包括边框和内边距部分
                    E=r?0:t.offsetWidth-v-t.clientWidth, // 计算宽度不包括边框和内边距部分
                    R=b+v, // 计算实际宽度
                    z=F+W, // 计算实际高度
                    M=r?t.width:parseFloat(o.width)-R-E, // 计算内容区域宽度
                    O=r?t.height:parseFloat(o.height)-z-y; // 计算内容区域高度
                if(n.has(t)){ // 如果 WeakMap n 中存在 t
                    var k=n.get(t); // 获取 t 对应的值
                    if(k[0]===M&&k[1]===O) // 如果第一个和第二个值与当前值相同
                        return // 直接返回
                }
                n.set(t,[M,O]); // 将 t 和对应的内容区域宽度和高度存入 WeakMap n
                var S=Object.create(h.prototype); // 创建以 h.prototype 为原型的对象 S
                S.target=t, // 设置 S 的目标属性为 t
                S.contentRect=new c(u,d,M,O), // 设置 S 的内容区域属性为 c 的实例
                s.has(i)||(s.set(i,[]),p.add(i)), // 如果 WeakMap s 中不存在 i，则添加 i 到 Set p 中
                s.get(i).push(S) // 将 S 添加到 WeakMap s 中对应的值中
            })) // 结束内部 forEach 函数
        })) // 结束外部 forEach 函数
        p.forEach((function(e){ // 遍历 Set p 中的每个元素
            i.get(e).call(e,s.get(e),e) // 调用 WeakMap i 中 e 对应的值，并传入对应的参数
        })) // 结束 forEach 函数
    } // 结束函数 d

    return s.prototype.observe=function(i){ // 将函数 observe 添加到 s.prototype 中，接受参数 i
        if(i instanceof window.Element){ // 如果 i 是 Element 元素
            r.has(i)||(r.set(i,new Set),o.add(i),a.set(i,window.getComputedStyle(i))); // 如果 WeakMap r 中没有 i，则设置 i 对应的值
            var n=r.get(i); // 获取 i 对应的值
            n.has(this)||n.add(this), // 如果没有当前对象，则添加到对应的 Set 中
            cancelAnimationFrame(t), // 取消动画帧
            t=requestAnimationFrame(d) // 请求动画帧，调用函数 d
        }
    }, // 结束函数 observe

    s.prototype.unobserve=function(i){ // 将函数 unobserve 添加到 s.prototype 中，接受参数 i
        if(i instanceof window.Element&&r.has(i)){ // 如果 i 是 Element 元素且 WeakMap r 中有 i
            var n=r.get(i); // 获取 i 对应的值
            n.has(this)&&(n.delete(this),n.size||(r.delete(i),o.delete(i))), // 如果包含当前对象，则删除当前对象，并检查 Set 是否为空
            n.size||r.delete(i), // 如果 Set 为空，则删除对应的 WeakMap
            o.size||cancelAnimationFrame(t) // 如果 Set o 为空，则取消动画帧
        }
    }, // 结束函数 unobserve

    A.DOMRectReadOnly=c, // 将 c 指定给 A.DOMRectReadOnly
    A.ResizeObserver=s, // 将 s 指定给 A.ResizeObserver
    A.ResizeObserverEntry=h // 将 h 指定给 A.ResizeObserverEntry
}; // eslint-disable-line
```