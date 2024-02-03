# `stable-diffusion-webui\javascript\resizeHandle.js`

```
(function() {
    // 定义全局常量，最小宽度为 320
    const GRADIO_MIN_WIDTH = 320;
    // 定义网格模板列布局
    const GRID_TEMPLATE_COLUMNS = '1fr 16px 1fr';
    // 定义常量 PAD 为 16
    const PAD = 16;
    // 定义防抖时间为 100 毫秒
    const DEBOUNCE_TIME = 100;

    // 定义对象 R，包含跟踪状态、父元素、父元素宽度、左列元素、左列元素初始宽度、鼠标位置
    const R = {
        tracking: false,
        parent: null,
        parentWidth: null,
        leftCol: null,
        leftColStartWidth: null,
        screenX: null,
    };

    // 定义定时器变量
    let resizeTimer;
    // 定义父元素数组
    let parents = [];

    // 设置左列元素的网格模板列布局
    function setLeftColGridTemplate(el, width) {
        el.style.gridTemplateColumns = `${width}px 16px 1fr`;
    }

    // 显示或隐藏调整大小的手柄
    function displayResizeHandle(parent) {
        if (window.innerWidth < GRADIO_MIN_WIDTH * 2 + PAD * 4) {
            parent.style.display = 'flex';
            if (R.handle != null) {
                R.handle.style.opacity = '0';
            }
            return false;
        } else {
            parent.style.display = 'grid';
            if (R.handle != null) {
                R.handle.style.opacity = '100';
            }
            return true;
        }
    }

    // 调整大小后的处理
    function afterResize(parent) {
        // 如果显示调整大小的手柄，并且父元素的网格模板列布局不等于 GRID_TEMPLATE_COLUMNS
        if (displayResizeHandle(parent) && parent.style.gridTemplateColumns != GRID_TEMPLATE_COLUMNS) {
            // 获取旧的父元素宽度、新的父元素宽度、左列元素宽度
            const oldParentWidth = R.parentWidth;
            const newParentWidth = parent.offsetWidth;
            const widthL = parseInt(parent.style.gridTemplateColumns.split(' ')[0]);

            // 计算宽度比例
            const ratio = newParentWidth / oldParentWidth;

            // 计算新的左列元素宽度
            const newWidthL = Math.max(Math.floor(ratio * widthL), GRADIO_MIN_WIDTH);
            setLeftColGridTemplate(parent, newWidthL);

            // 更新父元素宽度
            R.parentWidth = newParentWidth;
        }
    }
    // 设置布局
    function setup(parent) {
        // 获取父元素的第一个子元素和最后一个子元素
        const leftCol = parent.firstElementChild;
        const rightCol = parent.lastElementChild;

        // 将父元素添加到父元素数组中
        parents.push(parent);

        // 设置父元素的显示方式为网格布局
        parent.style.display = 'grid';
        parent.style.gap = '0';
        parent.style.gridTemplateColumns = GRID_TEMPLATE_COLUMNS;

        // 创建一个调整大小的手柄元素，并插入到右侧列之前
        const resizeHandle = document.createElement('div');
        resizeHandle.classList.add('resize-handle');
        parent.insertBefore(resizeHandle, rightCol);

        // 添加鼠标按下事件监听器
        resizeHandle.addEventListener('mousedown', (evt) => {
            // 如果不是左键点击，则返回
            if (evt.button !== 0) return;

            evt.preventDefault();
            evt.stopPropagation();

            // 添加 'resizing' 类到 body 元素
            document.body.classList.add('resizing');

            // 设置跟踪状态和相关属性
            R.tracking = true;
            R.parent = parent;
            R.parentWidth = parent.offsetWidth;
            R.handle = resizeHandle;
            R.leftCol = leftCol;
            R.leftColStartWidth = leftCol.offsetWidth;
            R.screenX = evt.screenX;
        });

        // 添加双击事件监听器
        resizeHandle.addEventListener('dblclick', (evt) => {
            evt.preventDefault();
            evt.stopPropagation();

            // 恢复父元素的网格布局
            parent.style.gridTemplateColumns = GRID_TEMPLATE_COLUMNS;
        });

        // 调整大小后的处理
        afterResize(parent);
    }

    // 添加鼠标移动事件监听器
    window.addEventListener('mousemove', (evt) => {
        // 如果不是左键点击，则返回
        if (evt.button !== 0) return;

        // 如果正在跟踪调整大小
        if (R.tracking) {
            evt.preventDefault();
            evt.stopPropagation();

            // 计算鼠标移动距离
            const delta = R.screenX - evt.screenX;
            // 计算左侧列的宽度
            const leftColWidth = Math.max(Math.min(R.leftColStartWidth - delta, R.parent.offsetWidth - GRADIO_MIN_WIDTH - PAD), GRADIO_MIN_WIDTH);
            // 设置左侧列的网格布局
            setLeftColGridTemplate(R.parent, leftColWidth);
        }
    });

    // 添加鼠标松开事件监听器
    window.addEventListener('mouseup', (evt) => {
        // 如果不是左键点击，则返回
        if (evt.button !== 0) return;

        // 如果正在跟踪调整大小
        if (R.tracking) {
            evt.preventDefault();
            evt.stopPropagation();

            // 结束跟踪调整大小
            R.tracking = false;

            // 移除 'resizing' 类从 body 元素
            document.body.classList.remove('resizing');
        }
    });
    # 当窗口大小改变时触发事件监听器
    window.addEventListener('resize', () => {
        # 清除之前设置的延迟执行函数
        clearTimeout(resizeTimer);

        # 设置新的延迟执行函数
        resizeTimer = setTimeout(function() {
            # 遍历父元素数组，对每个父元素执行afterResize函数
            for (const parent of parents) {
                afterResize(parent);
            }
        }, DEBOUNCE_TIME);
    });

    # 将setup函数赋值给setupResizeHandle变量
    setupResizeHandle = setup;
// 立即执行函数，用于封装代码，避免全局变量污染
})();

// 当 UI 加载完成时执行的回调函数
onUiLoaded(function() {
    // 遍历所有包含类名为 'resize-handle-row' 的元素
    for (var elem of gradioApp().querySelectorAll('.resize-handle-row')) {
        // 如果当前元素下没有类名为 'resize-handle' 的元素
        if (!elem.querySelector('.resize-handle')) {
            // 调用设置调整大小句柄的函数
            setupResizeHandle(elem);
        }
    }
});
```