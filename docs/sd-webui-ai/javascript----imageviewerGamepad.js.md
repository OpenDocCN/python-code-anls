# `stable-diffusion-webui\javascript\imageviewerGamepad.js`

```py
// 初始化游戏手柄数组
let gamepads = [];

// 监听游戏手柄连接事件
window.addEventListener('gamepadconnected', (e) => {
    // 获取游戏手柄索引
    const index = e.gamepad.index;
    let isWaiting = false;
    // 设置定时器，检测游戏手柄状态
    gamepads[index] = setInterval(async() => {
        // 如果未启用游戏手柄功能或正在等待，则返回
        if (!opts.js_modal_lightbox_gamepad || isWaiting) return;
        // 获取当前游戏手柄对象
        const gamepad = navigator.getGamepads()[index];
        // 获取 X 轴数值
        const xValue = gamepad.axes[0];
        // 根据 X 轴数值执行不同操作
        if (xValue <= -0.3) {
            modalPrevImage(e);
            isWaiting = true;
        } else if (xValue >= 0.3) {
            modalNextImage(e);
            isWaiting = true;
        }
        // 如果正在等待，则延迟执行
        if (isWaiting) {
            await sleepUntil(() => {
                const xValue = navigator.getGamepads()[index].axes[0];
                if (xValue < 0.3 && xValue > -0.3) {
                    return true;
                }
            }, opts.js_modal_lightbox_gamepad_repeat);
            isWaiting = false;
        }
    }, 10);
});

// 监听游戏手柄断开连接事件
window.addEventListener('gamepaddisconnected', (e) => {
    // 清除对应游戏手柄的定时器
    clearInterval(gamepads[e.gamepad.index]);
});

/*
主要用于 VR 控制器类型的指针设备。
我使用滚轮事件，因为目前在 Web XR 中没有正确的方法来实现它。
*/
let isScrolling = false;
// 监听滚轮事件
window.addEventListener('wheel', (e) => {
    // 如果未启用游戏手柄功能或正在滚动，则返回
    if (!opts.js_modal_lightbox_gamepad || isScrolling) return;
    isScrolling = true;

    // 根据滚动方向执行不同操作
    if (e.deltaX <= -0.6) {
        modalPrevImage(e);
    } else if (e.deltaX >= 0.6) {
        modalNextImage(e);
    }

    // 设置延迟恢复滚动状态
    setTimeout(() => {
        isScrolling = false;
    }, opts.js_modal_lightbox_gamepad_repeat);
});

// 定义等待直到条件满足的函数
function sleepUntil(f, timeout) {
    return new Promise((resolve) => {
        const timeStart = new Date();
        const wait = setInterval(function() {
            if (f() || new Date() - timeStart > timeout) {
                clearInterval(wait);
                resolve();
            }
        }, 20);
    });
}
```