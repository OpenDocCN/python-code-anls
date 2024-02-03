# `stable-diffusion-webui\script.js`

```
/**
 * 定义一个函数用于获取 gradio-app 元素，如果不存在则返回 document
 */
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    // 如果找到了 gradio-app 元素，则为其添加 getElementById 方法，用于获取 document 中的元素
    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    // 返回 gradio-app 元素的 shadowRoot 或者元素本身
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * 获取当前选定的顶级 UI 选项卡按钮（例如，显示“Extras”按钮）
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

/**
 * 获取当前可见的第一个顶级 UI 选项卡内容（例如，承载“txt2img” UI 的 div）
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

// 定义一系列回调函数数组
var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * 注册在每次 UI 更新时调用的回调函数
 * 回调函数接收一个 MutationRecords 数组作为参数
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * 注册在 UI 更新后不久调用的回调函数
 * 回调函数不接收任何参数
 *
 * 如果不需要访问 MutationRecords，建议使用此函数，因为您的函数不会被调用得太频繁
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * 注册在 UI 加载时调用的回调函数
 * 回调函数不接收任何参数
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * 注册在 UI 选项卡更改时调用的回调函数
 * 回调函数不接收任何参数
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}
/**
 * 注册回调函数，当选项更改时调用。
 * 回调函数不接收任何参数。
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

/**
 * 执行队列中的回调函数，并传入参数。
 * @param queue
 * @param arg
 */
function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * 安排执行在 onAfterUiUpdate 注册的回调函数。
 * 回调函数将在短时间后执行，除非在此期间再次调用此函数。
 * 换句话说，即使观察到多个突变，回调函数也只执行一次。
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#txt2img_prompt')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

/**
 * 添加 ctrl+enter 作为启动生成的快捷键。
 */
document.addEventListener('keydown', function(e) {
    const isEnter = e.key === 'Enter' || e.keyCode === 13;
    const isModifierKey = e.metaKey || e.ctrlKey || e.altKey;

    const interruptButton = get_uiCurrentTabContent().querySelector('button[id$=_interrupt]');
}
    // 获取当前标签页内容中最后一个 id 以 _generate 结尾的按钮元素
    const generateButton = get_uiCurrentTabContent().querySelector('button[id$=_generate');

    // 如果按下回车键并且同时按下修饰键
    if (isEnter && isModifierKey) {
        // 如果中断按钮显示为块级元素
        if (interruptButton.style.display === 'block') {
            // 点击中断按钮
            interruptButton.click();
            // 定义回调函数，用于监听中断按钮的样式变化
            const callback = (mutationList) => {
                for (const mutation of mutationList) {
                    // 如果是属性变化且属性名为 style
                    if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                        // 如果中断按钮显示为隐藏状态
                        if (interruptButton.style.display === 'none') {
                            // 点击生成按钮
                            generateButton.click();
                            // 停止监听中断按钮的变化
                            observer.disconnect();
                        }
                    }
                }
            };
            // 创建 MutationObserver 实例，监听中断按钮的属性变化
            const observer = new MutationObserver(callback);
            observer.observe(interruptButton, {attributes: true});
        } else {
            // 如果中断按钮不显示为块级元素，则直接点击生成按钮
            generateButton.click();
        }
        // 阻止默认事件
        e.preventDefault();
    }
// 检查一个 UI 元素是否不在另一个隐藏元素或选项卡内容中
function uiElementIsVisible(el) {
    // 如果元素是文档本身，则可见
    if (el === document) {
        return true;
    }

    // 获取元素的计算样式
    const computedStyle = getComputedStyle(el);
    // 检查元素是否显示
    const isVisible = computedStyle.display !== 'none';

    // 如果不可见，则返回 false
    if (!isVisible) return false;
    // 递归检查元素的父元素是否可见
    return uiElementIsVisible(el.parentNode);
}

// 检查一个 UI 元素是否在视野内
function uiElementInSight(el) {
    // 获取元素相对于视口的位置信息
    const clRect = el.getBoundingClientRect();
    // 获取窗口的高度
    const windowHeight = window.innerHeight;
    // 检查元素是否在屏幕上可见
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}
```