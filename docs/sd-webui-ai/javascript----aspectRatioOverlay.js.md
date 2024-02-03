# `stable-diffusion-webui\javascript\aspectRatioOverlay.js`

```py
// 初始化当前宽度和高度为null
let currentWidth = null;
let currentHeight = null;
// 初始化一个定时器，用于延迟执行空函数
let arFrameTimeout = setTimeout(function() {}, 0);

// 定义一个函数，用于处理尺寸改变事件
function dimensionChange(e, is_width, is_height) {

    // 如果是宽度改变事件，更新当前宽度值
    if (is_width) {
        currentWidth = e.target.value * 1.0;
    }
    // 如果是高度改变事件，更新当前高度值
    if (is_height) {
        currentHeight = e.target.value * 1.0;
    }

    // 检查当前是否在img2img选项卡下
    var inImg2img = gradioApp().querySelector("#tab_img2img").style.display == "block";

    // 如果不在img2img选项卡下，直接返回
    if (!inImg2img) {
        return;
    }

    // 初始化目标元素为null
    var targetElement = null;

    // 获取当前选项卡的索引
    var tabIndex = get_tab_index('mode_img2img');
    // 根据选项卡索引选择目标元素
    if (tabIndex == 0) { // img2img
        targetElement = gradioApp().querySelector('#img2img_image div[data-testid=image] img');
    } else if (tabIndex == 1) { //Sketch
        targetElement = gradioApp().querySelector('#img2img_sketch div[data-testid=image] img');
    } else if (tabIndex == 2) { // Inpaint
        targetElement = gradioApp().querySelector('#img2maskimg div[data-testid=image] img');
    } else if (tabIndex == 3) { // Inpaint sketch
        targetElement = gradioApp().querySelector('#inpaint_sketch div[data-testid=image] img');
    }
    // 检查目标元素是否存在
    if (targetElement) {

        // 获取用于显示 AR 预览的元素
        var arPreviewRect = gradioApp().querySelector('#imageARPreview');
        // 如果 AR 预览元素不存在，则创建一个新的 div 元素
        if (!arPreviewRect) {
            arPreviewRect = document.createElement('div');
            arPreviewRect.id = "imageARPreview";
            gradioApp().appendChild(arPreviewRect);
        }

        // 获取目标元素相对于视口的位置信息
        var viewportOffset = targetElement.getBoundingClientRect();

        // 计算视口缩放比例
        var viewportscale = Math.min(targetElement.clientWidth / targetElement.naturalWidth, targetElement.clientHeight / targetElement.naturalHeight);

        // 计算缩放后的宽度和高度
        var scaledx = targetElement.naturalWidth * viewportscale;
        var scaledy = targetElement.naturalHeight * viewportscale;

        // 计算目标元素在视口中心的位置
        var cleintRectTop = (viewportOffset.top + window.scrollY);
        var cleintRectLeft = (viewportOffset.left + window.scrollX);
        var cleintRectCentreY = cleintRectTop + (targetElement.clientHeight / 2);
        var cleintRectCentreX = cleintRectLeft + (targetElement.clientWidth / 2);

        // 计算 AR 预览的缩放比例
        var arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
        var arscaledx = currentWidth * arscale;
        var arscaledy = currentHeight * arscale;

        // 计算 AR 预览框的位置和大小
        var arRectTop = cleintRectCentreY - (arscaledy / 2);
        var arRectLeft = cleintRectCentreX - (arscaledx / 2);
        var arRectWidth = arscaledx;
        var arRectHeight = arscaledy;

        // 设置 AR 预览框的位置和大小
        arPreviewRect.style.top = arRectTop + 'px';
        arPreviewRect.style.left = arRectLeft + 'px';
        arPreviewRect.style.width = arRectWidth + 'px';
        arPreviewRect.style.height = arRectHeight + 'px';

        // 清除之前的定时器，并在 2000 毫秒后隐藏 AR 预览框
        clearTimeout(arFrameTimeout);
        arFrameTimeout = setTimeout(function() {
            arPreviewRect.style.display = 'none';
        }, 2000);

        // 显示 AR 预览框
        arPreviewRect.style.display = 'block';

    }
// 在 UI 更新后执行的回调函数
onAfterUiUpdate(function() {
    // 获取 AR 预览区域的元素
    var arPreviewRect = gradioApp().querySelector('#imageARPreview');
    // 如果 AR 预览区域存在
    if (arPreviewRect) {
        // 隐藏 AR 预览区域
        arPreviewRect.style.display = 'none';
    }
    // 获取图像转换选项卡的元素
    var tabImg2img = gradioApp().querySelector("#tab_img2img");
    // 如果图像转换选项卡存在
    if (tabImg2img) {
        // 检查图像转换选项卡是否显示
        var inImg2img = tabImg2img.style.display == "block";
        // 如果图像转换选项卡显示
        if (inImg2img) {
            // 获取所有输入框元素
            let inputs = gradioApp().querySelectorAll('input');
            // 遍历所有输入框元素
            inputs.forEach(function(e) {
                // 检查当前输入框是否为宽度输入框
                var is_width = e.parentElement.id == "img2img_width";
                // 检查当前输入框是否为高度输入框
                var is_height = e.parentElement.id == "img2img_height";

                // 如果当前输入框为宽度或高度输入框，并且未添加滚动监听类
                if ((is_width || is_height) && !e.classList.contains('scrollwatch')) {
                    // 添加输入事件监听器，触发维度变化函数
                    e.addEventListener('input', function(e) {
                        dimensionChange(e, is_width, is_height);
                    });
                    // 添加滚动监听类
                    e.classList.add('scrollwatch');
                }
                // 如果当前输入框为宽度输入框
                if (is_width) {
                    // 更新当前宽度值
                    currentWidth = e.value * 1.0;
                }
                // 如果当前输入框为高度输入框
                if (is_height) {
                    // 更新当前高度值
                    currentHeight = e.value * 1.0;
                }
            });
        }
    }
});
```