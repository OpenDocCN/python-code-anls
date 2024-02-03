# `stable-diffusion-webui\extensions-builtin\mobile\javascript\mobile.js`

```
// 定义一个变量，用于标识是否为移动设备
var isSetupForMobile = false;

// 检查当前设备是否为移动设备
function isMobile() {
    // 遍历指定的标签数组
    for (var tab of ["txt2img", "img2img"]) {
        // 获取指定标签的元素
        var imageTab = gradioApp().getElementById(tab + '_results');
        // 检查元素是否存在且在可见区域且在左侧
        if (imageTab && imageTab.offsetParent && imageTab.offsetLeft == 0) {
            return true;
        }
    }

    return false;
}

// 监听窗口大小变化事件
function reportWindowSize() {
    // 如果是紧凑提示布局，则不适用
    if (gradioApp().querySelector('.toprow-compact-tools')) return;

    // 检查当前设备是否为移动设备
    var currentlyMobile = isMobile();
    // 如果当前设备类型与之前设置的相同，则不做任何操作
    if (currentlyMobile == isSetupForMobile) return;
    // 更新设备类型设置
    isSetupForMobile = currentlyMobile;

    // 遍历指定的标签数组
    for (var tab of ["txt2img", "img2img"]) {
        // 获取生成按钮元素
        var button = gradioApp().getElementById(tab + '_generate_box');
        // 获取目标元素
        var target = gradioApp().getElementById(currentlyMobile ? tab + '_results' : tab + '_actions_column');
        // 将生成按钮插入到目标元素的第一个子元素之前
        target.insertBefore(button, target.firstElementChild);

        // 切换指定标签的结果元素的类，根据当前设备类型添加或移除'mobile'类
        gradioApp().getElementById(tab + '_results').classList.toggle('mobile', currentlyMobile);
    }
}

// 监听窗口大小变化事件
window.addEventListener("resize", reportWindowSize);

// 当 UI 加载完成时执行
onUiLoaded(function() {
    // 执行窗口大小变化处理函数
    reportWindowSize();
});
```