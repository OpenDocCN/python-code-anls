# `stable-diffusion-webui\javascript\generationParams.js`

```py
// 定义变量 txt2img_gallery, img2img_gallery, modal，初始值为 undefined
let txt2img_gallery, img2img_gallery, modal = undefined;
// 在 UI 更新后执行的回调函数
onAfterUiUpdate(function() {
    // 如果 txt2img_gallery 不存在，则将其赋值为 txt2img 图库的监听器
    if (!txt2img_gallery) {
        txt2img_gallery = attachGalleryListeners("txt2img");
    }
    // 如果 img2img_gallery 不存在，则将其赋值为 img2img 图库的监听器
    if (!img2img_gallery) {
        img2img_gallery = attachGalleryListeners("img2img");
    }
    // 如果 modal 不存在，则将其赋值为 lightboxModal 元素，并添加属性变化的观察器
    if (!modal) {
        modal = gradioApp().getElementById('lightboxModal');
        modalObserver.observe(modal, {attributes: true, attributeFilter: ['style']});
    }
});

// 创建 MutationObserver 实例 modalObserver，用于监听 modal 元素的属性变化
let modalObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutationRecord) {
        // 获取当前选中的标签页的文本内容
        let selectedTab = gradioApp().querySelector('#tabs div button.selected')?.innerText;
        // 如果 modal 元素的 display 样式为 'none'，且当前选中的标签页为 'txt2img' 或 'img2img'
        if (mutationRecord.target.style.display === 'none' && (selectedTab === 'txt2img' || selectedTab === 'img2img')) {
            // 触发选中标签页的生成信息按钮的点击事件
            gradioApp().getElementById(selectedTab + "_generation_info_button")?.click();
        }
    });
});

// 定义函数 attachGalleryListeners，用于为指定标签页的图库添加事件监听器
function attachGalleryListeners(tab_name) {
    // 获取指定标签页的图库元素
    var gallery = gradioApp().querySelector('#' + tab_name + '_gallery');
    // 添加点击事件监听器，点击时触发生成信息按钮的点击事件
    gallery?.addEventListener('click', () => gradioApp().getElementById(tab_name + "_generation_info_button").click());
    // 添加键盘按键事件监听器，当按下左右箭头键时触发生成信息按钮的点击事件
    gallery?.addEventListener('keydown', (e) => {
        if (e.keyCode == 37 || e.keyCode == 39) { // left or right arrow
            gradioApp().getElementById(tab_name + "_generation_info_button").click();
        }
    });
    // 返回图库元素
    return gallery;
}
```