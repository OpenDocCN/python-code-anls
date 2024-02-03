# `stable-diffusion-webui\javascript\imageMaskFix.js`

```
/**
 * 临时修复 https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/668 的问题
 * 参考 https://github.com/gradio-app/gradio/issues/1721
 */
function imageMaskResize() {
    // 获取所有具有 '.touch-none canvas' 类的元素
    const canvases = gradioApp().querySelectorAll('#img2maskimg .touch-none canvas');
    // 如果没有找到 canvas 元素，则移除 resize 事件监听器并返回
    if (!canvases.length) {
        window.removeEventListener('resize', imageMaskResize);
        return;
    }

    // 获取包裹 canvas 的元素
    const wrapper = canvases[0].closest('.touch-none');
    // 获取前一个兄弟元素作为预览图片
    const previewImage = wrapper.previousElementSibling;

    // 如果预览图片未加载完成，则添加 load 事件监听器并返回
    if (!previewImage.complete) {
        previewImage.addEventListener('load', imageMaskResize);
        return;
    }

    // 获取预览图片的宽度和高度
    const w = previewImage.width;
    const h = previewImage.height;
    const nw = previewImage.naturalWidth;
    const nh = previewImage.naturalHeight;
    const portrait = nh > nw;

    // 计算调整后的宽度和高度
    const wW = Math.min(w, portrait ? h / nh * nw : w / nw * nw);
    const wH = Math.min(h, portrait ? h / nh * nh : w / nw * nh);

    // 设置包裹元素的样式
    wrapper.style.width = `${wW}px`;
    wrapper.style.height = `${wH}px`;
    wrapper.style.left = `0px`;
    wrapper.style.top = `0px`;

    // 设置每个 canvas 元素的样式
    canvases.forEach(c => {
        c.style.width = c.style.height = '';
        c.style.maxWidth = '100%';
        c.style.maxHeight = '100%';
        c.style.objectFit = 'contain';
    });
}

// 在 UI 更新后调用 imageMaskResize 函数
onAfterUiUpdate(imageMaskResize);
// 添加 resize 事件监听器，以便在窗口大小改变时调整图片遮罩大小
window.addEventListener('resize', imageMaskResize);
```