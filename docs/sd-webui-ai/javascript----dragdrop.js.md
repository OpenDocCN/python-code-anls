# `stable-diffusion-webui\javascript\dragdrop.js`

```py
// 检查文件列表是否包含有效的图片文件，只允许包含一个图片文件且类型为 'image/png', 'image/gif', 'image/jpeg'
function isValidImageList(files) {
    return files && files?.length === 1 && ['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type);
}

// 替换图片元素中的图片，根据传入的文件列表
function dropReplaceImage(imgWrap, files) {
    // 如果文件列表不包含有效的图片文件，直接返回
    if (!isValidImageList(files)) {
        return;
    }

    // 保存第一个文件
    const tmpFile = files[0];

    // 触发点击事件，模拟用户点击上传按钮
    imgWrap.querySelector('.modify-upload button + button, .touch-none + div button + button')?.click();
    // 回调函数，用于处理文件输入
    const callback = () => {
        const fileInput = imgWrap.querySelector('input[type="file"]');
        if (fileInput) {
            // 如果文件列表为空，创建新的 DataTransfer 对象并添加文件，然后设置文件输入的文件列表
            if (files.length === 0) {
                files = new DataTransfer();
                files.items.add(tmpFile);
                fileInput.files = files.files;
            } else {
                fileInput.files = files;
            }
            // 触发文件输入的 change 事件
            fileInput.dispatchEvent(new Event('change'));
        }
    };

    // 如果图片元素位于特定的 PNG Info 标签页，等待 fetch 请求完成后执行回调
    if (imgWrap.closest('#pnginfo_image')) {
        // 临时保存原始的 fetch 函数
        const oldFetch = window.fetch;
        // 重写 fetch 函数，处理特定的 fetch 请求
        window.fetch = async(input, options) => {
            const response = await oldFetch(input, options);
            if ('api/predict/' === input) {
                const content = await response.text();
                window.fetch = oldFetch;
                // 在下一个动画帧执行回调函数
                window.requestAnimationFrame(() => callback());
                return new Response(content, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: response.headers
                });
            }
            return response;
        };
    } else {
        // 在下一个动画帧执行回调函数
        window.requestAnimationFrame(() => callback());
    }
}

// 检查事件是否包含文件
function eventHasFiles(e) {
    if (!e.dataTransfer || !e.dataTransfer.files) return false;
    if (e.dataTransfer.files.length > 0) return true;
    if (e.dataTransfer.items.length > 0 && e.dataTransfer.items[0].kind == "file") return true;
}
    # 返回布尔值 false
    return false;
}

// 检查目标元素是否为提示框
function dragDropTargetIsPrompt(target) {
    // 如果目标元素有 placeholder 属性且包含 "Prompt" 字符串，则返回 true
    if (target?.placeholder && target?.placeholder.indexOf("Prompt") >= 0) return true;
    // 如果目标元素的父节点的父节点的类名包含 "prompt" 字符串，则返回 true
    if (target?.parentNode?.parentNode?.className?.indexOf("prompt") > 0) return true;
    // 否则返回 false
    return false;
}

// 监听文档的 dragover 事件
window.document.addEventListener('dragover', e => {
    // 获取事件的目标元素
    const target = e.composedPath()[0];
    // 如果事件不包含文件，则返回
    if (!eventHasFiles(e)) return;

    // 获取最近的包含属性 data-testid="image" 的父元素
    var targetImage = target.closest('[data-testid="image"]');
    // 如果目标不是提示框且不是图片元素，则返回
    if (!dragDropTargetIsPrompt(target) && !targetImage) return;

    // 阻止事件冒泡
    e.stopPropagation();
    // 阻止默认行为
    e.preventDefault();
    // 设置拖放效果为复制
    e.dataTransfer.dropEffect = 'copy';
});

// 监听文档的 drop 事件
window.document.addEventListener('drop', e => {
    // 获取事件的目标元素
    const target = e.composedPath()[0];
    // 如果事件不包含文件，则返回
    if (!eventHasFiles(e)) return;

    // 如果目标是提示框
    if (dragDropTargetIsPrompt(target)) {
        // 阻止事件冒泡
        e.stopPropagation();
        // 阻止默认行为
        e.preventDefault();

        // 根据当前选项卡索引确定目标元素
        let prompt_target = get_tab_index('tabs') == 1 ? "img2img_prompt_image" : "txt2img_prompt_image";

        // 获取目标元素
        const imgParent = gradioApp().getElementById(prompt_target);
        const files = e.dataTransfer.files;
        const fileInput = imgParent.querySelector('input[type="file"]');
        if (fileInput) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    }

    // 获取最近的包含属性 data-testid="image" 的父元素
    var targetImage = target.closest('[data-testid="image"]');
    // 如果目标是图片元素
    if (targetImage) {
        // 阻止事件冒泡
        e.stopPropagation();
        // 阻止默认行为
        e.preventDefault();
        // 获取拖放的文件并替换图片
        const files = e.dataTransfer.files;
        dropReplaceImage(targetImage, files);
        return;
    }
});

// 监听窗口的 paste 事件
window.addEventListener('paste', e => {
    // 获取剪贴板中的文件
    const files = e.clipboardData.files;
    // 如果文件不是有效的图片列表，则返回
    if (!isValidImageList(files)) {
        return;
    }

    // 获取所有可见的图片字段元素，并按可见性排序
    const visibleImageFields = [...gradioApp().querySelectorAll('[data-testid="image"]')]
        .filter(el => uiElementIsVisible(el))
        .sort((a, b) => uiElementInSight(b) - uiElementInSight(a));

    // 如果没有可见的图片字段元素，则返回
    if (!visibleImageFields.length) {
        return;
    }
    // 找到第一个可见的图片字段中没有图片的字段
    const firstFreeImageField = visibleImageFields
        .filter(el => !el.querySelector('img'))?.[0];

    // 调用 dropReplaceImage 函数，传入第一个可见的空图片字段或者最后一个可见的图片字段以及文件列表
    dropReplaceImage(
        firstFreeImageField ?
            firstFreeImageField :
            visibleImageFields[visibleImageFields.length - 1]
        , files
    );
});
```