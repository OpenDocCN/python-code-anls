# `stable-diffusion-webui\javascript\imageviewer.js`

```py
// 关闭模态框，当左键点击图库预览时显示
function closeModal() {
    // 设置模态框的显示样式为隐藏
    gradioApp().getElementById("lightboxModal").style.display = "none";
}

// 显示模态框，当事件发生时
function showModal(event) {
    // 获取事件源
    const source = event.target || event.srcElement;
    // 获取模态框中的图片元素
    const modalImage = gradioApp().getElementById("modalImage");
    // 获取模态框元素
    const lb = gradioApp().getElementById("lightboxModal");
    // 设置模态框中的图片源为事件源的源
    modalImage.src = source.src;
    // 如果模态框中的图片样式为隐藏
    if (modalImage.style.display === 'none') {
        // 设置模态框的背景图片为事件源的源
        lb.style.setProperty('background-image', 'url(' + source.src + ')');
    }
    // 显示模态框
    lb.style.display = "flex";
    // 设置焦点到模态框
    lb.focus();

    // 获取文本到图片和图片到图片标签元素
    const tabTxt2Img = gradioApp().getElementById("tab_txt2img");
    const tabImg2Img = gradioApp().getElementById("tab_img2img");
    // 在文本到图片或图片到图片标签中显示保存按钮
    if (tabTxt2Img.style.display != "none" || tabImg2Img.style.display != "none") {
        gradioApp().getElementById("modal_save").style.display = "inline";
    } else {
        // 否则隐藏保存按钮
        gradioApp().getElementById("modal_save").style.display = "none";
    }
    // 阻止事件冒泡
    event.stopPropagation();
}

// 计算 n 对 m 取模的负数
function negmod(n, m) {
    return ((n % m) + m) % m;
}

// 当背景改变时更新
function updateOnBackgroundChange() {
    // 获取模态框中的图片元素
    const modalImage = gradioApp().getElementById("modalImage");
}
    // 检查 modalImage 是否存在且在文档流中
    if (modalImage && modalImage.offsetParent) {
        // 获取当前选中的按钮
        let currentButton = selected_gallery_button();
        // 获取所有 livePreview 下的 img 元素
        let preview = gradioApp().querySelectorAll('.livePreview > img');
        // 如果启用了在模态框灯箱中实时预览并且存在预览图像，则显示预览图像
        if (opts.js_live_preview_in_modal_lightbox && preview.length > 0) {
            // 显示预览图像
            modalImage.src = preview[preview.length - 1].src;
        } else if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) {
            // 如果当前按钮存在子元素且 modalImage 的 src 与当前按钮的第一个子元素的 src 不相同
            // 将 modalImage 的 src 设置为当前按钮的第一个子元素的 src
            modalImage.src = currentButton.children[0].src;
            // 如果 modalImage 的 display 属性为 'none'
            if (modalImage.style.display === 'none') {
                // 获取灯箱模态框元素
                const modal = gradioApp().getElementById("lightboxModal");
                // 设置灯箱模态框的背景图像为 modalImage 的 src
                modal.style.setProperty('background-image', `url(${modalImage.src})`);
            }
        }
    }
// 定义一个函数，用于切换模态框中的图片
function modalImageSwitch(offset) {
    // 获取所有的图片按钮
    var galleryButtons = all_gallery_buttons();

    // 如果图片按钮数量大于1
    if (galleryButtons.length > 1) {
        // 获取当前选中的按钮
        var currentButton = selected_gallery_button();

        // 初始化结果为-1
        var result = -1;
        // 遍历图片按钮数组
        galleryButtons.forEach(function(v, i) {
            // 如果当前按钮等于选中的按钮
            if (v == currentButton) {
                // 更新结果为当前索引
                result = i;
            }
        });

        // 如果结果不为-1
        if (result != -1) {
            // 计算下一个按钮的索引
            var nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
            // 点击下一个按钮
            nextButton.click();
            // 获取模态框中的图片元素
            const modalImage = gradioApp().getElementById("modalImage");
            // 获取模态框元素
            const modal = gradioApp().getElementById("lightboxModal");
            // 设置模态框图片的源
            modalImage.src = nextButton.children[0].src;
            // 如果模态框图片不可见
            if (modalImage.style.display === 'none') {
                // 设置模态框背景图片
                modal.style.setProperty('background-image', `url(${modalImage.src})`);
            }
            // 延迟10毫秒后聚焦模态框
            setTimeout(function() {
                modal.focus();
            }, 10);
        }
    }
}

// 定义一个函数，用于保存图片
function saveImage() {
    // 获取文本转图片标签
    const tabTxt2Img = gradioApp().getElementById("tab_txt2img");
    // 获取图片转图片标签
    const tabImg2Img = gradioApp().getElementById("tab_img2img");
    // 保存文本转图片按钮ID
    const saveTxt2Img = "save_txt2img";
    // 保存图片转图片按钮ID
    const saveImg2Img = "save_img2img";
    // 如果文本转图片标签不可见
    if (tabTxt2Img.style.display != "none") {
        // 点击保存文本转图片按钮
        gradioApp().getElementById(saveTxt2Img).click();
    } else if (tabImg2Img.style.display != "none") {
        // 如果图片转图片标签不可见，点击保存图片转图片按钮
        gradioApp().getElementById(saveImg2Img).click();
    } else {
        // 否则，输出错误信息
        console.error("missing implementation for saving modal of this type");
    }
}

// 定义一个函数，用于模态框保存图片
function modalSaveImage(event) {
    // 保存图片
    saveImage();
    // 阻止事件冒泡
    event.stopPropagation();
}

// 定义一个函数，用于模态框切换到下一张图片
function modalNextImage(event) {
    // 切换到下一张图片
    modalImageSwitch(1);
    // 阻止事件冒泡
    event.stopPropagation();
}

// 定义一个函数，用于模态框切换到上一张图片
function modalPrevImage(event) {
    // 切换到上一张图片
    modalImageSwitch(-1);
    // 阻止事件冒泡
    event.stopPropagation();
}

// 定义一个函数，用于处理模态框的按键事件
function modalKeyHandler(event) {
    // 根据按键执行相应操作
    switch (event.key) {
        case "s":
            // 如果按下"s"键，保存图片
            saveImage();
            break;
        case "ArrowLeft":
            // 如果按下左箭头键，切换到上一张图片
            modalPrevImage(event);
            break;
    # 如果按下的是右箭头键，则调用 modalNextImage 函数
    case "ArrowRight":
        modalNextImage(event);
        break;
    # 如果按下的是 Escape 键，则调用 closeModal 函数
    case "Escape":
        closeModal();
        break;
    }
// 设置图片以用于灯箱效果
function setupImageForLightbox(e) {
    // 如果已经修改过，则直接返回
    if (e.dataset.modded) {
        return;
    }

    // 标记为已修改
    e.dataset.modded = true;
    // 设置鼠标样式为手型
    e.style.cursor = 'pointer';
    // 禁止用户选择图片内容
    e.style.userSelect = 'none';

    // 检测是否为 Firefox 浏览器
    var isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;

    // 根据浏览器类型设置事件类型
    var event = isFirefox ? 'mousedown' : 'click';

    // 添加事件监听器
    e.addEventListener(event, function(evt) {
        // 如果是中键点击，则打开图片链接
        if (evt.button == 1) {
            open(evt.target.src);
            evt.preventDefault();
            return;
        }
        // 如果未启用 js_modal_lightbox 或者不是左键点击，则返回
        if (!opts.js_modal_lightbox || evt.button != 0) return;

        // 设置灯箱图片的缩放状态
        modalZoomSet(gradioApp().getElementById('modalImage'), opts.js_modal_lightbox_initially_zoomed);
        evt.preventDefault();
        // 显示灯箱
        showModal(evt);
    }, true);

}

// 设置灯箱图片的缩放状态
function modalZoomSet(modalImage, enable) {
    if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable);
}

// 切换灯箱图片的缩放状态
function modalZoomToggle(event) {
    var modalImage = gradioApp().getElementById("modalImage");
    modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'));
    event.stopPropagation();
}

// 切换灯箱图片的平铺状态
function modalTileImageToggle(event) {
    const modalImage = gradioApp().getElementById("modalImage");
    const modal = gradioApp().getElementById("lightboxModal");
    const isTiling = modalImage.style.display === 'none';
    if (isTiling) {
        modalImage.style.display = 'block';
        modal.style.setProperty('background-image', 'none');
    } else {
        modalImage.style.display = 'none';
        modal.style.setProperty('background-image', `url(${modalImage.src})`);
    }

    event.stopPropagation();
}

// 在 UI 更新后执行
onAfterUiUpdate(function() {
    // 获取所有图片元素
    var fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > div > img');
});
    # 如果 fullImg_preview 不为 null，则遍历其中的每个元素，并对每个元素调用 setupImageForLightbox 函数
    if (fullImg_preview != null) {
        fullImg_preview.forEach(setupImageForLightbox);
    }
    # 调用 updateOnBackgroundChange 函数
    updateOnBackgroundChange();
});

document.addEventListener("DOMContentLoaded", function() {
    // 创建一个 div 元素作为模态框
    const modal = document.createElement('div');
    // 点击模态框时关闭
    modal.onclick = closeModal;
    // 设置模态框的 id
    modal.id = "lightboxModal";
    // 设置模态框可获得焦点
    modal.tabIndex = 0;
    // 添加键盘事件监听器
    modal.addEventListener('keydown', modalKeyHandler, true);

    // 创建一个 div 元素作为模态框控制面板
    const modalControls = document.createElement('div');
    modalControls.className = 'modalControls gradio-container';
    // 将控制面板添加到模态框中
    modal.append(modalControls);

    // 创建一个 span 元素作为缩放按钮
    const modalZoom = document.createElement('span');
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;';
    // 添加点击事件监听器
    modalZoom.addEventListener('click', modalZoomToggle, true);
    modalZoom.title = "Toggle zoomed view";
    // 将缩放按钮添加到控制面板中
    modalControls.appendChild(modalZoom);

    // 创建一个 span 元素作为平铺按钮
    const modalTileImage = document.createElement('span');
    modalTileImage.className = 'modalTileImage cursor';
    modalTileImage.innerHTML = '&#8862;';
    // 添加点击事件监听器
    modalTileImage.addEventListener('click', modalTileImageToggle, true);
    modalTileImage.title = "Preview tiling";
    // 将平铺按钮添加到控制面板中
    modalControls.appendChild(modalTileImage);

    // 创建一个 span 元素作为保存按钮
    const modalSave = document.createElement("span");
    modalSave.className = "modalSave cursor";
    modalSave.id = "modal_save";
    modalSave.innerHTML = "&#x1F5AB;";
    // 添加点击事件监听器
    modalSave.addEventListener("click", modalSaveImage, true);
    modalSave.title = "Save Image(s)";
    // 将保存按钮添加到控制面板中
    modalControls.appendChild(modalSave);

    // 创建一个 span 元素作为关闭按钮
    const modalClose = document.createElement('span');
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;';
    // 点击关闭按钮时关闭模态框
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    // 将关闭按钮添加到控制面板中
    modalControls.appendChild(modalClose);

    // 创建一个 img 元素作为模态框中的图片展示区域
    const modalImage = document.createElement('img');
    modalImage.id = 'modalImage';
    // 点击图片时关闭模态框
    modalImage.onclick = closeModal;
    // 设置图片可获得焦点
    modalImage.tabIndex = 0;
    // 添加键盘事件监听器
    modalImage.addEventListener('keydown', modalKeyHandler, true);
    // 将图片展示区域添加到模态框中
    modal.appendChild(modalImage);

    // 创建一个 a 元素作为上一张图片按钮
    # 创建一个具有指定类名的元素节点，用于显示上一张图片的按钮
    modalPrev.className = 'modalPrev';
    # 设置上一张图片按钮的显示内容为向左箭头
    modalPrev.innerHTML = '&#10094;';
    # 设置上一张图片按钮可以获取焦点
    modalPrev.tabIndex = 0;
    # 给上一张图片按钮添加点击事件监听器，点击时执行 modalPrevImage 函数
    modalPrev.addEventListener('click', modalPrevImage, true);
    # 给上一张图片按钮添加键盘事件监听器，按下键盘时执行 modalKeyHandler 函数
    modalPrev.addEventListener('keydown', modalKeyHandler, true);
    # 将上一张图片按钮添加到模态框中
    modal.appendChild(modalPrev);

    # 创建一个具有指定类名的元素节点，用于显示下一张图片的按钮
    const modalNext = document.createElement('a');
    modalNext.className = 'modalNext';
    # 设置下一张图片按钮的显示内容为向右箭头
    modalNext.innerHTML = '&#10095;';
    # 设置下一张图片按钮可以获取焦点
    modalNext.tabIndex = 0;
    # 给下一张图片按钮添加点击事件监听器，点击时执行 modalNextImage 函数
    modalNext.addEventListener('click', modalNextImage, true);
    # 给下一张图片按钮添加键盘事件监听器，按下键盘时执行 modalKeyHandler 函数
    modalNext.addEventListener('keydown', modalKeyHandler, true);
    
    # 将下一张图片按钮添加到模态框中
    modal.appendChild(modalNext);

    # 尝试将模态框添加到 gradioApp() 中
    try {
        gradioApp().appendChild(modal);
    } catch (e) {
        # 如果添加失败，则将模态框添加到 body 中
        gradioApp().body.appendChild(modal);
    }

    # 将模态框添加到页面的 body 中
    document.body.appendChild(modal);
});
```