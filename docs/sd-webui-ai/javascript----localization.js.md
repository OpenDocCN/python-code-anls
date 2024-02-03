# `stable-diffusion-webui\javascript\localization.js`

```
// 定义一个对象，用于存储需要忽略的元素的 id 和类型
var ignore_ids_for_localization = {
    setting_sd_hypernetwork: 'OPTION',
    setting_sd_model_checkpoint: 'OPTION',
    modelmerger_primary_model_name: 'OPTION',
    modelmerger_secondary_model_name: 'OPTION',
    modelmerger_tertiary_model_name: 'OPTION',
    train_embedding: 'OPTION',
    train_hypernetwork: 'OPTION',
    txt2img_styles: 'OPTION',
    img2img_styles: 'OPTION',
    setting_random_artist_categories: 'OPTION',
    setting_face_restoration_model: 'OPTION',
    setting_realesrgan_enabled_models: 'OPTION',
    extras_upscaler_1: 'OPTION',
    extras_upscaler_2: 'OPTION',
};

// 定义正则表达式，用于匹配数字
var re_num = /^[.\d]+$/;
// 定义正则表达式，用于匹配表情符号
var re_emoji = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F9B0}-\u{1F9B3}]/u;

// 定义两个空对象，用于存储原始文本和翻译后的文本
var original_lines = {};
var translated_lines = {};

// 检查是否存在本地化数据
function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

// 获取元素下的所有文本节点
function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

// 检查文本是否可以被翻译
function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;

    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;

    if (parentType == 'OPTION' || parentType == 'SPAN') {
        var pnode = node;
        for (var level = 0; level < 4; level++) {
            pnode = pnode.parentElement;
            if (!pnode) break;

            if (ignore_ids_for_localization[pnode.id] == parentType) return false;
        }
    }

    if (re_num.test(text)) return false;
    if (re_emoji.test(text)) return false;
    return true;
}

// 获取文本的翻译
function getTranslation(text) {
    if (!text) return undefined;

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }

    var tl = localization[text];
}
    # 如果翻译行数不是未定义的
    if (tl !== undefined) {
        # 将翻译行数添加到已翻译行数的字典中
        translated_lines[tl] = 1;
    }

    # 返回翻译行数
    return tl;
}

// 处理文本节点，将文本内容进行处理
function processTextNode(node) {
    // 获取节点的文本内容并去除首尾空格
    var text = node.textContent.trim();

    // 如果节点不能被翻译，则直接返回
    if (!canBeTranslated(node, text)) return;

    // 获取文本的翻译结果
    var tl = getTranslation(text);
    // 如果翻译结果不为空，则替换节点的文本内容
    if (tl !== undefined) {
        node.textContent = tl;
    }
}

// 处理节点，根据节点类型进行相应处理
function processNode(node) {
    // 如果节点是文本节点，则调用处理文本节点的函数
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    // 如果节点有 title 属性，则获取翻译结果并替换
    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) {
            node.title = tl;
        }
    }

    // 如果节点有 placeholder 属性，则获取翻译结果并替换
    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) {
            node.placeholder = tl;
        }
    }

    // 遍历节点下的所有文本节点，并处理
    textNodesUnder(node).forEach(function(node) {
        processTextNode(node);
    });
}

// 对整个页面进行本地化处理
function localizeWholePage() {
    // 处理根节点
    processNode(gradioApp());

    // 定义获取元素的函数
    function elem(comp) {
        var elem_id = comp.props.elem_id ? comp.props.elem_id : "component-" + comp.id;
        return gradioApp().getElementById(elem_id);
    }

    // 遍历页面中的组件，处理 title 和 placeholder 属性
    for (var comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip) {
            let e = elem(comp);

            let tl = e ? getTranslation(e.title) : undefined;
            if (tl !== undefined) {
                e.title = tl;
            }
        }
        if (comp.props.placeholder) {
            let e = elem(comp);
            let textbox = e ? e.querySelector('[placeholder]') : null;

            let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
            if (tl !== undefined) {
                textbox.placeholder = tl;
            }
        }
    }
}

// 导出翻译结果
function dumpTranslations() {
    if (!hasLocalization()) {
        // 如果没有本地化，则对整个页面进行本地化处理
        localizeWholePage();
    }
    var dumped = {};
    if (localization.rtl) {
        dumped.rtl = true;
    }
}
    // 遍历原始文本行中的每一行文本
    for (const text in original_lines) {
        // 如果已经处理过该文本，则跳过
        if (dumped[text] !== undefined) continue;
        // 将本地化文本存储到 dumped 中，如果本地化文本不存在，则使用原始文本
        dumped[text] = localization[text] || text;
    }

    // 返回处理后的文本对象
    return dumped;
}

// 下载本地化文件
function download_localization() {
    // 将翻译数据转换为格式化的 JSON 字符串
    var text = JSON.stringify(dumpTranslations(), null, 4);

    // 创建一个下载链接元素
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', "localization.json");
    element.style.display = 'none';
    document.body.appendChild(element);

    // 模拟点击下载链接
    element.click();

    // 移除下载链接元素
    document.body.removeChild(element);
}

// 当文档加载完成时执行
document.addEventListener("DOMContentLoaded", function() {
    // 如果没有本地化数据，则返回
    if (!hasLocalization()) {
        return;
    }

    // 当 UI 更新时执行回调函数
    onUiUpdate(function(m) {
        m.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                // 处理新增节点
                processNode(node);
            });
        });
    });

    // 本地化整个页面
    localizeWholePage();

    // 如果本地化语言是从右到左
    if (localization.rtl) {
        // 监听节点变化，等待样式加载
        (new MutationObserver((mutations, observer) => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        // 查找所有 rtl 媒体规则
                        for (const x of node.sheet.rules) {
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all'); // 启用这些规则
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), {childList: true});
    }
});
```