# `stable-diffusion-webui\javascript\ui.js`

```
// 与 ui.py 交互的各种函数，代码量不大，不值得将它们放在单独的文件中

// 设置主题
function set_theme(theme) {
    // 获取当前页面的 URL
    var gradioURL = window.location.href;
    // 如果 URL 中不包含主题参数，则重定向到包含主题参数的 URL
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

// 获取所有可见的图库按钮
function all_gallery_buttons() {
    // 获取所有图库按钮元素
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    var visibleGalleryButtons = [];
    // 遍历所有图库按钮，筛选出可见的按钮
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

// 获取选中的图库按钮
function selected_gallery_button() {
    return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
}

// 获取选中的图库按钮的索引
function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
}

// 从图库中提取图像
function extract_image_from_gallery(gallery) {
    if (gallery.length == 0) {
        return [null];
    }
    if (gallery.length == 1) {
        return [gallery[0]];
    }

    var index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // 如果索引超出范围，则使用图库中的第一个图像作为默认值
        index = 0;
    }

    return [gallery[index]];
}

// 兼容性处理，将参数转换为数组
window.args_to_array = Array.from;

// 切换到 txt2img 模式
function switch_to_txt2img() {
    // 点击第一个标签页按钮
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

// 切换到 img2img 模式的指定标签页
function switch_to_img2img_tab(no) {
    // 点击第二个标签页按钮
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    // 点击 img2img 模式下的指定按钮
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}

// 切换到 img2img 模式
function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

// 切换到 sketch 模式
function switch_to_sketch() {
    switch_to_img2img_tab(1);
}
    # 将传入的参数转换为数组并返回
    return Array.from(arguments);
// 结束当前函数
}

// 切换到修复图像选项卡
function switch_to_inpaint() {
    // 调用切换到图像到图像选项卡函数，参数为2
    switch_to_img2img_tab(2);
    // 返回参数数组
    return Array.from(arguments);
}

// 切换到修复图像草图选项卡
function switch_to_inpaint_sketch() {
    // 调用切换到图像到图像选项卡函数，参数为3
    switch_to_img2img_tab(3);
    // 返回参数数组
    return Array.from(arguments);
}

// 切换到额外选项卡
function switch_to_extras() {
    // 获取 Gradio 应用中的选项卡按钮，点击第三个按钮
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();
    // 返回参数数组
    return Array.from(arguments);
}

// 获取选项卡的索引
function get_tab_index(tabId) {
    // 获取指定选项卡中的按钮
    let buttons = gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button');
    // 遍历按钮，找到选中的按钮索引并返回
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].classList.contains('selected')) {
            return i;
        }
    }
    // 如果没有选中的按钮，则返回0
    return 0;
}

// 创建选项卡索引参数
function create_tab_index_args(tabId, args) {
    // 复制参数数组
    var res = Array.from(args);
    // 将选项卡索引放入参数数组的第一个位置
    res[0] = get_tab_index(tabId);
    // 返回修改后的参数数组
    return res;
}

// 获取图像到图像选项卡的索引
function get_img2img_tab_index() {
    // 复制参数数组
    let res = Array.from(arguments);
    // 删除最后两个参数
    res.splice(-2);
    // 将图像到图像选项卡的索引放入参数数组的第一个位置
    res[0] = get_tab_index('mode_img2img');
    // 返回修改后的参数数组
    return res;
}

// 创建提交参数
function create_submit_args(args) {
    // 复制参数数组
    var res = Array.from(args);

    // 当前情况下，txt2img 和 img2img 在生成新图像时会发送回先前的输出参数（txt2img_gallery、generation_info、html_info）
    // 这可能导致上传大量先前生成的图像库，从而导致提交和开始生成之间出现不必要的延迟
    // 我不知道为什么 gradio 在发送输入时也发送输出，但我们可以在这里阻止发送图像库，这似乎对某些人造成了问题
    // 如果 gradio 在某个时候停止发送输出，这可能会导致某些问题
    if (Array.isArray(res[res.length - 3])) {
        res[res.length - 3] = null;
    }

    // 返回修改后的参数数组
    return res;
}

// 显示提交按钮
function showSubmitButtons(tabname, show) {
    // 根据显示参数设置提交按钮和跳过按钮的显示状态
    gradioApp().getElementById(tabname + '_interrupt').style.display = show ? "none" : "block";
    gradioApp().getElementById(tabname + '_skip').style.display = show ? "none" : "block";
}

// 显示恢复进度按钮
function showRestoreProgressButton(tabname, show) {
    # 获取指定标签页的恢复进度按钮元素
    var button = gradioApp().getElementById(tabname + "_restore_progress");
    # 如果按钮不存在，则直接返回
    if (!button) return;

    # 根据参数决定是否显示按钮，显示则设置为flex，否则设置为none
    button.style.display = show ? "flex" : "none";
function submit() {
    // 隐藏提交按钮
    showSubmitButtons('txt2img', false);

    // 生成随机任务ID并存储到本地
    var id = randomId();
    localSet("txt2img_task_id", id);

    // 请求进度并显示进度条
    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
        // 显示提交按钮
        showSubmitButtons('txt2img', true);
        // 移除本地存储的任务ID
        localRemove("txt2img_task_id");
        // 隐藏恢复进度按钮
        showRestoreProgressButton('txt2img', false);
    });

    // 创建提交参数
    var res = create_submit_args(arguments);

    // 设置任务ID到参数中
    res[0] = id;

    return res;
}

function submit_img2img() {
    // 隐藏提交按钮
    showSubmitButtons('img2img', false);

    // 生成随机任务ID并存储到本地
    var id = randomId();
    localSet("img2img_task_id", id);

    // 请求进度并显示进度条
    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
        // 显示提交按钮
        showSubmitButtons('img2img', true);
        // 移除本地存储的任务ID
        localRemove("img2img_task_id");
        // 隐藏恢复进度按钮
        showRestoreProgressButton('img2img', false);
    });

    // 创建提交参数
    var res = create_submit_args(arguments);

    // 设置任务ID和模式到参数中
    res[0] = id;
    res[1] = get_tab_index('mode_img2img');

    return res;
}

function submit_extras() {
    // 隐藏提交按钮
    showSubmitButtons('extras', false);

    // 生成随机任务ID
    var id = randomId();

    // 请求进度并显示进度条
    requestProgress(id, gradioApp().getElementById('extras_gallery_container'), gradioApp().getElementById('extras_gallery'), function() {
        // 显示提交按钮
        showSubmitButtons('extras', true);
    });

    // 创建提交参数
    var res = create_submit_args(arguments);

    // 设置任务ID到参数中
    res[0] = id;

    // 打印参数并返回
    console.log(res);
    return res;
}

function restoreProgressTxt2img() {
    // 隐藏恢复进度按钮
    showRestoreProgressButton("txt2img", false);
    // 从本地获取任务ID
    var id = localGet("txt2img_task_id");

    // 如果存在任务ID，则请求进度并显示进度条
    if (id) {
        requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
            // 显示提交按钮
            showSubmitButtons('txt2img', true);
        }, null, 0);
    }

    return id;
}

function restoreProgressImg2img() {
    // 隐藏恢复进度按钮
    showRestoreProgressButton("img2img", false);

    // 从本地获取任务ID
    var id = localGet("img2img_task_id");
    # 如果存在id，则执行以下操作
    if (id) {
        # 请求进度并显示在指定的元素上，完成后显示提交按钮
        requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
            showSubmitButtons('img2img', true);
        }, null, 0);
    }
    
    # 返回id
    return id;
}

/**
 * 配置 `tabname` 上的宽度和高度元素，以接受以 "宽度 x 高度" 形式粘贴的分辨率。
 */
function setupResolutionPasting(tabname) {
    // 获取宽度和高度输入框元素
    var width = gradioApp().querySelector(`#${tabname}_width input[type=number]`);
    var height = gradioApp().querySelector(`#${tabname}_height input[type=number]`);
    // 遍历宽度和高度输入框元素
    for (const el of [width, height]) {
        // 添加粘贴事件监听器
        el.addEventListener('paste', function(event) {
            // 获取粘贴的数据
            var pasteData = event.clipboardData.getData('text/plain');
            // 解析粘贴的数据，匹配宽度和高度
            var parsed = pasteData.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (parsed) {
                // 更新宽度和高度输入框的值
                width.value = parsed[1];
                height.value = parsed[2];
                // 更新输入框
                updateInput(width);
                updateInput(height);
                // 阻止默认粘贴行为
                event.preventDefault();
            }
        });
    }
}

// 当 UI 加载完成时执行
onUiLoaded(function() {
    // 显示恢复进度按钮
    showRestoreProgressButton('txt2img', localGet("txt2img_task_id"));
    showRestoreProgressButton('img2img', localGet("img2img_task_id"));
    // 配置文本到图像和图像到图像的分辨率粘贴
    setupResolutionPasting('txt2img');
    setupResolutionPasting('img2img');
});

// 模型合并函数
function modelmerger() {
    // 生成随机 ID
    var id = randomId();
    // 请求进度
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function() {});

    // 创建提交参数
    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

// 请求样式名称函数
function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    // 提示用户输入样式名称
    var name_ = prompt('Style name:');
    return [name_, prompt_text, negative_prompt_text];
}

// 确认清除提示函数
function confirm_clear_prompt(prompt, negative_prompt) {
    // 如果确认删除提示
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}

// 选项对象
var opts = {};
// 在 UI 更新后执行
onAfterUiUpdate(function() {
    // 如果选项对象为空，则返回
    if (Object.keys(opts).length != 0) return;

    // 获取设置 JSON 元素
    var json_elem = gradioApp().getElementById('settings_json');
    if (json_elem == null) return;

    // 获取文本域元素
    var textarea = json_elem.querySelector('textarea');
    var jsdata = textarea.value;
    // 解析 JSON 数据
    opts = JSON.parse(jsdata);
}
    // 执行 optionsChangedCallbacks 回调函数数组中的所有回调函数
    executeCallbacks(optionsChangedCallbacks); /*global optionsChangedCallbacks*/

    // 为 textarea 元素定义一个 value 属性，实现设置和获取值的功能
    Object.defineProperty(textarea, 'value', {
        // 设置属性时的操作
        set: function(newValue) {
            // 获取 textarea 的 value 属性描述符
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            // 获取旧值
            var oldValue = valueProp.get.call(textarea);
            // 设置新值
            valueProp.set.call(textarea, newValue);

            // 如果新值和旧值不相等，则解析新值为 JSON 对象
            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            // 执行 optionsChangedCallbacks 回调函数数组中的所有回调函数
            executeCallbacks(optionsChangedCallbacks);
        },
        // 获取属性时的操作
        get: function() {
            // 获取 textarea 的 value 属性描述符
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            // 返回 textarea 的值
            return valueProp.get.call(textarea);
        }
    });

    // 隐藏 json_elem 元素的父元素
    json_elem.parentElement.style.display = "none";

    // 设置 token 计数器
    setupTokenCounters();
// 当选项改变时触发的函数
onOptionsChanged(function() {
    // 获取元素
    var elem = gradioApp().getElementById('sd_checkpoint_hash');
    // 获取检查点哈希值或为空字符串
    var sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    // 获取哈希值的前10位
    var shorthash = sd_checkpoint_hash.substring(0, 10);

    // 如果元素存在且内容不等于哈希值的前10位
    if (elem && elem.textContent != shorthash) {
        // 更新元素内容为哈希值的前10位
        elem.textContent = shorthash;
        // 设置元素标题为完整哈希值
        elem.title = sd_checkpoint_hash;
        // 设置元素链接为搜索完整哈希值的谷歌搜索链接
        elem.href = "https://google.com/search?q=" + sd_checkpoint_hash;
    }
});

// 初始化变量
let txt2img_textarea, img2img_textarea = undefined;

// 重新加载页面
function restart_reload() {
    // 更改页面内容为重新加载提示
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';

    // 发送请求以检查是否重新加载
    var requestPing = function() {
        requestGet("./internal/ping", {}, function(data) {
            // 重新加载页面
            location.reload();
        }, function() {
            // 如果请求失败，延迟500毫秒后再次发送请求
            setTimeout(requestPing, 500);
        });
    };

    // 延迟2秒后发送请求
    setTimeout(requestPing, 2000);

    // 返回空数组
    return [];
}

// 模拟 Gradio 文本框组件的 `input` DOM 事件
function updateInput(target) {
    // 创建事件
    let e = new Event("input", {bubbles: true});
    // 设置事件目标
    Object.defineProperty(e, "target", {value: target});
    // 分发事件
    target.dispatchEvent(e);
}

// 初始化变量
var desiredCheckpointName = null;

// 选择检查点
function selectCheckpoint(name) {
    // 设置所需检查点名称
    desiredCheckpointName = name;
    // 点击更改检查点按钮
    gradioApp().getElementById('change_checkpoint').click();
}

// 获取当前 Img2Img 源图像的分辨率
function currentImg2imgSourceResolution(w, h, scaleBy) {
    // 获取 Img2Img 图像元素
    var img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
    // 如果图像存在，返回其自然宽度、高度和缩放比例
    return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

// 在更改图像后更新 Img2Img 调整大小文本
function updateImg2imgResizeToTextAfterChangingImage() {
    // 在此时调用时，图像尚未被替换
    // 可能有更好的解决方案，但这是简单直接的方法
}
    # 在延迟500毫秒后执行以下代码
    setTimeout(function() {
        # 通过gradioApp()获取元素并模拟点击操作
        gradioApp().getElementById('img2img_update_resize_to').click();
    }, 500);
    
    # 返回一个空列表
    return [];
// 设置随机种子
function setRandomSeed(elem_id) {
    // 获取指定元素的输入框
    var input = gradioApp().querySelector("#" + elem_id + " input");
    // 如果输入框不存在，则返回空数组
    if (!input) return [];

    // 将输入框的值设置为"-1"
    input.value = "-1";
    // 更新输入框的数值
    updateInput(input);
    // 返回空数组
    return [];
}

// 切换宽度和高度
function switchWidthHeight(tabname) {
    // 获取指定标签页的宽度和高度输入框
    var width = gradioApp().querySelector("#" + tabname + "_width input[type=number]");
    var height = gradioApp().querySelector("#" + tabname + "_height input[type=number]");
    // 如果宽度或高度输入框不存在，则返回空数组
    if (!width || !height) return [];

    // 交换宽度和高度的值
    var tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    // 更新宽度和高度输入框的数值
    updateInput(width);
    updateInput(height);
    // 返回空数组
    return [];
}

// 定时编辑
var onEditTimers = {};

// 在用户输入元素后的一段时间后调用函数
function onEdit(editId, elem, afterMs, func) {
    // 编辑后的操作
    var edited = function() {
        // 获取已存在的定时器
        var existingTimer = onEditTimers[editId];
        // 如果已存在定时器，则清除
        if (existingTimer) clearTimeout(existingTimer);

        // 设置新的定时器，延迟执行函数
        onEditTimers[editId] = setTimeout(func, afterMs);
    };

    // 监听输入元素的输入事件
    elem.addEventListener("input", edited);

    // 返回编辑后的操作函数
    return edited;
}
```