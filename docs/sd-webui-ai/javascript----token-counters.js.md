# `stable-diffusion-webui\javascript\token-counters.js`

```py
// 存储更新 token 计数器的函数
let promptTokenCountUpdateFunctions = {};

// 更新 txt2img tokens 的函数
function update_txt2img_tokens(...args) {
    // 从 Gradio 调用
    update_token_counter("txt2img_token_button");
    update_token_counter("txt2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

// 更新 img2img tokens 的函数
function update_img2img_tokens(...args) {
    // 从 Gradio 调用
    update_token_counter("img2img_token_button");
    update_token_counter("img2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

// 更新 token 计数器的函数
function update_token_counter(button_id) {
    promptTokenCountUpdateFunctions[button_id]?.();
}

// 重新计算提示 tokens 的函数
function recalculatePromptTokens(name) {
    promptTokenCountUpdateFunctions[name]?.();
}

// 重新计算 txt2img 提示的函数
function recalculate_prompts_txt2img() {
    // 从 Gradio 调用
    recalculatePromptTokens('txt2img_prompt');
    recalculatePromptTokens('txt2img_neg_prompt');
    return Array.from(arguments);
}

// 重新计算 img2img 提示的函数
function recalculate_prompts_img2img() {
    // 从 Gradio 调用
    recalculatePromptTokens('img2img_prompt');
    recalculatePromptTokens('img2img_neg_prompt');
    return Array.from(arguments);
}

// 设置 token 计数器的函数
function setupTokenCounting(id, id_counter, id_button) {
    // 获取元素
    var prompt = gradioApp().getElementById(id);
    var counter = gradioApp().getElementById(id_counter);
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    // 如果禁用 token 计数器，则隐藏计数器
    if (opts.disable_token_counters) {
        counter.style.display = "none";
        return;
    }

    // 如果计数器已经在提示元素的父元素中，则返回
    if (counter.parentElement == prompt.parentElement) {
        return;
    }

    // 将计数器插入到提示元素之前
    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = "relative";

    // 创建编辑事件监听函数
    var func = onEdit(id, textarea, 800, function() {
        gradioApp().getElementById(id_button)?.click();
    });
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
}

// 设置 token 计数器的函数
function setupTokenCounters() {
    # 设置不同输入框和按钮的令牌计数器
    setupTokenCounting('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
    # 设置不同输入框和按钮的负面令牌计数器
    setupTokenCounting('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
    # 设置不同输入框和按钮的令牌计数器
    setupTokenCounting('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
    # 设置不同输入框和按钮的负面令牌计数器
    setupTokenCounting('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
}



# 闭合一个代码块，表示该代码块的结束
```