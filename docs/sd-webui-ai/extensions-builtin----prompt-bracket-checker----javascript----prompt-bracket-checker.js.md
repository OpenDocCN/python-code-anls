# `stable-diffusion-webui\extensions-builtin\prompt-bracket-checker\javascript\prompt-bracket-checker.js`

```
// 检查文本框中的括号匹配情况，包括圆括号、方括号和花括号
function checkBrackets(textArea, counterElt) {
    // 初始化一个空对象用于存储各种括号的数量
    var counts = {};
    // 遍历文本框中的内容，统计各种括号的数量
    (textArea.value.match(/[(){}[\]]/g) || []).forEach(bracket => {
        counts[bracket] = (counts[bracket] || 0) + 1;
    });
    // 初始化一个错误数组
    var errors = [];

    // 检查括号对是否匹配
    function checkPair(open, close, kind) {
        // 如果开括号和闭括号的数量不相等，则记录错误信息
        if (counts[open] !== counts[close]) {
            errors.push(
                `${open}...${close} - Detected ${counts[open] || 0} opening and ${counts[close] || 0} closing ${kind}.`
            );
        }
    }

    // 检查圆括号是否匹配
    checkPair('(', ')', 'round brackets');
    // 检查方括号是否匹配
    checkPair('[', ']', 'square brackets');
    // 检查花括号是否匹配
    checkPair('{', '}', 'curly brackets');
    // 设置错误信息为提示框的标题
    counterElt.title = errors.join('\n');
    // 如果存在错误信息，则将计数器元素的类设置为 'error'，以便显示红色
    counterElt.classList.toggle('error', errors.length !== 0);
}

// 设置括号检查功能
function setupBracketChecking(id_prompt, id_counter) {
    // 获取文本框和计数器元素
    var textarea = gradioApp().querySelector("#" + id_prompt + " > label > textarea");
    var counter = gradioApp().getElementById(id_counter);

    // 如果文本框和计数器元素存在，则添加输入事件监听器
    if (textarea && counter) {
        textarea.addEventListener("input", () => checkBrackets(textarea, counter));
    }
}

// 当页面加载完成时，设置括号检查功能
onUiLoaded(function() {
    // 设置不同文本框和计数器元素的括号检查
    setupBracketChecking('txt2img_prompt', 'txt2img_token_counter');
    setupBracketChecking('txt2img_neg_prompt', 'txt2img_negative_token_counter');
    setupBracketChecking('img2img_prompt', 'img2img_token_counter');
    setupBracketChecking('img2img_neg_prompt', 'img2img_negative_token_counter');
});
```