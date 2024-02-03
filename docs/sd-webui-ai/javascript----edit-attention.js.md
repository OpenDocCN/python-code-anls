# `stable-diffusion-webui\javascript\edit-attention.js`

```py
function keyupEditAttention(event) {
    // 获取事件的目标元素
    let target = event.originalTarget || event.composedPath()[0];
    // 如果目标元素不匹配指定条件，则返回
    if (!target.matches("*:is([id*='_toprow'] [id*='_prompt'], .prompt) textarea")) return;
    // 如果没有按下 meta 键或 ctrl 键，则返回
    if (!(event.metaKey || event.ctrlKey)) return;

    // 判断是否按下了向上箭头键
    let isPlus = event.key == "ArrowUp";
    // 判断是否按下了向下箭头键
    let isMinus = event.key == "ArrowDown";
    // 如果既不是向上箭头键也不是向下箭头键，则返回
    if (!isPlus && !isMinus) return;

    // 获取光标所在位置的起始和结束位置
    let selectionStart = target.selectionStart;
    let selectionEnd = target.selectionEnd;
    let text = target.value;

    // 定义函数，选择当前括号块
    function selectCurrentParenthesisBlock(OPEN, CLOSE) {
        // 如果起始和结束位置不相同，则返回
        if (selectionStart !== selectionEnd) return false;

        // 查找当前光标周围的开括号
        const before = text.substring(0, selectionStart);
        let beforeParen = before.lastIndexOf(OPEN);
        if (beforeParen == -1) return false;

        let beforeClosingParen = before.lastIndexOf(CLOSE);
        if (beforeClosingParen != -1 && beforeClosingParen > beforeParen) return false;

        // 查找当前光标周围的闭括号
        const after = text.substring(selectionStart);
        let afterParen = after.indexOf(CLOSE);
        if (afterParen == -1) return false;

        let afterOpeningParen = after.indexOf(OPEN);
        if (afterOpeningParen != -1 && afterOpeningParen < afterParen) return false;

        // 设置选择范围为括号之间的文本
        const parenContent = text.substring(beforeParen + 1, selectionStart + afterParen);
        if (/.*:-?[\d.]+/s.test(parenContent)) {
            const lastColon = parenContent.lastIndexOf(":");
            selectionStart = beforeParen + 1;
            selectionEnd = selectionStart + lastColon;
        } else {
            selectionStart = beforeParen + 1;
            selectionEnd = selectionStart + parenContent.length;
        }

        // 设置目标元素的选择范围
        target.setSelectionRange(selectionStart, selectionEnd);
        return true;
    }
    // 选择当前单词的函数
    function selectCurrentWord() {
        // 如果选择的起始位置和结束位置不相等，则返回false
        if (selectionStart !== selectionEnd) return false;
        // 定义空白符分隔符对象
        const whitespace_delimiters = {"Tab": "\t", "Carriage Return": "\r", "Line Feed": "\n"};
        // 初始化分隔符为用户定义的分隔符
        let delimiters = opts.keyedit_delimiters;

        // 将空白符分隔符加入到分隔符中
        for (let i of opts.keyedit_delimiters_whitespace) {
            delimiters += whitespace_delimiters[i];
        }

        // 向前查找，直到找到起始位置
        while (!delimiters.includes(text[selectionStart - 1]) && selectionStart > 0) {
            selectionStart--;
        }

        // 向后查找，直到找到结束位置
        while (!delimiters.includes(text[selectionEnd]) && selectionEnd < text.length) {
            selectionEnd++;
        }

        // 设置选择范围
        target.setSelectionRange(selectionStart, selectionEnd);
        return true;
    }

    // 如果用户没有选择任何内容，则选择当前括号块或单词
    if (!selectCurrentParenthesisBlock('<', '>') && !selectCurrentParenthesisBlock('(', ')') && !selectCurrentParenthesisBlock('[', ']')) {
        selectCurrentWord();
    }

    // 阻止默认事件
    event.preventDefault();

    // 初始化关闭字符和精度
    var closeCharacter = ')';
    var delta = opts.keyedit_precision_attention;
    var start = selectionStart > 0 ? text[selectionStart - 1] : "";
    var end = text[selectionEnd];

    // 如果起始字符为 '<'，则设置关闭字符和精度
    if (start == '<') {
        closeCharacter = '>';
        delta = opts.keyedit_precision_extra;
    }
    // 如果起始字符为 '('，结束字符为 ')' 或者起始字符为 '['，结束字符为 ']'，则转换旧式的 (((强调)))
    } else if (start == '(' && end == ')' || start == '[' && end == ']') {
        // 初始化括号数量
        let numParen = 0;

        // 循环检查起始和结束字符是否匹配
        while (text[selectionStart - numParen - 1] == start && text[selectionEnd + numParen] == end) {
            numParen++;
        }

        // 根据起始字符设置权重
        if (start == "[") {
            weight = (1 / 1.1) ** numParen;
        } else {
            weight = 1.1 ** numParen;
        }

        // 对权重进行四舍五入并根据精度调整
        weight = Math.round(weight / opts.keyedit_precision_attention) * opts.keyedit_precision_attention;

        // 更新文本内容
        text = text.slice(0, selectionStart - numParen) + "(" + text.slice(selectionStart, selectionEnd) + ":" + weight + ")" + text.slice(selectionEnd + numParen);
        selectionStart -= numParen - 1;
        selectionEnd -= numParen - 1;
    } else if (start != '(') {
        // 不包括末尾的空格
        while (selectionEnd > selectionStart && text[selectionEnd - 1] == ' ') {
            selectionEnd--;
        }

        // 如果选择范围为空，则返回
        if (selectionStart == selectionEnd) {
            return;
        }

        // 更新文本内容
        text = text.slice(0, selectionStart) + "(" + text.slice(selectionStart, selectionEnd) + ":1.0)" + text.slice(selectionEnd);

        selectionStart++;
        selectionEnd++;
    }

    // 如果选择范围末尾不是 ':'，则返回
    if (text[selectionEnd] != ':') return;
    // 获取权重的长度
    var weightLength = text.slice(selectionEnd + 1).indexOf(closeCharacter) + 1;
    // 解析权重值
    var weight = parseFloat(text.slice(selectionEnd + 1, selectionEnd + weightLength));
    // 如果权重值无效，则返回
    if (isNaN(weight)) return;

    // 根据正负号调整权重值
    weight += isPlus ? delta : -delta;
    // 保留权重值的精度
    weight = parseFloat(weight.toPrecision(12));
    // 如果权重值为整数，则添加小数点
    if (Number.isInteger(weight)) weight += ".0";

    // 如果结束字符为 ')' 且权重为 1，则删除括号及其内容
    if (closeCharacter == ')' && weight == 1) {
        // 获取结束括号位置
        var endParenPos = text.substring(selectionEnd).indexOf(')');
        // 更新文本内容
        text = text.slice(0, selectionStart - 1) + text.slice(selectionStart, selectionEnd) + text.slice(selectionEnd + endParenPos + 1);
        selectionStart--;
        selectionEnd--;
    } else {
        // 如果不是在光标处插入文本，则在指定位置插入权重值
        text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + weightLength);
    }

    // 将焦点设置回目标元素
    target.focus();
    // 将文本内容设置为修改后的文本
    target.value = text;
    // 设置文本框的选取起始位置
    target.selectionStart = selectionStart;
    // 设置文本框的选取结束位置
    target.selectionEnd = selectionEnd;

    // 更新输入框的内容
    updateInput(target);
}



addEventListener('keydown', (event) => {
    // 添加键盘按下事件监听器，当按键按下时执行后面的函数
    keyupEditAttention(event);
});
```