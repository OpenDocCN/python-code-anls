# `stable-diffusion-webui\javascript\edit-order.js`

```py
// 定义处理键盘按键事件的函数
function keyupEditOrder(event) {
    // 如果未启用键盘编辑移动功能，则直接返回
    if (!opts.keyedit_move) return;

    // 获取事件的目标元素
    let target = event.originalTarget || event.composedPath()[0];
    // 如果目标元素不是匹配指定条件的文本框，则直接返回
    if (!target.matches("*:is([id*='_toprow'] [id*='_prompt'], .prompt) textarea")) return;
    // 如果未按下 Alt 键，则直接返回
    if (!event.altKey) return;

    // 判断是否按下左箭头键或右箭头键
    let isLeft = event.key == "ArrowLeft";
    let isRight = event.key == "ArrowRight";
    // 如果不是左箭头键或右箭头键，则直接返回
    if (!isLeft && !isRight) return;
    // 阻止默认事件
    event.preventDefault();

    // 获取文本框的选中文本起始位置和结束位置
    let selectionStart = target.selectionStart;
    let selectionEnd = target.selectionEnd;
    let text = target.value;
    let items = text.split(",");
    // 计算选中文本起始位置和结束位置所在逗号的索引
    let indexStart = (text.slice(0, selectionStart).match(/,/g) || []).length;
    let indexEnd = (text.slice(0, selectionEnd).match(/,/g) || []).length;
    let range = indexEnd - indexStart + 1;

    // 如果按下左箭头键且选中文本起始位置大于 0
    if (isLeft && indexStart > 0) {
        // 在选中文本起始位置前插入选中文本，同时删除原有选中文本
        items.splice(indexStart - 1, 0, ...items.splice(indexStart, range));
        target.value = items.join();
        // 更新文本框的选中文本起始位置和结束位置
        target.selectionStart = items.slice(0, indexStart - 1).join().length + (indexStart == 1 ? 0 : 1);
        target.selectionEnd = items.slice(0, indexEnd).join().length;
    } 
    // 如果按下右箭头键且选中文本结束位置小于 items 数组长度减 1
    else if (isRight && indexEnd < items.length - 1) {
        // 在选中文本结束位置后插入选中文本，同时删除原有选中文本
        items.splice(indexStart + 1, 0, ...items.splice(indexStart, range));
        target.value = items.join();
        // 更新文本框的选中文本起始位置和结束位置
        target.selectionStart = items.slice(0, indexStart + 1).join().length + 1;
        target.selectionEnd = items.slice(0, indexEnd + 2).join().length;
    }

    // 阻止默认事件
    event.preventDefault();
    // 更新输入框
    updateInput(target);
}

// 添加键盘按键事件监听器
addEventListener('keydown', (event) => {
    keyupEditOrder(event);
});
```