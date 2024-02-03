# `stable-diffusion-webui\javascript\profilerVisualization.js`

```py
// 创建表格行，包含指定类型的单元格和数据
function createRow(table, cellName, items) {
    // 创建表格行元素
    var tr = document.createElement('tr');
    // 用于存储结果的数组
    var res = [];

    // 遍历数据数组
    items.forEach(function(x, i) {
        // 如果数据为 undefined，则将 null 添加到结果数组中
        if (x === undefined) {
            res.push(null);
            return;
        }

        // 创建指定类型的单元格元素
        var td = document.createElement(cellName);
        // 设置单元格文本内容为数据值
        td.textContent = x;
        // 将单元格添加到表格行中
        tr.appendChild(td);
        // 将单元格元素添加到结果数组中
        res.push(td);

        // 计算单元格的列合并数
        var colspan = 1;
        for (var n = i + 1; n < items.length; n++) {
            if (items[n] !== undefined) {
                break;
            }

            colspan += 1;
        }

        // 如果列合并数大于 1，则设置单元格的列合并数
        if (colspan > 1) {
            td.colSpan = colspan;
        }
    });

    // 将表格行添加到表格中
    table.appendChild(tr);

    // 返回结果数组
    return res;
}

function showProfile(path, cutoff = 0.05) {
    // 显示用户资料
}
```