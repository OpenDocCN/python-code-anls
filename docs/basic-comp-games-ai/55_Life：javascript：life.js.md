# `d:/src/tocomm/basic-computer-games\55_Life\javascript\life.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入类型
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var bs = [];  # 创建一个空数组 bs
var a = [];  # 创建一个空数组 a

// Main program  # 主程序开始
async function main()  # 异步函数 main 开始
{
    print(tab(34) + "LIFE\n");  # 打印在第 34 列开始的字符串 "LIFE"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印在第 15 列开始的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("\n");  # 打印一个空行
    print("ENTER YOUR PATTERN:\n");  # 打印提示信息 "ENTER YOUR PATTERN:"
    x1 = 1;  # 初始化变量 x1 为 1
    y1 = 1;  # 初始化变量 y1 为 1
    x2 = 24;  # 初始化变量 x2 为 24
    y2 = 70;  # 初始化变量 y2 为 70
    # 初始化变量，创建空字符串和空数组
    for (c = 1; c <= 24; c++) {
        bs[c] = "";
        a[c] = [];
        for (d = 1; d <= 70; d++)
            a[c][d] = 0;
    }
    # 初始化变量 c
    c = 1;
    # 循环读取输入，直到输入为 "DONE" 时跳出循环
    while (1) {
        bs[c] = await input();
        if (bs[c] == "DONE") {
            bs[c] = "";
            break;
        }
        # 如果输入字符串以 "." 开头，则去掉 "."，加上空格
        if (bs[c].substr(0, 1) == ".")
            bs[c] = " " + bs[c].substr(1);
        c++;
    }
    # 减去多余的 1
    c--;
    # 初始化变量 l
    l = 0;
    # 循环处理输入字符串
    for (x = 1; x <= c - 1; x++) {
        # 如果当前 bs[x] 的长度大于 l，则将 l 更新为 bs[x] 的长度
        if (bs[x].length > l)
            l = bs[x].length;
    }
    # 根据 c 和 l 计算出 x1 和 y1 的值
    x1 = 11 - (c >> 1);
    y1 = 33 - (l >> 1);
    # 初始化变量 p
    p = 0;
    # 遍历 bs 数组，将非空格字符对应的位置在数组 a 中标记为 1，并更新 p 的值
    for (x = 1; x <= c; x++) {
        for (y = 1; y <= bs[x].length; y++) {
            if (bs[x][y - 1] != " ") {
                a[x1 + x][y1 + y] = 1;
                p++;
            }
        }
    }
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    # 初始化变量 i9 和 g
    i9 = false;
    g = 0;
    # 进入循环，条件为 g 小于 100
    while (g < 100) {
        # 打印生成的信息，包括 GENERATION 和 POPULATION
        print("GENERATION: " + g + " POPULATION: " + p + " ");
        # 如果 i9 为真，则打印 "INVALID!"
        if (i9)
            print("INVALID!");
        # 初始化 x3, y3, x4, y4, p
        x3 = 24;
        y3 = 70;
        x4 = 1;
        y4 = 1;
        p = 0;
        # g 自增
        g++;
        # 循环打印换行符
        for (x = 1; x <= x1 - 1; x++)
            print("\n");
        # 循环打印换行符，并初始化 str
        for (x = x1; x <= x2; x++) {
            print("\n");
            str = "";
            # 循环遍历 y1 到 y2，根据 a[x][y] 的值进行相应操作
            for (y = y1; y <= y2; y++) {
                # 如果 a[x][y] 为 2，则将其置为 0 并继续下一次循环
                if (a[x][y] == 2) {
                    a[x][y] = 0;
                    continue;
                # 如果 a[x][y] 为 3，则将其置为 1
                } else if (a[x][y] == 3) {
                    a[x][y] = 1;
                } else if (a[x][y] != 1) {  # 如果二维数组中当前位置的值不等于1，跳过本次循环
                    continue;
                }
                while (str.length < y)  # 当字符串长度小于y时，往字符串末尾添加空格
                    str += " ";
                str += "*";  # 在字符串末尾添加"*"
                if (x < x3)  # 如果当前x小于x3，更新x3的值为当前x
                    x3 = x;
                if (x > x4)  # 如果当前x大于x4，更新x4的值为当前x
                    x4 = x;
                if (y < y3)  # 如果当前y小于y3，更新y3的值为当前y
                    y3 = y;
                if (y > y4)  # 如果当前y大于y4，更新y4的值为当前y
                    y4 = y;
            }
            print(str);  # 打印字符串
        }
        for (x = x2 + 1; x <= 24; x++)  # 循环打印24个换行符
            print("\n");
        x1 = x3;  # 更新x1的值为x3
        x2 = x4;  # 将变量 x4 的值赋给变量 x2
        y1 = y3;  # 将变量 y3 的值赋给变量 y1
        y2 = y4;  # 将变量 y4 的值赋给变量 y2
        if (x1 < 3) {  # 如果变量 x1 的值小于 3
            x1 = 3;  # 将变量 x1 的值设为 3
            i9 = true;  # 将变量 i9 的值设为 true
        }
        if (x2 > 22) {  # 如果变量 x2 的值大于 22
            x2 = 22;  # 将变量 x2 的值设为 22
            i9 = true;  # 将变量 i9 的值设为 true
        }
        if (y1 < 3) {  # 如果变量 y1 的值小于 3
            y1 = 3;  # 将变量 y1 的值设为 3
            i9 = true;  # 将变量 i9 的值设为 true
        }
        if (y2 > 68) {  # 如果变量 y2 的值大于 68
            y2 = 68;  # 将变量 y2 的值设为 68
            i9 = true;  # 将变量 i9 的值设为 true
        }
        p = 0;  # 将变量 p 的值设为 0
        for (x = x1 - 1; x <= x2 + 1; x++) {  # 从 x1 - 1 到 x2 + 1 遍历 x 坐标
            for (y = y1 - 1; y <= y2 + 1; y++) {  # 从 y1 - 1 到 y2 + 1 遍历 y 坐标
                c = 0;  # 初始化 c 为 0
                for (i = x - 1; i <= x + 1; i++) {  # 从 x - 1 到 x + 1 遍历 i 坐标
                    for (j = y - 1; j <= y + 1; j++) {  # 从 y - 1 到 y + 1 遍历 j 坐标
                        if (a[i][j] == 1 || a[i][j] == 2)  # 如果 a[i][j] 的值为 1 或 2
                            c++;  # c 自增 1
                    }
                }
                if (a[x][y] == 0) {  # 如果 a[x][y] 的值为 0
                    if (c == 3) {  # 如果 c 的值为 3
                        a[x][y] = 3;  # 将 a[x][y] 的值设为 3
                        p++;  # p 自增 1
                    }
                } else {  # 否则
                    if (c < 3 || c > 4) {  # 如果 c 的值小于 3 或大于 4
                        a[x][y] = 2;  # 将 a[x][y] 的值设为 2
                    } else {  # 否则
                        p++;  # p 自增 1
                    }
        }
    }
}
x1--;  // 将变量 x1 减一
y1--;  // 将变量 y1 减一
x2++;  // 将变量 x2 加一
y2++;  // 将变量 y2 加一
}

main();  // 调用主函数
```