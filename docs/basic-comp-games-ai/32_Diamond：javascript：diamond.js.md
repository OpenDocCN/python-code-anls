# `d:/src/tocomm/basic-computer-games\32_Diamond\javascript\diamond.js`

```
// 创建一个名为print的函数，用于在页面上输出文本
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 创建一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为给定的JavaScript代码添加了注释，解释了每个语句的作用。这样做有助于其他程序员理解代码的功能和逻辑。
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
# 结束函数 tab 的定义
}

# 定义函数 tab，接受一个参数 space
function tab(space)
{
    # 初始化字符串 str 为空
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "DIAMOND\n");  // 在指定位置打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("FOR A PRETTY DIAMOND PATTERN,\n");  // 打印提示信息
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21");  // 打印提示信息
    r = parseInt(await input());  // 获取用户输入并转换为整数赋值给变量r
    q = Math.floor(60 / r);  // 计算60除以r的结果并向下取整赋值给变量q
    as = "CC"  // 初始化变量as为字符串"CC"
    x = 1;  // 初始化变量x为1
    y = r;  // 变量y的值为r
    z = 2;  // 初始化变量z为2
    for (l = 1; l <= q; l++) {
        # 循环遍历 l 从 1 到 q
        for (n = x; z < 0 ? n >= y : n <= y; n += z) {
            # 循环遍历 n 从 x 开始，根据 z 的正负决定递增或递减，直到达到 y
            str = "";
            # 初始化空字符串 str
            while (str.length < (r - n) / 2)
                # 当字符串长度小于 (r - n) / 2 时执行循环
                str += " ";
                # 在字符串末尾添加空格
            for (m = 1; m <= q; m++) {
                # 循环遍历 m 从 1 到 q
                c = 1;
                # 初始化变量 c 为 1
                for (a = 1; a <= n; a++) {
                    # 循环遍历 a 从 1 到 n
                    if (c > as.length)
                        # 如果 c 大于 as 数组的长度
                        str += "!";
                        # 在字符串末尾添加感叹号
                    else
                        str += as[c++ - 1];
                        # 在字符串末尾添加 as 数组中的字符，并递增 c
                }
                if (m == q)
                    # 如果 m 等于 q
                    break;
                    # 退出内层循环
                while (str.length < r * m + (r - n) / 2)
                    # 当字符串长度小于 r * m + (r - n) / 2 时执行循环
                    str += " ";
                    # 在字符串末尾添加空格
            }
            print(str + "\n");
            # 打印字符串 str 并换行
        }
    }
# 如果 x 不等于 1，则执行下面的代码块
if (x != 1) {
    # 将 x 赋值为 1
    x = 1;
    # 将 y 赋值为 r
    y = r;
    # 将 z 赋值为 2
    z = 2;
# 如果 x 等于 1，则执行下面的代码块
} else {
    # 将 x 赋值为 r 减去 2
    x = r - 2;
    # 将 y 赋值为 1
    y = 1;
    # 将 z 赋值为 -2
    z = -2;
    # 将 l 减 1
    l--;
}
# 结束 if-else 语句块

# 调用 main 函数
main();
```