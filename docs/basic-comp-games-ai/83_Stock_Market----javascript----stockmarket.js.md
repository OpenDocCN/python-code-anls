# `basic-computer-games\83_Stock_Market\javascript\stockmarket.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
        // 创建一个输入元素
        input_element = document.createElement("INPUT");

        // 打印提示符
        print("? ");
        // 设置输入元素的类型和长度
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        // 将输入元素添加到输出元素中
        document.getElementById("output").appendChild(input_element);
        // 让输入元素获得焦点
        input_element.focus();
        // 初始化输入字符串
        input_str = undefined;
        // 监听键盘事件
        input_element.addEventListener("keydown", function (event) {
            // 如果按下回车键
            if (event.keyCode == 13) {
                // 获取输入的字符串
                input_str = input_element.value;
                // 从输出元素中移除输入元素
                document.getElementById("output").removeChild(input_element);
                // 打印输入的字符串
                print(input_str);
                // 打印换行符
                print("\n");
                // 解析输入的字符串
                resolve(input_str);
            }
        });
    });
}

// 定义一个生成空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 声明变量
var sa = [];
var pa = [];
var za = [];
var ca = [];
var i1;
var n1;
var e1;
var i2;
var n2;
var e2;
var x1;
var w3;
var t8;
var a;
var s4;

// 新股票值 - 子程序
function randomize_initial()
{
    // 随机生成新的股票值
    // 根据前一天的值
    // n1,n2 是分别决定股票 i1 增加 10 点和股票 i2 减少 10 点的随机天数
    // 如果 n1 天已经过去，选择一个 i1，设置 e1，确定新的 n1
    if (n1 <= 0) {
        i1 = Math.floor(4.99 * Math.random() + 1);
        n1 = Math.floor(4.99 * Math.random() + 1);
        e1 = 1;
    }
    // 如果 n2 天已经过去，选择一个 i2，设置 e2，确定新的 n2
    if (n2 <= 0) {
        i2 = Math.floor(4.99 * Math.random() + 1);
        n2 = Math.floor(4.99 * Math.random() + 1);
        e2 = 1;
    }
    // 从 n1 和 n2 中减去一天
    n1--;
    n2--;
    // 遍历所有股票
    for (i = 1; i <= 5; i++) {
        x1 = Math.random();
        if (x1 < 0.25) {
            x1 = 0.25;
        } else if (x1 < 0.5) {
            x1 = 0.5;
        } else if (x1 < 0.75) {
            x1 = 0.75;
        } else {
            x1 = 0.0;
        }
        // 大变化常数：w3（初始设置为零）
        w3 = 0;
        if (e1 >= 1 && Math.floor(i1 + 0.5) == Math.floor(i + 0.5)) {
            // 给这只股票加 10 点；重置 e1
            w3 = 10;
            e1 = 0;
        }
        if (e2 >= 1 && Math.floor(i2 + 0.5) == Math.floor(i + 0.5)) {
            // 从这只股票减去 10 点；重置 e2
            w3 -= 10;
            e2 = 0;
        }
        // c(i) 是股票值的变化
        ca[i] = Math.floor(a * sa[i]) + x1 + Math.floor(3 - 6 * Math.random() + 0.5) + w3;
        ca[i] = Math.floor(100 * ca[i] + 0.5) / 100;
        sa[i] += ca[i];
        if (sa[i] <= 0) {
            ca[i] = 0;
            sa[i] = 0;
        } else {
            sa[i] = Math.floor(100 * sa[i] + 0.5) / 100;
        }
    }
    // 在 t8 天后随机改变趋势符号和斜率
    if (--t8 < 1) {
        // 随机改变趋势符号和斜率（a），以及趋势的持续时间（t8）
        t8 = Math.floor(4.99 * Math.random() + 1);
        a = Math.floor((Math.random() / 10) * 100 + 0.5) / 100;
        s4 = Math.random();
        if (s4 > 0.5)
            a = -a;
    }
}

// 主程序
async function main()
{
    // 主程序逻辑
}

// 调用主程序
main();

```