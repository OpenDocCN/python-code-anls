# `basic-computer-games\88_3-D_Tic-Tac-Toe\javascript\qubit.js`

```

// 定义一个打印函数，用于在页面上输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象，用于异步获取用户输入
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
        // 创建一个输入框元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");
        // 设置输入框属性
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        // 将输入框添加到页面上
        document.getElementById("output").appendChild(input_element);
        // 让输入框获得焦点
        input_element.focus();
        // 初始化输入字符串
        input_str = undefined;
        // 监听输入框的键盘事件
        input_element.addEventListener("keydown", function (event) {
            // 如果按下的是回车键
            if (event.keyCode == 13) {
                // 获取输入的字符串
                input_str = input_element.value;
                // 从页面上移除输入框
                document.getElementById("output").removeChild(input_element);
                // 在页面上输出输入的字符串
                print(input_str);
                print("\n");
                // 解析输入的字符串并返回
                resolve(input_str);
            }
        });
    });
}

// 定义一个函数，用于生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化一些数组
var xa = [];
var la = [];
var ma = [[], ...]; // 省略部分数组初始化
var ya = [,1,49,52,4,13,61,64,16,22,39,23,38,26,42,27,43]; // 初始化一个数组

// 定义一个函数，用于在页面上展示棋盘
function show_board()
{
    // 输出一些空行
    for (xx = 1; xx <= 9; xx++)
        print("\n");
    // 遍历棋盘数组，生成棋盘的字符串并输出到页面上
    // 省略部分代码
}

// 定义一个函数，用于处理棋盘数组
function process_board()
{
    // 遍历棋盘数组，将特定值的元素置为0
    // 省略部分代码
}

// 定义一个函数，用于检查棋盘上的线
function check_for_lines()
{
    // 遍历 ma 数组，计算每一行的和并存储到 la 数组中
    // 省略部分代码
}

// 定义一个函数，用于在页面上展示特定位置的方块
function show_square(m)
{
    // 根据位置 m 计算出对应的坐标并输出到页面上
    // 省略部分代码
}

// 定义一个函数，用于选择机器的移动
function select_move() {
    // 根据条件选择机器的移动，并在页面上展示
    // 省略部分代码
}

// 主控制部分，异步执行 main 函数
async function main()
{
    // 省略部分代码
}

// 执行主控制部分
main();

```