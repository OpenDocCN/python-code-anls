# `basic-computer-games\60_Mastermind\javascript\mastermind.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
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
        // 创建一个输入框元素
        input_element = document.createElement("INPUT");

        // 输出提示符
        print("? ");
        // 设置输入框属性
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");
        // 将输入框添加到指定元素中
        document.getElementById("output").appendChild(input_element);
        // 设置输入框焦点
        input_element.focus();
        // 初始化输入字符串
        input_str = undefined;
        // 监听键盘事件
        input_element.addEventListener("keydown", function (event) {
            // 如果按下回车键
            if (event.keyCode == 13) {
                // 获取输入的字符串
                input_str = input_element.value;
                // 移除输入框
                document.getElementById("output").removeChild(input_element);
                // 输出输入的字符串
                print(input_str);
                print("\n");
                // 解析输入的字符串
                resolve(input_str);
            }
        });
    });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化变量
var p9;
var c9;
var b;
var w;
var f;
var m;

var qa;
var sa;
var ss;
var as;
var gs;
var hs;

// 初始化 qa 数组
function initialize_qa()
{
    for (s = 1; s <= p9; s++)
        qa[s] = 0;
}

// 增加 qa 数组的值
function increment_qa()
{
    if (qa[1] <= 0) {
        // 如果为零，这是我们的第一个增量：将所有值设为1
        for (s = 1; s <= p9; s++)
            qa[s] = 1;
    } else {
        q = 1;
        while (1) {
            qa[q] = qa[q] + 1;
            if (qa[q] <= c9)
                return;
            qa[q] = 1;
            q++;
        }
    }
}

// 转换 qa 数组
function convert_qa()
{
    for (s = 1; s <= p9; s++) {
        as[s] = ls.substr(qa[s] - 1, 1);
    }
}

// 获取数字
function get_number()
{
    b = 0;
    w = 0;
    f = 0;
    for (s = 1; s <= p9; s++) {
        if (gs[s] == as[s]) {
            b++;
            gs[s] = String.fromCharCode(f);
            as[s] = String.fromCharCode(f + 1);
            f += 2;
        } else {
            for (t = 1; t <= p9; t++) {
                if (gs[s] == as[t] && gs[t] != as[t]) {
                    w++;
                    as[t] = String.fromCharCode(f);
                    gs[s] = String.fromCharCode(f + 1);
                    f += 2;
                    break;
                }
            }
        }
    }
}

// 转换 qa 数组到 hs 数组
function convert_qa_hs()
{
    for (s = 1; s <= p9; s++) {
        hs[s] = ls.substr(qa[s] - 1, 1);
    }
}

// 复制 hs 数组到 gs 数组
function copy_hs()
{
    for (s = 1; s <= p9; s++) {
        gs[s] = hs[s];
    }
}

// 输出棋盘
function board_printout()
{
    print("\n");
    print("BOARD\n");
    print("MOVE     GUESS          BLACK     WHITE\n");
    for (z = 1; z <= m - 1; z++) {
        str = " " + z + " ";
        while (str.length < 9)
            str += " ";
        str += ss[z];
        while (str.length < 25)
            str += " ";
        str += sa[z][1];
        while (str.length < 35)
            str += " ";
        str += sa[z][2];
        print(str + "\n");
    }
    print("\n");
}

// 退出函数
function quit()
{
    print("QUITTER!  MY COMBINATION WAS: ");
    convert_qa();
    for (x = 1; x <= p9; x++) {
        print(as[x]);
    }
    print("\n");
    print("GOOD BYE\n");
}

// 显示得分
function show_score()
{
    print("SCORE:\n");
    show_points();
}

// 显示分数
function show_points()
{
    print("     COMPUTER " + c + "\n");
    print("     HUMAN    " + h + "\n");
    print("\n");
}

// 颜色数组
var color = ["BLACK", "WHITE", "RED", "GREEN",
             "ORANGE", "YELLOW", "PURPLE", "TAN"];

// 主程序
async function main()
}

// 调用主程序
main();

```