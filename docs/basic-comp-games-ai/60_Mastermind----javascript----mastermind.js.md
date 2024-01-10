# `basic-computer-games\60_Mastermind\javascript\mastermind.js`

```
// 定义一个打印函数，将字符串输出到指定的元素上
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素上
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
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

// 定义变量
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
        # 如果条件不满足，则执行以下代码块
        q = 1;
        # 初始化变量 q 为 1
        while (1) {
            # 进入无限循环
            qa[q] = qa[q] + 1;
            # 对数组 qa 中索引为 q 的元素加 1
            if (qa[q] <= c9)
                # 如果 qa[q] 小于等于 c9，则返回
                return;
            qa[q] = 1;
            # 否则，将 qa[q] 重置为 1
            q++;
            # q 自增
        }
    }
// 结束函数定义
}

// 将 ls 中的字符根据 qa 数组的值转换为 as 数组
function convert_qa()
{
    // 遍历 qa 数组
    for (s = 1; s <= p9; s++) {
        // 从 ls 中截取指定位置的字符，存入 as 数组
        as[s] = ls.substr(qa[s] - 1, 1);
    }
}

// 获取数字
function get_number()
{
    // 初始化变量
    b = 0;
    w = 0;
    f = 0;
    // 遍历数组
    for (s = 1; s <= p9; s++) {
        // 判断字符是否相等
        if (gs[s] == as[s]) {
            // 如果相等，增加 b 的计数，修改 gs 和 as 数组的值
            b++;
            gs[s] = String.fromCharCode(f);
            as[s] = String.fromCharCode(f + 1);
            f += 2;
        } else {
            // 如果不相等，再次遍历数组
            for (t = 1; t <= p9; t++) {
                // 判断字符是否相等且位置不同
                if (gs[s] == as[t] && gs[t] != as[t]) {
                    // 增加 w 的计数，修改 as 和 gs 数组的值
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

// 将 ls 中的字符根据 qa 数组的值转换为 hs 数组
function convert_qa_hs()
{
    // 遍历 qa 数组
    for (s = 1; s <= p9; s++) {
        // 从 ls 中截取指定位置的字符，存入 hs 数组
        hs[s] = ls.substr(qa[s] - 1, 1);
    }
}

// 复制 hs 数组到 gs 数组
function copy_hs()
{
    // 遍历数组
    for (s = 1; s <= p9; s++) {
        // 复制 hs 数组的值到 gs 数组
        gs[s] = hs[s];
    }
}

// 打印游戏板
function board_printout()
{
    // 打印空行
    print("\n");
    // 打印标题
    print("BOARD\n");
    // 打印表头
    print("MOVE     GUESS          BLACK     WHITE\n");
    // 遍历数组
    for (z = 1; z <= m - 1; z++) {
        // 构建字符串
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
        // 打印字符串
        print(str + "\n");
    }
    // 打印空行
    print("\n");
}

// 退出游戏
function quit()
{
    // 打印消息
    print("QUITTER!  MY COMBINATION WAS: ");
    // 调用 convert_qa 函数
    convert_qa();
    // 遍历数组，打印值
    for (x = 1; x <= p9; x++) {
        print(as[x]);
    }
    // 打印消息
    print("\n");
    print("GOOD BYE\n");
}

// 显示分数
function show_score()
{
    // 打印消息
    print("SCORE:\n");
    // 调用 show_points 函数
    show_points();
}

// 显示分数
function show_points()
{
    // 打印消息
    print("     COMPUTER " + c + "\n");
    print("     HUMAN    " + h + "\n");
    // 打印空行
    print("\n");
}

// 颜色数组
var color = ["BLACK", "WHITE", "RED", "GREEN",
             "ORANGE", "YELLOW", "PURPLE", "TAN"];

// 主程序
async function main()
{
    // 打印标题
    print(tab(30) + "MASTERMIND\n");
    # 打印标题和作者信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    # 打印游戏信息
    //
    //  MASTERMIND II
    //  STEVE NORTH
    //  CREATIVE COMPUTING
    //  PO BOX 789-M MORRISTOWN NEW JERSEY 07960
    //
    //
    # 进入游戏循环
    while (1) {
        # 获取颜色数量
        print("NUMBER OF COLORS");
        c9 = parseInt(await input());
        # 如果颜色数量小于等于8，则跳出循环
        if (c9 <= 8)
            break;
        # 打印错误信息
        print("NO MORE THAN 8, PLEASE!\n");
    }
    # 获取位置数量
    print("NUMBER OF POSITIONS");
    p9 = parseInt(await input());
    # 获取回合数量
    print("NUMBER OF ROUNDS");
    r9 = parseInt(await input());
    # 计算总可能性
    p = Math.pow(c9, p9);
    print("TOTAL POSSIBILITIES = " + p + "\n");
    # 初始化变量
    h = 0;
    c = 0;
    qa = [];
    sa = [];
    ss = [];
    as = [];
    gs = [];
    ia = [];
    hs = [];
    ls = "BWRGOYPT";
    # 打印颜色和对应字母
    print("\n");
    print("\n");
    print("COLOR    LETTER\n");
    print("=====    ======\n");
    for (x = 1; x <= c9; x++) {
        str = color[x - 1];
        while (str.length < 13)
            str += " ";
        str += ls.substr(x - 1, 1);
        print(str + "\n");
    }
    print("\n");
    }
    # 打印游戏结束信息
    print("GAME OVER\n");
    # 打印最终得分
    print("FINAL SCORE:\n");
    # 显示得分
    show_points();
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```