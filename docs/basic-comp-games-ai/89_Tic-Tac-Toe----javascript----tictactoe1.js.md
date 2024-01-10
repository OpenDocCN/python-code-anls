# `basic-computer-games\89_Tic-Tac-Toe\javascript\tictactoe1.js`

```
// TIC TAC TOE 1
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串输出到指定元素
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
                       // 设置输入框长度
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       // 初始化输入字符串
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
                                                      // 输出换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义取模函数，返回 x 对 8 取模的结果
function mf(x)
{
    return x - 8 * Math.floor((x - 1) / 8);
}

// 定义计算机移动函数
function computer_moves()
{
    print("COMPUTER MOVES " + m + "\n");
}

var m;

// 主控制部分，使用 async 函数定义
async function main()
{
    // 输出标题
    print(tab(30) + "TIC TAC TOE\n");
    // 输出副标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 输出多个换行符
    print("\n");
    print("\n");
    print("\n");
    //
    // This program plays Tic Tac Toe
    // The machine goes first
    # 打印游戏板的编号
    print("THE GAME BOARD IS NUMBERED:\n");
    print("\n");
    print("1  2  3\n");
    print("8  9  4\n");
    print("7  6  5\n");
    print("\n");
    #
    # 主程序
    while (1):
        # 打印空行
        print("\n");
        print("\n");
        # 初始化变量a和m
        a = 9;
        m = a;

        # 计算机移动
        computer_moves();
        # 打印提示信息
        print("YOUR MOVE");
        # 获取用户输入并转换为整数
        m = parseInt(await input());

        # 保存用户输入
        p = m;
        # 计算下一个位置
        b = mf(p + 1);
        m = b;

        # 计算机移动
        computer_moves();
        # 打印提示信息
        print("YOUR MOVE");
        # 获取用户输入并转换为整数
        m = parseInt(await input());

        # 保存用户输入
        q = m;
        # 判断用户输入是否等于下一个位置的值
        if (q != mf(b + 4)):
            # 计算下一个位置
            c = mf(b + 4);
            m = c;
            # 计算机移动
            computer_moves();
            # 打印提示信息
            print("AND WINS ********\n");
            # 继续循环
            continue;

        # 计算下一个位置
        c = mf(b + 2);
        m = c;

        # 计算机移动
        computer_moves();
        # 打印提示信息
        print("YOUR MOVE");
        # 获取用户输入并转换为整数
        m = parseInt(await input());

        # 保存用户输入
        r = m;
        # 判断用户输入是否等于下一个位置的值
        if (r != mf(c + 4)):
            # 计算下一个位置
            d = mf(c + 4);
            m = d;
            # 计算机移动
            computer_moves();
            # 打印提示信息
            print("AND WINS ********\n");
            # 继续循环
            continue;

        # 判断p是否为偶数
        if (p % 2 == 0):
            # 计算下一个位置
            d = mf(c + 7);
            m = d;
            # 计算机移动
            computer_moves();
            # 打印提示信息
            print("AND WINS ********\n");
            # 继续循环
            continue;

        # 计算下一个位置
        d = mf(c + 3);
        m = d;

        # 计算机移动
        computer_moves();
        # 打印提示信息
        print("YOUR MOVE");
        # 获取用户输入并转换为整数
        m = parseInt(await input());

        # 保存用户输入
        s = m;
        # 判断用户输入是否等于下一个位置的值
        if (s != mf(d + 4)):
            # 计算下一个位置
            e = mf(d + 4);
            m = e;
            # 计算机移动
            computer_moves();
        # 计算下一个位置
        e = mf(d + 6);
        m = e;
        # 计算机移动
        computer_moves();
        # 打印提示信息
        print("THE GAME IS A DRAW.\n");
# 调用名为main的函数
main();
```