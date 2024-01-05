# `d:/src/tocomm/basic-computer-games\89_Tic-Tac-Toe\javascript\tictactoe1.js`

```
// 定义一个名为print的函数，用于向页面输出文本
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 在页面输出问号作为提示
                       print("? ");
                       // 设置input元素的类型为文本
                       input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下的键为回车键时，获取输入字符串，移除输入元素，打印输入字符串并解析输入字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器
});
}

# 定义一个函数 tab，用于生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环减少 space 并在 str 中添加一个空格
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

function mf(x)  # 定义函数mf，参数为x
{
    return x - 8 * Math.floor((x - 1) / 8);  # 返回x减去8乘以向下取整后的值
}

function computer_moves()  # 定义函数computer_moves
{
    print("COMPUTER MOVES " + m + "\n");  # 打印"COMPUTER MOVES "和变量m的值，换行
}

var m;  # 声明变量m

// Main control section
async function main()  # 定义异步函数main
{
    print(tab(30) + "TIC TAC TOE\n");  # 打印30个空格和"TIC TAC TOE"，换行
    # 打印欢迎信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    #
    # 这个程序玩井字游戏
    # 机器先走
    # 打印游戏板的编号
    print("THE GAME BOARD IS NUMBERED:\n");
    print("\n");
    print("1  2  3\n");
    print("8  9  4\n");
    print("7  6  5\n");
    print("\n");
    #
    # 主程序
    while (1) {
        # 打印空行
        print("\n");
        print("\n");
        # 初始化变量a为9
        a = 9;
        # 将a的值赋给m
        m = a;
        computer_moves();  # 调用函数，让计算机进行移动
        print("YOUR MOVE");  # 打印提示信息，要求玩家输入移动
        m = parseInt(await input());  # 将玩家输入的值转换为整数并赋给变量m

        p = m;  # 将m的值赋给变量p
        b = mf(p + 1);  # 调用函数mf，将p加1的结果赋给变量b
        m = b;  # 将b的值赋给变量m

        computer_moves();  # 调用函数，让计算机进行移动
        print("YOUR MOVE");  # 打印提示信息，要求玩家输入移动
        m = parseInt(await input());  # 将玩家输入的值转换为整数并赋给变量m

        q = m;  # 将m的值赋给变量q
        if (q != mf(b + 4)) {  # 如果q不等于调用函数mf，将b加4的结果
            c = mf(b + 4);  # 调用函数mf，将b加4的结果赋给变量c
            m = c;  # 将c的值赋给变量m
            computer_moves();  # 调用函数，让计算机进行移动
            print("AND WINS ********\n");  # 打印提示信息，表示计算机获胜
            continue;  # 继续循环
        }

        c = mf(b + 2);  # 将变量b加2后的值赋给变量c
        m = c;  # 将变量c的值赋给变量m

        computer_moves();  # 调用计算机移动的函数
        print("YOUR MOVE");  # 打印提示信息
        m = parseInt(await input());  # 从输入中获取用户的移动，并将其转换为整数赋给变量m

        r = m;  # 将变量m的值赋给变量r
        if (r != mf(c + 4)) {  # 如果r不等于变量c加4后的值
            d = mf(c + 4);  # 将变量c加4后的值赋给变量d
            m = d;  # 将变量d的值赋给变量m
            computer_moves();  # 调用计算机移动的函数
            print("AND WINS ********\n");  # 打印提示信息
            continue;  # 继续循环
        }

        if (p % 2 == 0) {  # 如果p除以2的余数等于0
            d = mf(c + 7);  # 将变量c加7后的值赋给变量d
            m = d;  # 将变量d的值赋给变量m
            computer_moves();  # 调用一个名为computer_moves的函数
            print("AND WINS ********\n");  # 打印字符串"AND WINS ********\n"
            continue;  # 继续执行下一次循环

        }

        d = mf(c + 3);  # 将mf(c + 3)的返回值赋给变量d
        m = d;  # 将变量d的值赋给变量m

        computer_moves();  # 调用一个名为computer_moves的函数
        print("YOUR MOVE");  # 打印字符串"YOUR MOVE"
        m = parseInt(await input());  # 将用户输入的值转换为整数并赋给变量m

        s = m;  # 将变量m的值赋给变量s
        if (s != mf(d + 4)) {  # 如果变量s的值不等于mf(d + 4)的返回值
            e = mf(d + 4);  # 将mf(d + 4)的返回值赋给变量e
            m = e;  # 将变量e的值赋给变量m
            computer_moves();  # 调用一个名为computer_moves的函数
        }
        e = mf(d + 6);  # 将mf(d + 6)的返回值赋给变量e
        m = e;  # 将变量 m 的值设置为变量 e 的值
        computer_moves();  # 调用名为 computer_moves 的函数
        print("THE GAME IS A DRAW.\n");  # 打印文本 "THE GAME IS A DRAW."
    }
}

main();  # 调用名为 main 的函数
```