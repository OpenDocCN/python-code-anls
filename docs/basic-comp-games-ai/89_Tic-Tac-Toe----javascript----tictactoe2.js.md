# `basic-computer-games\89_Tic-Tac-Toe\javascript\tictactoe2.js`

```
// 定义一个打印函数，将字符串添加到输出元素中
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
                       input_str = undefined;
                       // 监听输入元素的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个空数组
var s = [];

// 定义一个判断胜负的函数
function who_win(piece)
{
    if (piece == -1) {
        // 如果 piece 为 -1，打印“我赢了，笨蛋!!!”
        print("I WIN, TURKEY!!!\n");
    } else if (piece == 1) {
        // 如果 piece 为 1，打印“你赢了!! 好游戏。”
        print("YOU BEAT ME!! GOOD GAME.\n");
    }
}

// 定义一个显示棋盘的函数
function show_board()
{
    // 打印换行符
    print("\n");
    # 遍历九宫格的每个位置
    for (i = 1; i <= 9; i++) {
        # 打印空格
        print(" ");
        # 如果当前位置为空
        if (s[i] == -1) {
            # 打印当前位置的标记
            print(qs + " ");
        } 
        # 如果当前位置为零
        else if (s[i] == 0) {
            # 打印两个空格
            print("  ");
        } 
        # 如果当前位置为另一种标记
        else {
            # 打印当前位置的标记
            print(ps + " ");
        }
        # 如果当前位置是第3或第6个位置
        if (i == 3 || i == 6) {
            # 打印换行和九宫格的分隔线
            print("\n");
            print("---+---+---\n");
        } 
        # 如果当前位置不是第9个位置
        else if (i != 9) {
            # 打印感叹号
            print("!");
        }
    }
    # 打印多个换行
    print("\n");
    print("\n");
    print("\n");
    # 检查是否有玩家获胜
    for (i = 1; i <= 7; i += 3) {
        if (s[i] && s[i] == s[i + 1] && s[i] == s[i + 2]) {
            # 调用函数宣布获胜者，并返回true
            who_win(s[i]);
            return true;
        }
    }
    for (i = 1; i <= 3; i++) {
        if (s[i] && s[i] == s[i + 3] && s[i] == s[i + 6]) {
            # 调用函数宣布获胜者，并返回true
            who_win(s[i]);
            return true;
        }
    }
    if (s[1] && s[1] == s[5] && s[1] == s[9]) {
        # 调用函数宣布获胜者，并返回true
        who_win(s[1]);
        return true;
    }
    if (s[3] && s[3] == s[5] && s[3] == s[7]) {
        # 调用函数宣布获胜者，并返回true
        who_win(s[3]);
        return true;
    }
    # 检查是否为平局
    for (i = 1; i <= 9; i++) {
        if (s[i] == 0)
            break;
    }
    # 如果所有位置都被占满
    if (i > 9) {
        # 打印平局信息，并返回true
        print("IT'S A DRAW. THANK YOU.\n");
        return true;
    }
    # 返回false
    return false;
// 结束了前面的函数定义，接下来是主控制部分的代码

// 异步函数，程序的入口
async function main()
{
    // 打印游戏标题
    print(tab(30) + "TIC-TAC-TOE\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 初始化游戏棋盘
    for (i = 1; i <= 9; i++)
        s[i] = 0;
    // 打印棋盘编号
    print("THE BOARD IS NUMBERED:\n");
    print(" 1  2  3\n");
    print(" 4  5  6\n");
    print(" 7  8  9\n");
    print("\n");
    print("\n");
    print("\n");
    // 询问玩家选择 'X' 还是 'O'
    print("DO YOU WANT 'X' OR 'O'");
    // 等待用户输入
    str = await input();
    // 根据用户选择设置玩家和电脑的角色，并记录是否第一次选择
    if (str == "X") {
        ps = "X";
        qs = "O";
        first_time = true;
    } else {
        ps = "O";
        qs = "X";
        first_time = false;
    }
    // 结束主函数
}
// 调用主函数
main();
```