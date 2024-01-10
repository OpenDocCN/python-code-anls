# `basic-computer-games\46_Hexapawn\javascript\hexapawn.js`

```
// HEXAPAWN
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

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
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
                       // 设置输入框长度
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
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

// 定义一个制表符函数，返回指定数量空格的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}
// 定义二维数组 ba，表示棋盘上每个位置的状态，-1表示黑子，0表示空，1表示白子
var ba = [,
          [,-1,-1,-1,1,0,0,0,1,1],
          [,-1,-1,-1,0,1,0,1,0,1],
          [,-1,0,-1,-1,1,0,0,0,1],
          [,0,-1,-1,1,-1,0,0,0,1],
          [,-1,0,-1,1,1,0,0,1,0],
          [,-1,-1,0,1,0,1,0,0,1],
          [,0,-1,-1,0,-1,1,1,0,0],
          [,0,-1,-1,-1,1,1,1,0,0],
          [,-1,0,-1,-1,0,1,0,1,0],
          [,0,-1,-1,0,1,0,0,0,1],
          [,0,-1,-1,0,1,0,1,0,0],
          [,-1,0,-1,1,0,0,0,0,1],
          [,0,0,-1,-1,-1,1,0,0,0],
          [,-1,0,0,1,1,1,0,0,0],
          [,0,-1,0,-1,1,1,0,0,0],
          [,-1,0,0,-1,-1,1,0,0,0],
          [,0,0,-1,-1,1,0,0,0,0],
          [,0,-1,0,1,-1,0,0,0,0],
          [,-1,0,0,-1,1,0,0,0,0]];
// 定义二维数组 ma，表示棋盘上每个位置的权重
var ma = [,
          [,24,25,36,0],
          [,14,15,36,0],
          [,15,35,36,47],
          [,36,58,59,0],
          [,15,35,36,0],
          [,24,25,26,0],
          [,26,57,58,0],
          [,26,35,0,0],
          [,47,48,0,0],
          [,35,36,0,0],
          [,35,36,0,0],
          [,36,0,0,0],
          [,47,58,0,0],
          [,15,0,0,0],
          [,26,47,0,0],
          [,47,58,0,0],
          [,35,36,47,0],
          [,28,58,0,0],
          [,15,47,0,0]];
// 定义空数组 s 和 t
var s = [];
var t = [];
// 定义字符串 ps
var ps = "X.O";

// 定义函数 show_board，用于展示棋盘
function show_board()
{
    // 打印换行符
    print("\n");
    // 遍历棋盘的每一行
    for (var i = 1; i <= 3; i++) {
        // 打印空格
        print(tab(10));
        // 遍历棋盘的每一列
        for (var j = 1; j <= 3; j++) {
            // 根据棋盘状态打印对应的棋子
            print(ps[s[(i - 1) * 3 + j] + 1]);
        }
        // 打印换行符
        print("\n");
    }
}

// 定义函数 mirror，用于返回棋子的镜像位置
function mirror(x)
{
    // 根据输入的位置返回对应的镜像位置
    if (x == 1)
        return 3;
    if (x == 3)
        return 1;
    if (x == 6)
        return 4;
    if (x == 4)
        return 6;
    if (x == 9)
        return 7;
    if (x == 7)
        return 9;
    return x;
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(32) + "HEXAPAWN\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏介绍
    // HEXAPAWN:  INTERPRETATION OF HEXAPAWN GAME AS PRESENTED IN
    // MARTIN GARDNER'S "THE UNEXPECTED HANGING AND OTHER MATHEMATIC-
    // 初始化数组 s，将每个元素都设为 0
    for (i = 0; i <= 9; i++) {
        s[i] = 0;
    }
    // 初始化变量 w 和 l，分别表示赢的次数和输的次数
    w = 0;
    l = 0;
    // 循环直到用户输入 Y 或 N
    do {
        // 打印提示信息，要求用户输入 Y 或 N
        print("INSTRUCTIONS (Y-N)");
        // 等待用户输入，并截取第一个字符
        str = await input();
        str = str.substr(0, 1);
    } while (str != "Y" && str != "N") ;
    # 如果输入的字符串为 "Y"，则执行以下操作
    if (str == "Y") {
        # 打印空行
        print("\n");
        # 打印游戏规则说明
        print("THIS PROGRAM PLAYS THE GAME OF HEXAPAWN.\n");
        print("HEXAPAWN IS PLAYED WITH CHESS PAWNS ON A 3 BY 3 BOARD.\n");
        print("THE PAWNS ARE MOVED AS IN CHESS - ONE SPACE FORWARD TO\n");
        print("AN EMPTY SPACE OR ONE SPACE FORWARD AND DIAGONALLY TO\n");
        print("CAPTURE AN OPPOSING MAN.  ON THE BOARD, YOUR PAWNS\n");
        print("ARE 'O', THE COMPUTER'S PAWNS ARE 'X', AND EMPTY \n");
        print("SQUARES ARE '.'.  TO ENTER A MOVE, TYPE THE NUMBER OF\n");
        print("THE SQUARE YOU ARE MOVING FROM, FOLLOWED BY THE NUMBER\n");
        print("OF THE SQUARE YOU WILL MOVE TO.  THE NUMBERS MUST BE\n");
        print("SEPERATED BY A COMMA.\n");
        print("\n");
        print("THE COMPUTER STARTS A SERIES OF GAMES KNOWING ONLY WHEN\n");
        print("THE GAME IS WON (A DRAW IS IMPOSSIBLE) AND HOW TO MOVE.\n");
        print("IT HAS NO STRATEGY AT FIRST AND JUST MOVES RANDOMLY.\n");
        print("HOWEVER, IT LEARNS FROM EACH GAME.  THUS, WINNING BECOMES\n");
        print("MORE AND MORE DIFFICULT.  ALSO, TO HELP OFFSET YOUR\n");
        print("INITIAL ADVANTAGE, YOU WILL NOT BE TOLD HOW TO WIN THE\n");
        print("GAME BUT MUST LEARN THIS BY PLAYING.\n");
        print("\n");
        print("THE NUMBERING OF THE BOARD IS AS FOLLOWS:\n");
        print(tab(10) + "123\n");
        print(tab(10) + "456\n");
        print(tab(10) + "789\n");
        print("\n");
        print("FOR EXAMPLE, TO MOVE YOUR RIGHTMOST PAWN FORWARD,\n");
        print("YOU WOULD TYPE 9,6 IN RESPONSE TO THE QUESTION\n");
        print("'YOUR MOVE ?'.  SINCE I'M A GOOD SPORT, YOU'LL ALWAYS\n");
        print("GO FIRST.\n");
        print("\n");
    }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```