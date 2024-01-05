# `d:/src/tocomm/basic-computer-games\46_Hexapawn\javascript\hexapawn.js`

```
// HEXAPAWN
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

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

                       // 在输出元素中打印提示符
                       print("? ");

                       // 设置输入元素的类型为文本
                       input_element.setAttribute("type", "text");
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

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var ba = [,  # 创建一个二维数组
          [,-1,-1,-1,1,0,0,0,1,1],  # 第一个子数组
          [,-1,-1,-1,0,1,0,1,0,1],  # 第二个子数组
          [,-1,0,-1,-1,1,0,0,0,1],  # 第三个子数组
          [,0,-1,-1,1,-1,0,0,0,1],  # 第四个子数组
          [,-1,0,-1,1,1,0,0,1,0],  # 第五个子数组
          [,-1,-1,0,1,0,1,0,0,1],  # 第六个子数组
          [,0,-1,-1,0,-1,1,1,0,0],  # 第七个子数组
          [,0,-1,-1,-1,1,1,1,0,0],  # 第八个子数组
          [,-1,0,-1,-1,0,1,0,1,0],  # 第九个子数组
          [,0,-1,-1,0,1,0,0,0,1],  # 第十个子数组
          [,0,-1,-1,0,1,0,1,0,0],  # 第十一个子数组
          [,-1,0,-1,1,0,0,0,0,1],  # 第十二个子数组
          [,0,0,-1,-1,-1,1,0,0,0],  # 第十三个子数组
          [,-1,0,0,1,1,1,0,0,0],  # 第十四个子数组
          [,0,-1,0,-1,1,1,0,0,0],  # 第十五个子数组
# 创建一个二维数组，表示一个迷宫的布局，其中-1表示墙壁，0表示通道
var maze = [
          [-1,0,0,-1,-1,1,0,0,0,0],
          [0,0,-1,-1,1,0,0,0,0,0],
          [0,-1,0,1,-1,0,0,0,0,0],
          [-1,0,0,-1,1,0,0,0,0,0]
];

# 创建一个二维数组，表示迷宫中每个位置的数字代表的含义
var ma = [
          [0,24,25,36,0],
          [0,14,15,36,0],
          [0,15,35,36,47],
          [0,36,58,59,0],
          [0,15,35,36,0],
          [0,24,25,26,0],
          [0,26,57,58,0],
          [0,26,35,0,0],
          [0,47,48,0,0],
          [0,35,36,0,0],
          [0,35,36,0,0],
          [0,36,0,0,0],
          [0,47,58,0,0],
          [0,15,0,0,0],
          [0,26,47,0,0]
];
# 定义一个二维数组，表示棋盘的状态，每个元素代表一个格子的状态
var board = [
    [,47,58,0,0],
    [,35,36,47,0],
    [,28,58,0,0],
    [,15,47,0,0]
];
# 定义两个空数组，用于存储玩家的走棋记录
var s = [];
var t = [];
# 定义一个字符串，表示玩家的棋子类型
var ps = "X.O";

# 定义一个函数，用于展示当前棋盘的状态
function show_board()
{
    # 打印换行符
    print("\n");
    # 遍历棋盘的行
    for (var i = 1; i <= 3; i++) {
        # 打印制表符
        print(tab(10));
        # 遍历棋盘的列
        for (var j = 1; j <= 3; j++) {
            # 根据玩家的走棋记录和棋子类型，打印对应的棋子
            print(ps[s[(i - 1) * 3 + j] + 1]);
        }
        # 打印换行符
        print("\n");
    }
}
# 定义一个名为mirror的函数，接受一个参数x
function mirror(x)
{
    # 如果x等于1，返回3
    if (x == 1)
        return 3;
    # 如果x等于3，返回1
    if (x == 3)
        return 1;
    # 如果x等于6，返回4
    if (x == 6)
        return 4;
    # 如果x等于4，返回6
    if (x == 4)
        return 6;
    # 如果x等于9，返回7
    if (x == 9)
        return 7;
    # 如果x等于7，返回9
    if (x == 7)
        return 9;
    # 如果以上条件都不满足，返回x
    return x;
}

# 异步函数main，作为程序的入口
async function main()
{
    # 打印字符串 "HEXAPAWN"，并在前面加上 32 个空格
    print(tab(32) + "HEXAPAWN\n");
    # 打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面加上 15 个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 打印游戏说明
    # HEXAPAWN:  INTERPRETATION OF HEXAPAWN GAME AS PRESENTED IN
    # MARTIN GARDNER'S "THE UNEXPECTED HANGING AND OTHER MATHEMATIC-
    # AL DIVERSIONS", CHAPTER EIGHT:  A MATCHBOX GAME-LEARNING MACHINE
    # ORIGINAL VERSION FOR H-P TIMESHARE SYSTEM BY R.A. KAAPKE 5/5/76
    # INSTRUCTIONS BY JEFF DALTON
    # CONVERSION TO MITS BASIC BY STEVE NORTH
    # 初始化数组 s 的值为 0
    for (i = 0; i <= 9; i++) {
        s[i] = 0;
    }
    # 初始化变量 w 和 l 的值为 0
    w = 0;
    l = 0;
    # 循环，直到用户输入 "Y" 或 "N"
    do {
        print("INSTRUCTIONS (Y-N)");
        # 获取用户输入的字符串
        str = await input();
        # 截取字符串的第一个字符
        str = str.substr(0, 1);
    } while (str != "Y" && str != "N") ;
    # 循环直到用户输入为Y或N
    if (str == "Y") {
        # 如果用户输入为Y，则打印游戏规则和说明
        print("\n");
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
    }
```
这段代码看起来是C语言的代码，不是Python。因此，我将其转换为Python注释的形式。
        print("GAME BUT MUST LEARN THIS BY PLAYING.\n");  # 打印游戏提示信息
        print("\n");  # 打印空行
        print("THE NUMBERING OF THE BOARD IS AS FOLLOWS:\n");  # 打印游戏棋盘编号提示
        print(tab(10) + "123\n");  # 打印棋盘第一行编号
        print(tab(10) + "456\n");  # 打印棋盘第二行编号
        print(tab(10) + "789\n");  # 打印棋盘第三行编号
        print("\n");  # 打印空行
        print("FOR EXAMPLE, TO MOVE YOUR RIGHTMOST PAWN FORWARD,\n");  # 打印游戏操作提示
        print("YOU WOULD TYPE 9,6 IN RESPONSE TO THE QUESTION\n");  # 打印游戏操作示例
        print("'YOUR MOVE ?'.  SINCE I'M A GOOD SPORT, YOU'LL ALWAYS\n");  # 打印游戏操作提示
        print("GO FIRST.\n");  # 打印游戏操作提示
        print("\n");  # 打印空行
    }
    while (1) {  # 进入游戏循环
        x = 0;  # 初始化 x 变量
        y = 0;  # 初始化 y 变量
        s[4] = 0;  # 初始化 s 数组的第四个元素
        s[5] = 0;  # 初始化 s 数组的第五个元素
        s[6] = 0;  # 初始化 s 数组的第六个元素
        s[1] = -1;  # 初始化 s 数组的第一个元素
        s[2] = -1;  // 将数组 s 的第 2 个元素设置为 -1
        s[3] = -1;  // 将数组 s 的第 3 个元素设置为 -1
        s[7] = 1;   // 将数组 s 的第 7 个元素设置为 1
        s[8] = 1;   // 将数组 s 的第 8 个元素设置为 1
        s[9] = 1;   // 将数组 s 的第 9 个元素设置为 1
        show_board();  // 调用函数显示游戏棋盘
        while (1) {  // 进入无限循环
            while (1) {  // 进入内部无限循环
                print("YOUR MOVE");  // 打印提示信息，等待玩家输入
                str = await input();  // 等待玩家输入并将结果存储在变量 str 中
                m1 = parseInt(str);  // 将输入的字符串转换为整数并存储在 m1 中
                m2 = parseInt(str.substr(str.indexOf(",") + 1));  // 从输入的字符串中提取逗号后的部分并转换为整数存储在 m2 中
                if (m1 > 0 && m1 < 10 && m2 > 0 && m2 < 10) {  // 检查输入的坐标是否合法
                    if (s[m1] != 1 || s[m2] == 1 || (m2 - m1 != -3 && s[m2] != -1) || (m2 > m1) || (m2 - m1 == -3 && s[m2] != 0) || (m2 - m1 < -4) || (m1 == 7 && m2 == 3))
                        print("ILLEGAL MOVE.\n");  // 如果移动不合法则打印提示信息
                    else
                        break;  // 如果移动合法则跳出内部循环
                } else {
                    print("ILLEGAL CO-ORDINATES.\n");  // 如果坐标不合法则打印提示信息
                }
            }

            // 移动玩家的棋子
            s[m1] = 0;
            s[m2] = 1;
            show_board();  // 显示游戏棋盘

            // 寻找计算机的棋子
            for (i = 1; i <= 9; i++) {
                if (s[i] == -1)
                    break;
            }
            // 如果没有棋子或者玩家到达顶部则结束游戏
            if (i > 9 || s[1] == 1 || s[2] == 1 || s[3] == 1) {
                computer = false;
                break;
            }
            // 寻找有有效移动的计算机棋子
            for (i = 1; i <= 9; i++) {
                if (s[i] != -1)
                    continue;  # 跳过当前循环，继续下一次循环
                if (s[i + 3] == 0  # 如果数组s中索引为i+3的元素为0
                 || (mirror(i) == i && (s[i + 2] == 1 || s[i + 4] == 1))  # 或者mirror(i)等于i且s[i+2]为1或者s[i+4]为1
                 || (i <= 3 && s[5] == 1)  # 或者i小于等于3且s[5]为1
                 || s[8] == 1)  # 或者s[8]为1
                    break;  # 跳出当前循环
            }
            if (i > 9) {  // 如果i大于9
                computer = false;  # 将computer设置为false
                break;  # 跳出当前循环
            }
            for (i = 1; i <= 19; i++) {  # 循环i从1到19
                for (j = 1; j <= 3; j++) {  # 循环j从1到3
                    for (k = 3; k >= 1; k--) {  # 循环k从3到1
                        t[(j - 1) * 3 + k] = ba[i][(j - 1) * 3 + 4 - k];  # 计算并赋值给t数组
                    }
                }
                for (j = 1; j <= 9; j++) {  # 循环j从1到9
                    if (s[j] != ba[i][j])  # 如果数组s中索引为j的元素不等于数组ba中索引为i的元素
                        break;  # 跳出当前循环
                }
                # 如果 j 大于 9，则将 r 设为 0 并跳出循环
                if (j > 9) {
                    r = 0;
                    break;
                }
                # 遍历 j 从 1 到 9，如果 s[j] 不等于 t[j]，则跳出循环
                for (j = 1; j <= 9; j++) {
                    if (s[j] != t[j])
                        break;
                }
                # 如果 j 大于 9，则将 r 设为 1 并跳出循环
                if (j > 9) {
                    r = 1;
                    break;
                }
            }
            # 如果 i 大于 19，则打印"ILLEGAL BOARD PATTERN"并跳出循环
            if (i > 19) {
                print("ILLEGAL BOARD PATTERN\n");
                break;
            }
            # 将 x 设为 i
            x = i;
            # 遍历 i 从 1 到 4
            for (i = 1; i <= 4; i++) {
            // 如果 ma[x][i] 不等于 0，则跳出循环
            if (ma[x][i] != 0)
                break;
        }
        // 如果 i 大于 4，则打印 "I RESIGN."，将 computer 设为 false，并跳出循环
        if (i > 4) {
            print("I RESIGN.\n");
            computer = false;
            break;
        }
        // 从可能的移动中随机选择一个移动
        do {
            y = Math.floor(Math.random() * 4 + 1);
        } while (ma[x][y] == 0) ;
        // 宣布移动
        if (r == 0) {
            print("I MOVE FROM " + Math.floor(ma[x][y] / 10) + " TO " + ma[x][y] % 10 + "\n");
            s[Math.floor(ma[x][y] / 10)] = 0;
            s[ma[x][y] % 10] = -1;
        } else {
            print("I MOVE FROM " + mirror(Math.floor(ma[x][y] / 10)) + " TO " + mirror(ma[x][y]) % 10 + "\n");
            s[mirror(Math.floor(ma[x][y] / 10))] = 0;
                s[mirror(ma[x][y] % 10)] = -1; // 将棋盘上对应位置的值设为-1，表示电脑占据该位置
            }
            show_board(); // 展示更新后的棋盘
            // 如果电脑到达底部，则游戏结束
            if (s[7] == -1 || s[8] == -1 || s[9] == -1) {
                computer = true; // 将电脑胜利标志设为true
                break; // 结束游戏循环
            }
            // 如果没有玩家的棋子了，则游戏结束
            for (i = 1; i <= 9; i++) {
                if (s[i] == 1)
                    break;
            }
            if (i > 9) {
                computer = true; // 将电脑胜利标志设为true
                break; // 结束游戏循环
            }
            // 如果玩家无法移动，则游戏结束
            for (i = 1; i <= 9; i++) {
                if (s[i] != 1) // 如果棋盘上的位置不是玩家的棋子
# 继续执行循环
                    continue;
                # 如果前三个位置的值为0，则跳出循环
                if (s[i - 3] == 0)
                    break;
                # 如果当前位置的镜像位置不等于当前位置，则执行以下操作
                if (mirror(i) != i) {
                    # 如果当前位置大于等于7，则执行以下操作
                    if (i >= 7) {
                        # 如果第5个位置的值为-1，则跳出循环
                        if (s[5] == -1)
                            break;
                    } else {
                        # 如果第2个位置的值为-1，则跳出循环
                        if (s[2] == -1)
                            break;
                    }
                } else {
                    # 如果当前位置的前两个位置或者前四个位置的值为-1，则跳出循环
                    if (s[i - 2] == -1 || s[i - 4] == -1)
                        break;
                }

            }
            # 如果当前位置大于9，则打印提示信息并将computer设置为true
            if (i > 9) {
                print("YOU CAN'T MOVE, SO ");
                computer = true;
                break;  # 结束当前循环，跳出循环体
            }
        }
        if (computer) {  # 如果computer为真
            print("I WIN.\n");  # 打印"I WIN."
            w++;  # w加1
        } else {
            print("YOU WIN\n");  # 打印"YOU WIN"
            ma[x][y] = 0;  # 将ma[x][y]赋值为0
            l++;  # l加1
        }
        print("I HAVE WON " + w + " AND YOU " + l + " OUT OF " + (l + w) + " GAMES.\n");  # 打印"I HAVE WON " + w + " AND YOU " + l + " OUT OF " + (l + w) + " GAMES."
        print("\n");  # 打印换行
    }
}

main();  # 调用main函数
```