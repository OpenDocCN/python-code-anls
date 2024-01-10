# `basic-computer-games\48_High_IQ\javascript\highiq.js`

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
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
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

// 定义变量 b、t、m、z、p
var b = [];
var t = [];
var m = [,13,14,15,
          22,23,24,
    29,30,31,32,33,34,35,
    38,39,40,41,42,43,44,
    47,48,49,50,51,52,53,
          58,59,60,
          67,68,69];

// 定义打印棋盘的函数
function print_board()
{
    # 循环遍历 x 变量从 1 到 9
    for (x = 1; x <= 9; x++) {
        # 初始化空字符串
        str = "";
        # 循环遍历 y 变量从 1 到 9
        for (y = 1; y <= 9; y++) {
            # 如果 x 或 y 为边界值，则跳过当前循环
            if (x == 1 || x == 9 || y == 1 || y == 9)
                continue;
            # 如果 x 或 y 为中心区域的边界值
            if (x == 4 || x == 5 || x == 6 || y == 4 || y == 5 || y == 6) {
                # 当字符串长度小于 y 的两倍时，往字符串中添加空格
                while (str.length < y * 2)
                    str += " ";
                # 如果 t[x][y] 的值为 5，则在字符串后添加感叹号，否则添加大写字母 O
                if (t[x][y] == 5)
                    str += "!";
                else
                    str += "O";
            }
        }
        # 打印字符串并换行
        print(str + "\n");
    }
}

//
// Update board
//
function update_board()
{
    c = 1;  // 初始化计数器 c
    for (var x = 1; x <= 9; x++) {  // 循环遍历 x 坐标
        for (var y = 1; y <= 9; y++, c++) {  // 循环遍历 y 坐标，每次循环计数器 c 自增
            if (c != z)  // 如果计数器 c 不等于 z，则继续下一次循环
                continue;
            if (c + 2 == p) {  // 如果计数器 c 加 2 等于 p
                if (t[x][y + 1] == 0)  // 如果 t[x][y + 1] 等于 0
                    return false;  // 返回 false
                t[x][y + 2] = 5;  // 将 t[x][y + 2] 设置为 5
                t[x][y + 1] = 0;  // 将 t[x][y + 1] 设置为 0
                b[c + 1] = -3;  // 将 b[c + 1] 设置为 -3
            } else if (c + 18 == p) {  // 如果计数器 c 加 18 等于 p
                if (t[x + 1][y] == 0)  // 如果 t[x + 1][y] 等于 0
                    return false;  // 返回 false
                t[x + 2][y] = 5;  // 将 t[x + 2][y] 设置为 5
                t[x + 1][y] = 0;  // 将 t[x + 1][y] 设置为 0
                b[c + 9] = -3;  // 将 b[c + 9] 设置为 -3
            } else if (c - 2 == p) {  // 如果计数器 c 减 2 等于 p
                if (t[x][y - 1] == 0)  // 如果 t[x][y - 1] 等于 0
                    return false;  // 返回 false
                t[x][y - 2] = 5;  // 将 t[x][y - 2] 设置为 5
                t[x][y - 1] = 0;  // 将 t[x][y - 1] 设置为 0
                b[c - 1] = -3;  // 将 b[c - 1] 设置为 -3
            } else if (c - 18 == p) {  // 如果计数器 c 减 18 等于 p
                if (t[x - 1][y] == 0)  // 如果 t[x - 1][y] 等于 0
                    return false;  // 返回 false
                t[x - 2][y] = 5;  // 将 t[x - 2][y] 设置为 5
                t[x - 1][y] = 0;  // 将 t[x - 1][y] 设置为 0
                b[c - 9] = -3;  // 将 b[c - 9] 设置为 -3
            } else {
                continue;  // 否则继续下一次循环
            }
            b[z] = -3;  // 将 b[z] 设置为 -3
            b[p] = -7;  // 将 b[p] 设置为 -7
            t[x][y] = 0;  // 将 t[x][y] 设置为 0
            return true;  // 返回 true
        }
    }
}

//
// Check for game over
//
// Rewritten because original subroutine was buggy
//
function check_game_over()
{
    f = 0;  // 初始化计数器 f
    for (r = 2; r <= 8; r++) {  // 循环遍历 r
        for (c = 2; c <= 8; c++) {  // 循环遍历 c
            if (t[r][c] != 5)  // 如果 t[r][c] 不等于 5
                continue;  // 继续下一次循环
            f++;  // 计数器 f 自增
            if (r > 3 && t[r - 1][c] == 5 && t[r - 2][c] == 0)  // 如果 r 大于 3 且 t[r - 1][c] 等于 5 且 t[r - 2][c] 等于 0
                return false;  // 返回 false
            if (c > 3 && t[r][c - 1] == 5 && t[r][c - 2] == 0)  // 如果 c 大于 3 且 t[r][c - 1] 等于 5 且 t[r][c - 2] 等于 0
                return false;  // 返回 false
            if (r < 7 && t[r + 1][c] == 5 && t[r + 2][c] == 0)  // 如果 r 小于 7 且 t[r + 1][c] 等于 5 且 t[r + 2][c] 等于 0
                return false;  // 返回 false
            if (c < 7 && t[r][c + 1] == 5 && t[r][c + 2] == 0)  // 如果 c 小于 7 且 t[r][c + 1] 等于 5 且 t[r][c + 2] 等于 0
                return false;  // 返回 false
        }
    }
    return true;  // 返回 true
}

// Main program
async function main()
{
    print(tab(33) + "H-I-Q\n");  // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印信息
    # 打印三个换行符
    print("\n");
    print("\n");
    print("\n");
    # 初始化数组 b 的值为 0
    for (r = 0; r <= 70; r++)
        b[r] = 0;
    # 打印游戏板的标题
    print("HERE IS THE BOARD:\n");
    print("\n");
    # 打印游戏板的第一行
    print("          !    !    !\n");
    print("         13   14   15\n");
    print("\n");
    # 打印游戏板的第二行
    print("          !    !    !\n");
    print("         22   23   24\n");
    print("\n");
    # 打印游戏板的第三行
    print("!    !    !    !    !    !    !\n");
    print("29   30   31   32   33   34   35\n");
    print("\n");
    # 打印游戏板的第四行
    print("!    !    !    !    !    !    !\n");
    print("38   39   40   41   42   43   44\n");
    print("\n");
    # 打印游戏板的第五行
    print("!    !    !    !    !    !    !\n");
    print("47   48   49   50   51   52   53\n");
    print("\n");
    # 打印游戏板的第六行
    print("          !    !    !\n");
    print("         58   59   60\n");
    print("\n");
    # 打印游戏板的第七行
    print("          !    !    !\n");
    print("         67   68   69\n");
    print("\n");
    # 打印游戏规则说明
    print("TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD\n");
    print("WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG\n");
    print("NUMBERS.  OK, LET'S BEGIN.\n");
    // 进入游戏循环
    while (1) {
        // 设置棋盘
        for (r = 1; r <= 9; r++) {
            // 初始化每一行的数组
            t[r] = [];
            for (c = 1; c <= 9; c++) {
                // 根据条件设置每个位置的值
                if (r == 4 || r == 5 || r == 6 || c == 4 || c == 5 || c == 6 && (r != 1 && c != 1 && r != 9 && c != 9)) {
                    t[r][c] = 5;
                } else {
                    t[r][c] = -5;
                }
            }
        }
        // 设置中心位置为0
        t[5][5] = 0;
        // 打印棋盘
        print_board();
        // 初始化次要棋盘
        for (w = 1; w <= 33; w++) {
            b[m[w]] = -7;
        }
        // 设置特定位置为-3
        b[41] = -3;
        // 输入移动并检查合法性
        do {
            while (1) {
                // 提示输入移动的棋子
                print("MOVE WHICH PIECE");
                // 解析输入的值
                z = parseInt(await input());
                // 检查移动是否合法
                if (b[z] == -7) {
                    print("TO WHERE");
                    // 解析输入的值
                    p = parseInt(await input());
                    // 检查移动是否合法
                    if (p != z
                        && b[p] != 0
                        && b[p] != -7
                        && (z + p) % 2 == 0
                        && (Math.abs(z - p) - 2) * (Math.abs(z - p) - 18) == 0
                        && update_board())
                        break;
                }
                // 提示移动不合法，重新输入
                print("ILLEGAL MOVE, TRY AGAIN...\n");
            }
            // 打印棋盘
            print_board();
        } while (!check_game_over()) ;
        // 游戏结束
        print("THE GAME IS OVER.\n");
        // 打印剩余棋子数量
        print("YOU HAD " + f + " PIECES REMAINING.\n");
        // 如果剩余棋子数量为1，打印完美得分的提示
        if (f == 1) {
            print("BRAVO!  YOU MADE A PERFECT SCORE!\n");
            print("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!\n");
        }
        // 提示是否再玩一次
        print("\n");
        print("PLAY AGAIN (YES OR NO)");
        // 解析输入的值
        str = await input();
        // 如果输入为"NO"，跳出循环
        if (str == "NO")
            break;
    }
    // 打印结束语
    print("\n");
    print("SO LONG FOR NOW.\n");
    print("\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```