# `d:/src/tocomm/basic-computer-games\48_High_IQ\javascript\highiq.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上显示提示符 "? "
// 设置 INPUT 元素的类型为文本输入
                       // 设置输入框的长度为50
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到页面中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 添加键盘按下事件监听器
                       input_element.addEventListener("keydown", function (event) {
                           // 如果按下的是回车键
                           if (event.keyCode == 13) {
                               // 获取输入框中的值
                               input_str = input_element.value;
                               // 从页面中移除输入框
                               document.getElementById("output").removeChild(input_element);
                               // 打印输入的字符串
                               print(input_str);
                               // 打印换行符
                               print("\n");
                               // 返回输入的字符串
                               resolve(input_str);
                           }
                       });
                   });
}

function tab(space)
{
    // 生成指定数量的空格字符串
    var str = "";
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

var b = [];  // 创建一个空数组 b
var t = [];  // 创建一个空数组 t
var m = [,13,14,15,  // 创建一个包含数字的数组 m
          22,23,24,
    29,30,31,32,33,34,35,
    38,39,40,41,42,43,44,
    47,48,49,50,51,52,53,
          58,59,60,
          67,68,69];
var z;  // 声明变量 z
var p;  // 声明变量 p

//
// Print board
//
function print_board()  // 定义一个名为 print_board 的函数
{
    for (x = 1; x <= 9; x++) {  // 循环遍历行数
        str = "";  // 初始化空字符串
        for (y = 1; y <= 9; y++) {  // 循环遍历列数
            if (x == 1 || x == 9 || y == 1 || y == 9)  // 如果在边界上，跳过
                continue;
            if (x == 4 || x == 5 || x == 6 || y == 4 || y == 5 || y == 6) {  // 如果在特定位置
                while (str.length < y * 2)  // 当字符串长度小于 y*2 时
                    str += " ";  // 添加空格
                if (t[x][y] == 5)  // 如果 t[x][y] 等于 5
                    str += "!";  // 添加感叹号
                else
                    str += "O";  // 否则添加大写字母 O
            }
        }
        print(str + "\n");  // 打印字符串并换行
    }
}
// 更新棋盘
function update_board() {
    c = 1; // 初始化计数器
    for (var x = 1; x <= 9; x++) { // 遍历 x 坐标
        for (var y = 1; y <= 9; y++, c++) { // 遍历 y 坐标，每次循环计数器加一
            if (c != z) // 如果计数器不等于 z，则跳过当前循环
                continue;
            if (c + 2 == p) { // 如果计数器加 2 等于 p
                if (t[x][y + 1] == 0) // 如果 t[x][y + 1] 等于 0
                    return false; // 返回 false
                t[x][y + 2] = 5; // 将 t[x][y + 2] 设置为 5
                t[x][y + 1] = 0; // 将 t[x][y + 1] 设置为 0
                b[c + 1] = -3; // 将 b[c + 1] 设置为 -3
            } else if (c + 18 == p) { // 如果计数器加 18 等于 p
                if (t[x + 1][y] == 0) // 如果 t[x + 1][y] 等于 0
                    return false; // 返回 false
                t[x + 2][y] = 5; // 将 t[x + 2][y] 设置为 5
                t[x + 1][y] = 0; // 将 t[x + 1][y] 设置为 0
            // 如果当前位置的数字为c+9，则向下移动
            b[c + 9] = -3;
        } else if (c - 2 == p) {
            // 如果当前位置的数字为c-2，则向上移动
            if (t[x][y - 1] == 0)
                return false;
            t[x][y - 2] = 5;
            t[x][y - 1] = 0;
            b[c - 1] = -3;
        } else if (c - 18 == p) {
            // 如果当前位置的数字为c-18，则向左移动
            if (t[x - 1][y] == 0)
                return false;
            t[x - 2][y] = 5;
            t[x - 1][y] = 0;
            b[c - 9] = -3;
        } else {
            // 如果以上条件都不满足，则继续循环
            continue;
        }
        // 设置新位置的数字为-3
        b[z] = -3;
        // 设置旧位置的数字为-7
        b[p] = -7;
        // 设置当前位置的数字为0
        t[x][y] = 0;
        // 返回移动成功
        return true;
// 检查游戏是否结束
// 重写了原始的子程序，因为原始的子程序存在错误
function check_game_over()
{
    f = 0;  // 初始化计数器 f
    for (r = 2; r <= 8; r++) {  // 遍历行
        for (c = 2; c <= 8; c++) {  // 遍历列
            if (t[r][c] != 5)  // 如果当前位置的值不等于5，跳过当前循环
                continue;
            f++;  // 计数器 f 自增
            if (r > 3 && t[r - 1][c] == 5 && t[r - 2][c] == 0)  // 如果满足条件，返回 false
                return false;
            if (c > 3 && t[r][c - 1] == 5 && t[r][c - 2] == 0)  // 如果满足条件，返回 false
                return false;  // 如果条件成立，返回 false
            if (r < 7 && t[r + 1][c] == 5 && t[r + 2][c] == 0)  // 如果满足条件，返回 false
                return false;  // 如果条件成立，返回 false
            if (c < 7 && t[r][c + 1] == 5 && t[r][c + 2] == 0)  // 如果满足条件，返回 false
                return false;  // 如果条件成立，返回 false
        }
    }
    return true;  // 如果以上条件都不成立，返回 true
}

// Main program
async function main()
{
    print(tab(33) + "H-I-Q\n");  // 打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印字符串
    print("\n");  // 打印换行
    print("\n");  // 打印换行
    print("\n");  // 打印换行
    for (r = 0; r <= 70; r++)  // 循环
        b[r] = 0;  // 将数组 b 的元素赋值为 0
    # 打印棋盘的布局
    print("HERE IS THE BOARD:\n");
    print("\n");
    print("          !    !    !\n");
    print("         13   14   15\n");
    print("\n");
    print("          !    !    !\n");
    print("         22   23   24\n");
    print("\n");
    print("!    !    !    !    !    !    !\n");
    print("29   30   31   32   33   34   35\n");
    print("\n");
    print("!    !    !    !    !    !    !\n");
    print("38   39   40   41   42   43   44\n");
    print("\n");
    print("!    !    !    !    !    !    !\n");
    print("47   48   49   50   51   52   53\n");
    print("\n");
    print("          !    !    !\n");
    print("         58   59   60\n");
    print("\n");
    # 打印游戏板的提示信息
    print("          !    !    !\n");
    print("         67   68   69\n");
    print("\n");
    print("TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD\n");
    print("WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG\n");
    print("NUMBERS.  OK, LET'S BEGIN.\n");
    # 进入游戏循环
    while (1) {
        # 设置游戏板
        for (r = 1; r <= 9; r++) {
            t[r] = [];
            for (c = 1; c <= 9; c++) {
                # 根据条件设置游戏板上的值
                if (r == 4 || r == 5 || r == 6 || c == 4 || c == 5 || c == 6 && (r != 1 && c != 1 && r != 9 && c != 9)) {
                    t[r][c] = 5;
                } else {
                    t[r][c] = -5;
                }
            }
        }
        # 设置中心位置的值为0
        t[5][5] = 0;
        # 打印游戏板
        print_board();
        // 初始化辅助棋盘
        for (w = 1; w <= 33; w++) {
            b[m[w]] = -7;
        }
        b[41] = -3;
        // 输入移动并检查合法性
        do {
            while (1) {
                print("移动哪个棋子");
                z = parseInt(await input());
                if (b[z] == -7) {
                    print("移动到哪里");
                    p = parseInt(await input());
                    if (p != z
                        && b[p] != 0
                        && b[p] != -7
                        && (z + p) % 2 == 0
                        && (Math.abs(z - p) - 2) * (Math.abs(z - p) - 18) == 0
                        && update_board())
                        break;
                }
                // 如果移动不合法，提示玩家重新尝试
                print("ILLEGAL MOVE, TRY AGAIN...\n");
            }
            // 打印游戏棋盘
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
        print("\n");
        // 询问玩家是否再次游戏
        print("PLAY AGAIN (YES OR NO)");
        str = await input();
        // 如果玩家选择不再次游戏，跳出循环
        if (str == "NO")
            break;
    }
    print("\n");
    // 打印结束语
    print("SO LONG FOR NOW.\n");
    print("\n");  # 打印一个换行符

}

main();  # 调用名为main的函数
```