# `basic-computer-games\88_3-D_Tic-Tac-Toe\javascript\qubit.js`

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
                       // 设置输入框类型为文本，长度为50
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
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

// 定义一些全局变量
var xa = [];
var la = [];
var ya = [,1,49,52,4,13,61,64,16,22,39,23,38,26,42,27,43];

// 定义一个展示棋盘的函数
function show_board()
{
    // 循环打印换行符
    for (xx = 1; xx <= 9; xx++)
        print("\n");
    # 循环变量 i 从 1 到 4
    for (i = 1; i <= 4; i++) {
        # 循环变量 j 从 1 到 4
        for (j = 1; j <= 4; j++) {
            # 初始化空字符串
            str = "";
            # 循环变量 i1 从 1 到 j
            for (i1 = 1; i1 <= j; i1++)
                # 拼接空格到字符串
                str += "   ";
            # 循环变量 k 从 1 到 4
            for (k = 1; k <= 4; k++) {
                # 计算索引值
                q = 16 * i + 4 * j + k - 20;
                # 判断索引值对应的元素是否为 0
                if (xa[q] == 0)
                    # 拼接字符串
                    str += "( )      ";
                # 判断索引值对应的元素是否为 5
                if (xa[q] == 5)
                    # 拼接字符串
                    str += "(M)      ";
                # 判断索引值对应的元素是否为 1
                if (xa[q] == 1)
                    # 拼接字符串
                    str += "(Y)      ";
                # 判断索引值对应的元素是否为 1/8
                if (xa[q] == 1 / 8)
                    # 拼接字符串
                    str += "( )      ";
            }
            # 打印字符串
            print(str + "\n");
            # 打印空行
            print("\n");
        }
        # 打印两个空行
        print("\n");
        print("\n");
    }
}

// 处理棋盘
function process_board()
{
    // 遍历64个方块
    for (i = 1; i <= 64; i++) {
        // 如果xa[i]等于1/8，将其置为0
        if (xa[i] == 1 / 8)
            xa[i] = 0;
    }
}

// 检查是否有连线
function check_for_lines()
{
    // 遍历76个可能的连线
    for (s = 1; s <= 76; s++) {
        // 获取连线的四个方块的索引
        j1 = ma[s][1];
        j2 = ma[s][2];
        j3 = ma[s][3];
        j4 = ma[s][4];
        // 计算这条连线上四个方块的值的和
        la[s] = xa[j1] + xa[j2] + xa[j3] + xa[j4];
    }
}

// 显示方块
function show_square(m)
{
    // 计算方块的位置
    k1 = Math.floor((m - 1) / 16) + 1;
    j2 = m - 16 * (k1 - 1);
    k2 = Math.floor((j2 - 1) / 4) + 1;
    k3 = m - (k1 - 1) * 16 - (k2 - 1) * 4;
    m = k1 * 100 + k2 * 10 + k3;
    // 打印方块的位置
    print(" " + m + " ");
}

// 选择移动
function select_move() {
    // 根据i的值确定a的取值
    if (i % 4 <= 1) {
        a = 1;
    } else {
        a = 2;
    }
    // 根据a的值遍历方块
    for (j = a; j <= 5 - a; j += 5 - 2 * a) {
        // 如果找到符合条件的方块，跳出循环
        if (xa[ma[i][j]] == s)
            break;
    }
    // 如果没有找到符合条件的方块，返回false
    if (j > 5 - a)
        return false;
    // 将找到的方块标记为s
    xa[ma[i][j]] = s;
    m = ma[i][j];
    // 打印机器选择的方块
    print("MACHINE TAKES");
    show_square(m);
    return true;
}

// 主控制部分
async function main()
{
    // 打印游戏名称和信息
    print(tab(33) + "QUBIC\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 循环直到用户输入正确的指令
    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        // 等待用户输入
        str = await input();
        // 截取用户输入的第一个字符
        str = str.substr(0, 1);
        // 如果用户输入了"Y"或"N"，跳出循环
        if (str == "Y" || str == "N")
            break;
        // 如果用户输入不是"Y"或"N"，提示用户重新输入
        print("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'");
    }
}
    # 如果输入的字符串为 "Y"，则执行以下操作
    if (str == "Y") {
        # 打印空行
        print("\n");
        # 打印游戏规则说明
        print("THE GAME IS TIC-TAC-TOE IN A 4 X 4 X 4 CUBE.\n");
        print("EACH MOVE IS INDICATED BY A 3 DIGIT NUMBER, WITH EACH\n");
        print("DIGIT BETWEEN 1 AND 4 INCLUSIVE.  THE DIGITS INDICATE THE\n");
        print("LEVEL, ROW, AND COLUMN, RESPECTIVELY, OF THE OCCUPIED\n");
        print("PLACE.  \n");
        print("\n");
        print("TO PRINT THE PLAYING BOARD, TYPE 0 (ZERO) AS YOUR MOVE.\n");
        print("THE PROGRAM WILL PRINT THE BOARD WITH YOUR MOVES INDI-\n");
        print("CATED WITH A (Y), THE MACHINE'S MOVES WITH AN (M), AND\n");
        print("UNUSED SQUARES WITH A ( ).  OUTPUT IS ON PAPER.\n");
        print("\n");
        print("TO STOP THE PROGRAM RUN, TYPE 1 AS YOUR MOVE.\n");
        print("\n");
        print("\n");
    }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```