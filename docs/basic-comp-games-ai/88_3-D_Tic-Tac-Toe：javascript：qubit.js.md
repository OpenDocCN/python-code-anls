# `d:/src/tocomm/basic-computer-games\88_3-D_Tic-Tac-Toe\javascript\qubit.js`

```
// 创建一个名为print的函数，用于向页面输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 创建一个名为input的函数，用于获取用户输入
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，用于异步处理用户输入
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 在页面上输出提示符
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
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的是回车键
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环执行
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

var xa = [];  // 创建一个空数组xa
var la = [];  // 创建一个空数组la
var ma = [[],  // 创建一个二维数组ma，第一个元素为空数组
          [,1,2,3,4],    // 1
          [,5,6,7,8],    // 2
          [,9,10,11,12], // 3
          [,13,14,15,16],    // 4
          [,17,18,19,20],    // 5
          [,21,22,23,24],    // 6
          [,25,26,27,28],    // 7
          [,29,30,31,32],    // 8
          [,33,34,35,36],    // 9
          [,37,38,39,40],    // 10
          [,41,42,43,44],    // 11
          [,45,46,47,48],    // 12
          [,49,50,51,52],    // 13
          [,53,54,55,56],    // 14  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为53, 54, 55, 56
          [,57,58,59,60],    // 15  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为57, 58, 59, 60
          [,61,62,63,64],    // 16  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为61, 62, 63, 64
          [,1,17,33,49], // 17  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为1, 17, 33, 49
          [,5,21,37,53],    // 18  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为5, 21, 37, 53
          [,9,25,41,57],   // 19  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为9, 25, 41, 57
          [,13,29,45,61], // 20  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为13, 29, 45, 61
          [,2,18,34,50], // 21  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为2, 18, 34, 50
          [,6,22,38,54],    // 22  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为6, 22, 38, 54
          [,10,26,42,58],  // 23  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为10, 26, 42, 58
          [,14,30,46,62],   // 24  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为14, 30, 46, 62
          [,3,19,35,51], // 25  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为3, 19, 35, 51
          [,7,23,39,55],    // 26  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为7, 23, 39, 55
          [,11,27,43,59],  // 27  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为11, 27, 43, 59
          [,15,31,47,63], // 28  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为15, 31, 47, 63
          [,4,20,36,52], // 29  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为4, 20, 36, 52
          [,8,24,40,56], // 30  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为8, 24, 40, 56
          [,12,28,44,60],    // 31  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为12, 28, 44, 60
          [,16,32,48,64],    // 32  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为16, 32, 48, 64
          [,1,5,9,13],   // 33  # 创建一个包含四个元素的数组，第一个元素为空，后面三个元素分别为1, 5, 9, 13
          [,17,21,25,29],    // 34
          [,33,37,41,45],    // 35
          [,49,53,57,61],    // 36
          [,2,6,10,14],  // 37
          [,18,22,26,30],    // 38
          [,34,38,42,46],    // 39
          [,50,54,58,62],    // 40
          [,3,7,11,15],  // 41
          [,19,23,27,31],    // 42
          [,35,39,43,47],    // 43
          [,51,55,59,63],    // 44
          [,4,8,12,16],  // 45
          [,20,24,28,32],    // 46
          [,36,40,44,48],    // 47
          [,52,56,60,64],    // 48
          [,1,6,11,16],  // 49
          [,17,22,27,32],    // 50
          [,33,38,43,48],    // 51
          [,49,54,59,64],    // 52
          [,13,10,7,4],  // 53
```

这段代码看起来像是一个二维数组，每个元素是一个包含4个数字的数组。每个数组代表一个方块的位置，分别是左上角、右上角、左下角、右下角的坐标。
          [,29,26,23,20],    // 54
          [,45,42,39,36],    // 55
          [,61,58,55,52],    // 56
          [,1,21,41,61], // 57
          [,2,22,42,62], // 58
          [,3,23,43,63], // 59
          [,4,24,44,64], // 60
          [,49,37,25,13],    // 61
          [,50,38,26,14],    // 62
          [,51,39,27,15],    // 63
          [,52,40,28,16],    // 64
          [,1,18,35,52], // 65
          [,5,22,39,56], // 66
          [,9,26,43,60], // 67
          [,13,30,47,64],    // 68
          [,49,34,19,4], // 69
          [,53,38,23,8], // 70
          [,57,42,27,12],    // 71
          [,61,46,31,16],    // 72
          [,1,22,43,64], // 73
```

这段代码看起来像是一个二维数组，每个元素是一个数组，包含了四个数字。每个注释表示了该数组元素的索引。
          [,16,27,38,49],    // 74  // 定义一个二维数组，用于存储一些数值
          [,4,23,42,61], // 75  // 定义一个二维数组，用于存储一些数值
          [,13,26,39,52] // 76  // 定义一个二维数组，用于存储一些数值
          ];
var ya = [,1,49,52,4,13,61,64,16,22,39,23,38,26,42,27,43];  // 定义一个数组，用于存储一些数值

function show_board()  // 定义一个名为 show_board 的函数
{
    for (xx = 1; xx <= 9; xx++)  // 循环，xx 从 1 到 9
        print("\n");  // 打印换行符
    for (i = 1; i <= 4; i++) {  // 循环，i 从 1 到 4
        for (j = 1; j <= 4; j++) {  // 循环，j 从 1 到 4
            str = "";  // 定义一个空字符串
            for (i1 = 1; i1 <= j; i1++)  // 循环，i1 从 1 到 j
                str += "   ";  // 在字符串后面添加三个空格
            for (k = 1; k <= 4; k++) {  // 循环，k 从 1 到 4
                q = 16 * i + 4 * j + k - 20;  // 计算 q 的值
                if (xa[q] == 0)  // 如果 xa[q] 的值等于 0
                    str += "( )      ";  // 在字符串后面添加 "( )      "
                if (xa[q] == 5)  // 如果 xa[q] 的值等于 5
                    str += "(M)      ";  # 如果xa[q]等于1，向字符串str中添加"(M)      "
                if (xa[q] == 1)  # 如果xa[q]等于1
                    str += "(Y)      ";  # 向字符串str中添加"(Y)      "
                if (xa[q] == 1 / 8)  # 如果xa[q]等于1/8
                    str += "( )      ";  # 向字符串str中添加"( )      "
            }
            print(str + "\n");  # 打印字符串str并换行
            print("\n");  # 打印空行
        }
        print("\n");  # 打印空行
        print("\n");  # 打印空行
    }
}

function process_board()  # 定义名为process_board的函数
{
    for (i = 1; i <= 64; i++) {  # 循环i从1到64
        if (xa[i] == 1 / 8)  # 如果xa[i]等于1/8
            xa[i] = 0;  # 将xa[i]赋值为0
    }
}

# 检查每行的数据并计算结果
function check_for_lines()
{
    for (s = 1; s <= 76; s++) {
        j1 = ma[s][1];  # 获取二维数组ma中第s行第1列的值
        j2 = ma[s][2];  # 获取二维数组ma中第s行第2列的值
        j3 = ma[s][3];  # 获取二维数组ma中第s行第3列的值
        j4 = ma[s][4];  # 获取二维数组ma中第s行第4列的值
        la[s] = xa[j1] + xa[j2] + xa[j3] + xa[j4];  # 计算并存储结果到数组la中的第s个位置
    }
}

# 显示方块的位置
function show_square(m)
{
    k1 = Math.floor((m - 1) / 16) + 1;  # 计算方块所在的行数
    j2 = m - 16 * (k1 - 1);  # 计算方块所在的列数
    k2 = Math.floor((j2 - 1) / 4) + 1;  # 计算方块所在的小块行数
    k3 = m - (k1 - 1) * 16 - (k2 - 1) * 4;  # 计算方块所在的小块列数
    m = k1 * 100 + k2 * 10 + k3;  # 重新计算方块的位置并存储到变量m中
```
这些注释解释了每个语句的作用和功能，使得代码更易于理解和维护。
    print(" " + m + " ");  # 打印出 m 变量的值，带有空格前后缀

function select_move() {
    if (i % 4 <= 1) {  # 如果 i 除以 4 的余数小于等于 1
        a = 1;  # 将 a 赋值为 1
    } else {
        a = 2;  # 否则将 a 赋值为 2
    }
    for (j = a; j <= 5 - a; j += 5 - 2 * a) {  # 循环 j 从 a 到 5-a，每次增加 5-2*a
        if (xa[ma[i][j]] == s)  # 如果 xa[ma[i][j]] 的值等于 s
            break;  # 退出循环
    }
    if (j > 5 - a)  # 如果 j 大于 5-a
        return false;  # 返回 false
    xa[ma[i][j]] = s;  # 将 xa[ma[i][j]] 的值赋为 s
    m = ma[i][j];  # 将 m 赋值为 ma[i][j]
    print("MACHINE TAKES");  # 打印出 "MACHINE TAKES"
    show_square(m);  # 调用 show_square 函数并传入 m 变量
    return true;  # 返回 true
}

// 主控制部分
async function main()
{
    // 打印标题
    print(tab(33) + "QUBIC\n");
    // 打印创意计算的地点
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    while (1) {
        // 询问用户是否需要说明
        print("DO YOU WANT INSTRUCTIONS");
        // 等待用户输入
        str = await input();
        // 截取用户输入的第一个字符
        str = str.substr(0, 1);
        // 如果用户输入是"Y"或"N"，则跳出循环
        if (str == "Y" || str == "N")
            break;
        // 如果用户输入不是"Y"或"N"，则提示输入错误
        print("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'");
    }
    // 如果用户输入是"Y"，则打印空行
    if (str == "Y") {
        print("\n");
# 打印游戏规则和提示信息
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
# 进入游戏循环
}
while (1) {
    # 初始化游戏变量
    for (i = 1; i <= 64; i++)
        xa[i] = 0;
    z = 1;
    # 提示玩家是否要先手
    print("DO YOU WANT TO MOVE FIRST");
        while (1) {
            # 无限循环，等待用户输入
            str = await input();
            # 截取用户输入的第一个字符
            str = str.substr(0, 1);
            # 如果用户输入的是"Y"或"N"，则跳出循环
            if (str == "Y" || str == "N")
                break;
            # 如果用户输入不是"Y"或"N"，则打印错误提示信息，继续等待用户输入
            print("INCORRECT ANSWER.  PLEASE TYPE 'YES' OR 'NO'");
        }
        while (1) {
            while (1) {
                # 打印提示信息，等待用户输入
                print(" \n");
                print("YOUR MOVE");
                # 将用户输入的字符串转换为整数
                j1 = parseInt(await input());
                # 如果用户输入的是0，则显示游戏棋盘并继续等待用户输入
                if (j1 == 0) {
                    show_board();
                    continue;
                }
                # 如果用户输入的是1，则结束当前函数的执行
                if (j1 == 1)
                    return;
                # 计算用户输入的数值的百位和个位
                k1 = Math.floor(j1 / 100);
                j2 = j1 - k1 * 100;
                # 计算 k2 的值，j2 除以 10 的商
                k2 = Math.floor(j2 / 10);
                # 计算 k3 的值，j2 减去 k2 乘以 10 的余数
                k3 = j2 - k2 * 10;
                # 计算 m 的值，根据给定的公式
                m = 16 * k1 + 4 * k2 + k3 - 20;
                # 如果 k1、k2、k3 的值不在指定范围内，则打印错误信息
                if (k1 < 1 || k2 < 1 || k3 < 1 || k1 > 4 || k2 > 4 || k3 >> 4) {
                    print("INCORRECT MOVE, RETYPE IT--");
                } else {
                    # 处理棋盘
                    process_board();
                    # 如果 xa[m] 不等于 0，则打印错误信息
                    if (xa[m] != 0) {
                        print("THAT SQUARE IS USED, TRY AGAIN.\n");
                    } else {
                        # 否则跳出循环
                        break;
                    }
                }
            }
            # 将 xa[m] 的值设为 1
            xa[m] = 1;
            # 检查是否有连成一线的情况
            check_for_lines();
            # 将状态设为 0
            status = 0;
            # 循环遍历 j 从 1 到 3
            for (j = 1; j <= 3; j++) {
                # 循环遍历 i 从 1 到 76
                for (i = 1; i <= 76; i++) {
                    # 如果 j 等于 1
                    if (j == 1) {
# 如果列表la的第i个元素不等于4，则跳过当前循环，继续下一次循环
if (la[i] != 4)
    continue;
# 打印“YOU WIN AS FOLLOWS”
print("YOU WIN AS FOLLOWS");
# 遍历1到4的数字，将ma[i][j]赋值给m，然后展示m对应的方块
for (j = 1; j <= 4; j++) {
    m = ma[i][j];
    show_square(m);
}
# 将status赋值为1
status = 1;
# 跳出循环
break;
# 如果j等于2，则执行以下代码
if (j == 2) {
    # 如果列表la的第i个元素不等于15，则跳过当前循环，继续下一次循环
    if (la[i] != 15)
        continue;
    # 遍历1到4的数字
    for (j = 1; j <= 4; j++) {
        # 将ma[i][j]赋值给m
        m = ma[i][j];
        # 如果xa[m]不等于0，则跳过当前循环，继续下一次循环
        if (xa[m] != 0)
            continue;
        # 将xa[m]赋值为5
        xa[m] = 5;
        # 打印“MACHINE MOVES TO ”
        print("MACHINE MOVES TO ");
        # 展示m对应的方块
        show_square(m);
                    }
                    # 打印", AND WINS AS FOLLOWS"
                    print(", AND WINS AS FOLLOWS");
                    # 遍历1到4的数字
                    for (j = 1; j <= 4; j++) {
                        # 将ma[i][j]的值赋给m
                        m = ma[i][j];
                        # 调用show_square函数，传入m作为参数
                        show_square(m);
                    }
                    # 将status赋值为1
                    status = 1;
                    # 跳出循环
                    break;
                }
                # 如果j等于3
                if (j == 3) {
                    # 如果la[i]不等于3，继续循环
                    if (la[i] != 3)
                        continue;
                    # 打印"NICE TRY, MACHINE MOVES TO"
                    print("NICE TRY, MACHINE MOVES TO");
                    # 遍历1到4的数字
                    for (j = 1; j <= 4; j++) {
                        # 将ma[i][j]的值赋给m
                        m = ma[i][j];
                        # 如果xa[m]不等于0，继续循环
                        if (xa[m] != 0)
                            continue;
                        # 将xa[m]赋值为5
                        xa[m] = 5;
                        # 调用show_square函数，传入m作为参数
                        show_square(m);
                        # 将status赋值为2
                        status = 2;
                        }  # 结束内层循环
                        break;  # 结束内层循环
                    }  # 结束内层循环
                }  # 结束内层循环
                if (i <= 76)  # 如果 i 小于等于 76
                    break;  # 结束外层循环
            }  # 结束外层循环
            if (status == 2)  # 如果 status 等于 2
                continue;  # 继续下一次循环
            if (status == 1)  # 如果 status 等于 1
                break;  # 结束循环
            // x = x; non-useful in original  # 注释：这行代码在原始代码中没有用处
            i = 1;  # 将 i 设为 1
            do {  # 开始 do-while 循环
                la[i] = xa[ma[i][1]] + xa[ma[i][2]] + xa[ma[i][3]] + xa[ma[i][4]];  # 计算 la[i] 的值
                l = la[i];  # 将 l 设为 la[i] 的值
                if (l == 10) {  # 如果 l 等于 10
                    for (j = 1; j <= 4; j++) {  # 开始 for 循环
                        if (xa[ma[i][j]] == 0)  # 如果 xa[ma[i][j]] 等于 0
                            xa[ma[i][j]] = 1 / 8;  # 将 xa[ma[i][j]] 设为 1/8
                    }
                }
            } while (++i <= 76) ;  # 使用do-while循环遍历76次，直到i大于76为止
            check_for_lines();  # 调用check_for_lines函数
            i = 1;  # 将i的值设为1
            do {
                if (la[i] == 0.5) {  # 如果la[i]的值等于0.5
                    s = 1 / 8;  # 将s的值设为1/8
                    select_move();  # 调用select_move函数
                    break;  # 跳出循环
                }
                if (la[i] == 5 + 3 / 8) {  # 如果la[i]的值等于5加3/8
                    s = 1 / 8;  # 将s的值设为1/8
                    select_move();  # 调用select_move函数
                    break;  # 跳出循环
                }
            } while (++i <= 76) ;  # 使用do-while循环遍历76次，直到i大于76为止
            if (i <= 76)  # 如果i小于等于76
                continue;  # 继续循环
            process_board();  # 调用 process_board 函数处理游戏板

            i = 1;  # 初始化变量 i 为 1
            do {
                la[i] = xa[ma[i][1]] + xa[ma[i][2]] + xa[ma[i][3]] + xa[ma[i][4]];  # 计算 la[i] 的值
                l = la[i];  # 将 la[i] 的值赋给变量 l
                if (l == 2) {  # 如果 l 的值等于 2
                    for (j = 1; j <= 4; j++) {  # 循环遍历 j 从 1 到 4
                        if (xa[ma[i][j]] == 0)  # 如果 xa[ma[i][j]] 的值等于 0
                            xa[ma[i][j]] = 1 / 8;  # 将 xa[ma[i][j]] 的值设置为 1/8
                    }
                }
            } while (++i <= 76) ;  # 当 i 小于等于 76 时继续循环
            check_for_lines();  # 调用 check_for_lines 函数检查是否有满行
            i = 1;  # 重新初始化变量 i 为 1
            do {
                if (la[i] == 0.5) {  # 如果 la[i] 的值等于 0.5
                    s = 1 / 8;  # 将变量 s 的值设置为 1/8
                    select_move();  # 调用 select_move 函数进行移动选择
                    break;  # 跳出循环
                }
                # 如果 la[i] 等于 1 + 3 / 8
                if (la[i] == 1 + 3 / 8) {
                    # 将 s 设为 1 / 8
                    s = 1 / 8;
                    # 调用 select_move() 函数
                    select_move();
                    # 跳出循环
                    break;
                }
            } while (++i <= 76) ;
            # 如果 i 小于等于 76，则继续循环
            if (i <= 76)
                continue;

            # 循环 k 从 1 到 18
            for (k = 1; k <= 18; k++) {
                # 将 p 设为 0
                p = 0;
                # 循环 i 从 4 * k - 3 到 4 * k
                for (i = 4 * k - 3; i <= 4 * k; i++) {
                    # 循环 j 从 1 到 4
                    for (j = 1; j <= 4; j++)
                        # 将 xa[ma[i][j]] 的值加到 p 上
                        p += xa[ma[i][j]];
                }
                # 如果 p 等于 4 或者等于 9
                if (p == 4 || p == 9) {
                    # 将 s 设为 1 / 8
                    s = 1 / 8;
                    # 再次循环 i 从 4 * k - 3 到 4 * k
                    for (i = 4 * k - 3; i <= 4 * k; i++) {
                        # 如果 select_move() 返回真值
                        if (select_move())
# 如果条件满足，则跳出当前循环
                            break;
                    }
                    # 重置变量 s 为 0
                    s = 0;
                }
            }
            # 如果条件满足，则继续执行下一次循环
            if (k <= 18)
                continue
            # 调用 process_board() 函数
            process_board();
            # 设置变量 z 为 1
            z = 1;
            # 执行 do-while 循环，直到条件不满足
            do {
                # 如果条件满足，则跳出当前循环
                if (xa[ya[z]] == 0)
                    break;
            } while (++z < 17) ;
            # 如果 z 大于等于 17，则执行以下代码块
            if (z >= 17) {
                # 执行 for 循环，i 从 1 到 64
                for (i = 1; i <= 64; i++) {
                    # 如果条件满足，则执行以下代码块
                    if (xa[i] == 0) {
                        # 设置 xa[i] 为 5
                        xa[i] = 5;
                        # 设置变量 m 为 i
                        m = i;
                        # 打印 "MACHINE LIKES"
                        print("MACHINE LIKES");
                        # 跳出当前循环
                        break;
                }
            }
            if (i > 64) {  # 如果游戏步数超过64步
                print("THE GAME IS A DRAW.\n");  # 打印游戏平局信息
                break;  # 结束游戏
            }
        } else {  # 如果不是玩家的回合
            m = ya[z];  # 从ya数组中获取机器移动的位置
            xa[m] = 5;  # 在xa数组中标记机器移动的位置为5
            print("MACHINE MOVES TO");  # 打印机器移动到的位置
        }
        show_square(m);  # 显示移动后的棋盘
    }
    print(" \n");  # 打印空行
    print("DO YOU WANT TO TRY ANOTHER GAME");  # 询问玩家是否想再玩一局游戏
    while (1) {  # 进入循环，直到玩家输入合法的选项
        str = await input();  # 获取玩家输入的字符串
        str = str.substr(0, 1);  # 截取字符串的第一个字符
        if (str == "Y" || str == "N")  # 如果玩家输入的是Y或者N
            break;  # 结束循环
# 打印错误提示信息，要求用户只能输入'YES'或'NO'
print("INCORRECT ANSWER. PLEASE TYPE 'YES' OR 'NO'");
# 如果用户输入的是'N'，则跳出循环
if (str == "N")
    break;
# 调用主函数
main();
```