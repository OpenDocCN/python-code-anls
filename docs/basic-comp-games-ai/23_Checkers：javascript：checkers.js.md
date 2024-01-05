# `d:/src/tocomm/basic-computer-games\23_Checkers\javascript\checkers.js`

```
// 定义一个名为print的函数，用于在页面上输出字符串
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为JavaScript代码添加了注释，解释了每个函数的作用和功能。
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
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 在字符串末尾添加一个空格
    return str;  // 返回处理后的字符串

}

// x,y = 原始位置的方块
// a,b = 移动方向
function try_computer()
{
    u = x + a;  // 计算新位置的横坐标
    v = y + b;  // 计算新位置的纵坐标
    if (u < 0 || u > 7 || v < 0 || v > 7)  // 如果新位置超出了边界
        return;  // 结束函数
    if (s[u][v] == 0) {  // 如果新位置为空
        eval_move();  // 执行移动
        return;  // 结束函数
    }
    if (s[u][v] < 0)	// 不能跳过自己的棋子
        return;  // 结束函数
    u += a;  // 计算新位置的横坐标
    u += b;  // 计算新位置的纵坐标
    # 如果目标方格的坐标超出了棋盘范围，则返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    # 如果目标方格上没有棋子，则调用 eval_move() 函数
    if (s[u][v] == 0)
        eval_move();
}

// x,y = 起始方格的坐标
// u,v = 目标方格的坐标
function eval_move()
{
    # 如果目标方格是第一行并且起始方格上的棋子是黑色，则增加 q 的值
    if (v == 0 && s[x][y] == -1)
        q += 2;
    # 如果目标方格与起始方格的列坐标差的绝对值为2，则增加 q 的值
    if (Math.abs(y - v) == 2)
        q += 5;
    # 如果目标方格的行坐标为7，则减少 q 的值
    if (y == 7)
        q -= 2;
    # 如果目标方格的列坐标为0或7，则增加 q 的值
    if (u == 0 || u == 7)
        q++;
    # 遍历相邻的方格
    for (c = -1; c <= 1; c += 2) {
        # 如果相邻方格的坐标超出了棋盘范围，则继续循环
        if (u + c < 0 || u + c > 7 || v + g < 0)
            continue;  // 跳过当前循环，继续执行下一次循环
        if (s[u + c][v + g] < 0) {  // 如果当前位置是计算机的棋子
            q++;  // 计数器加一
            continue;  // 跳过当前循环，继续执行下一次循环
        }
        if (u - c < 0 || u - c > 7 || v - g > 7)  // 如果目标位置超出棋盘范围
            continue;  // 跳过当前循环，继续执行下一次循环
        if (s[u + c][v + g] > 0 && (s[u - c][v - g] == 0 || (u - c == x && v - g == y)))  // 如果目标位置有玩家的棋子，并且目标位置为空或者是要移动的位置
            q -= 2;  // 计数器减二
    }
    if (q > r[0]) {  // 如果当前得分大于之前的最高得分
        r[0] = q;  // 更新最高得分
        r[1] = x;  // 记录起始位置
        r[2] = y;
        r[3] = u;  // 记录目标位置
        r[4] = v;
    }
    q = 0;  // 重置计数器
}
// 定义函数 more_captures，用于检查是否有更多的棋子可以被吃掉
function more_captures() {
    // 计算新的位置 u 和 v
    u = x + a;
    v = y + b;
    // 如果新位置超出了棋盘范围，则返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果新位置上没有棋子，并且中间位置有对方的棋子，则执行吃子操作
    if (s[u][v] == 0 && s[x + a / 2][y + b / 2] > 0)
        eval_move();
}

// 初始化变量 r 为 [-99, 0, 0, 0, 0]
var r = [-99, 0, 0, 0, 0];
// 初始化二维数组 s
var s = [];

// 初始化二维数组 s 的每一行
for (x = 0; x <= 7; x++)
    s[x] = [];

// 初始化变量 g 为 -1
var g = -1;
// 初始化数组 data
var data = [1, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, -1, 15];
// 初始化变量 p 和 q 为 0
var p = 0;
var q = 0;
// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(32) + "CHECKERS\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏规则
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,\n");
    print("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.\n");
    print("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.\n");
    print("(0,0) IS THE LOWER LEFT CORNER\n");
    print("(0,7) IS THE UPPER LEFT CORNER\n");
    print("(7,0) IS THE LOWER RIGHT CORNER\n");
    print("(7,7) IS THE UPPER RIGHT CORNER\n");
    print("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER\n");
    print("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.\n");
    print("\n");
    print("\n");
    print("\n");
}
    for (x = 0; x <= 7; x++) {  // 循环遍历 x 坐标，范围是 0 到 7
        for (y = 0; y <= 7; y++) {  // 在每个 x 坐标下，循环遍历 y 坐标，范围是 0 到 7
            if (data[p] == 15)  // 如果 data 数组中的值等于 15
                p = 0;  // 将 p 重置为 0
            s[x][y] = data[p];  // 将 data 数组中的值赋给 s 数组中对应的位置
            p++;  // p 自增
        }
    }
    while (1) {  // 进入无限循环

        // 在棋盘上搜索最佳移动
        for (x = 0; x <= 7; x++) {  // 循环遍历 x 坐标，范围是 0 到 7
            for (y = 0; y <= 7; y++) {  // 在每个 x 坐标下，循环遍历 y 坐标，范围是 0 到 7
                if (s[x][y] > -1)  // 如果 s 数组中的值大于 -1
                    continue;  // 继续下一次循环
                if (s[x][y] == -1) {	// 棋子
                    for (a = -1; a <= 1; a += 2) {  // 循环遍历 a，范围是 -1 到 1，步长为 2
                        b = g;	// 只前进
                        try_computer();  // 调用 try_computer 函数
                    }
                } else if (s[x][y] == -2) {	// King
                    // 如果当前位置是国王的位置
                    for (a = -1; a <= 1; a += 2) {
                        for (b = -1; b <= 1; b += 2) {
                            // 尝试计算国王的下一步移动
                            try_computer();
                        }
                    }
                }
            }
        }
        if (r[0] == -99) {
            // 如果计算机赢了游戏
            print("\n");
            print("YOU WIN.\n");
            break;
        }
        // 打印计算机的移动信息
        print("FROM " + r[1] + "," + r[2] + " TO " + r[3] + "," + r[4]);
        r[0] = -99;
        while (1) {
            if (r[4] == 0) {	// Computer reaches the bottom
                s[r[3]][r[4]] = -2;	// King
                break;
            }
            s[r[3]][r[4]] = s[r[1]][r[2]];	// Move  // 将源位置的棋子移动到目标位置
            s[r[1]][r[2]] = 0;  // 将源位置置空
            if (Math.abs(r[1] - r[3]) == 2) {  // 如果移动的距离为2
                s[(r[1] + r[3]) / 2][(r[2] + r[4]) / 2] = 0;	// Capture  // 将被跳过的对方棋子位置置空
                x = r[3];  // 记录目标位置的横坐标
                y = r[4];  // 记录目标位置的纵坐标
                if (s[x][y] == -1) {  // 如果目标位置的棋子是-1
                    b = -2;  // 将b置为-2
                    for (a = -2; a <= 2; a += 4) {  // 遍历a从-2到2，步长为4
                        more_captures();  // 调用函数进行更多的跳吃
                    }
                } else if (s[x][y] == -2) {  // 如果目标位置的棋子是-2
                    for (a = -2; a <= 2; a += 4) {  // 遍历a从-2到2，步长为4
                        for (b = -2; b <= 2; b += 4) {  // 遍历b从-2到2，步长为4
                            more_captures();  // 调用函数进行更多的跳吃
                        }
                    }
                }
                if (r[0] != -99) {  // 如果r[0]不等于-99
                    print(" TO " + r[3] + "," + r[4]);  # 打印字符串 " TO " 和 r[3]、r[4] 的值
                    r[0] = -99;  # 将 r[0] 的值设为 -99
                    continue;  # 继续下一次循环
                }
            }
            break;  # 跳出循环
        }
        print("\n");  # 打印换行符
        print("\n");  # 打印换行符
        print("\n");  # 打印换行符
        for (y = 7; y >= 0; y--) {  # 循环，y 从 7 到 0
            str = "";  # 初始化字符串 str
            for (x = 0; x <= 7; x++) {  # 循环，x 从 0 到 7
                if (s[x][y] == 0)  # 如果 s[x][y] 的值为 0
                    str += ".";  # 在字符串 str 后面添加字符 "."
                if (s[x][y] == 1)  # 如果 s[x][y] 的值为 1
                    str += "O";  # 在字符串 str 后面添加字符 "O"
                if (s[x][y] == -1)  # 如果 s[x][y] 的值为 -1
                    str += "X";  # 在字符串 str 后面添加字符 "X"
                if (s[x][y] == -2)  # 如果 s[x][y] 的值为 -2
                    str += "X*";  # 如果棋盘上的位置为1，表示黑子，将"X*"添加到字符串末尾
                if (s[x][y] == 2)  # 如果棋盘上的位置为2，表示白子
                    str += "O*";  # 将"O*"添加到字符串末尾
                while (str.length % 5)  # 当字符串长度不是5的倍数时
                    str += " ";  # 在末尾添加空格，使得字符串长度为5的倍数
            }
            print(str + "\n");  # 打印拼接好的字符串并换行
            print("\n");  # 打印一个空行
        }
        print("\n");  # 打印一个空行
        z = 0;  # 初始化变量z为0
        t = 0;  # 初始化变量t为0
        for (l = 0; l <= 7; l++) {  # 遍历棋盘的行
            for (m = 0; m <= 7; m++) {  # 遍历棋盘的列
                if (s[l][m] == 1 || s[l][m] == 2)  # 如果棋盘上的位置为1或2，表示有黑子或白子
                    z = 1;  # 将z设为1
                if (s[l][m] == -1 || s[l][m] == -2)  # 如果棋盘上的位置为-1或-2，表示有黑子或白子
                    t = 1;  # 将t设为1
            }
        }
        # 如果 z 不等于 1，则打印换行符和 "I WIN."，然后跳出循环
        if (z != 1) {
            print("\n");
            print("I WIN.\n");
            break;
        }
        # 如果 t 不等于 1，则打印换行符和 "YOU WIN."，然后跳出循环
        if (t != 1) {
            print("\n");
            print("YOU WIN.\n");
            break;
        }
        # 从用户输入获取坐标，并将其转换为整数
        do {
            print("FROM");
            e = await input();
            h = parseInt(e.substr(e.indexOf(",") + 1));
            e = parseInt(e);
            x = e;
            y = h;
        } while (s[x][y] <= 0) ;  # 当 s[x][y] 的值小于等于 0 时继续循环
        # 执行用户输入的目标坐标
        do {
            print("TO");
            a = await input();  # 从输入中获取值并赋给变量a
            b = parseInt(a.substr(a.indexOf(",") + 1));  # 从a中获取逗号后的部分并转换为整数赋给变量b
            a = parseInt(a);  # 将a转换为整数
            x = a;  # 将a的值赋给变量x
            y = b;  # 将b的值赋给变量y
            if (s[x][y] == 0 && Math.abs(a - e) <= 2 && Math.abs(a - e) == Math.abs(b - h))  # 如果s[x][y]等于0且a与e的绝对值小于等于2且a与e的绝对值等于b与h的绝对值
                break;  # 跳出循环
            print("WHAT?\n");  # 打印"WHAT?"
        } while (1) ;  # 循环直到条件不满足
        i = 46;  # 将46赋给变量i
        do {
            s[a][b] = s[e][h]  # 将s[e][h]的值赋给s[a][b]
            s[e][h] = 0;  # 将0赋给s[e][h]
            if (Math.abs(e - a) != 2)  # 如果e与a的绝对值不等于2
                break;  # 跳出循环
            s[(e + a) / 2][(h + b) / 2] = 0;  # 将0赋给s[(e + a) / 2][(h + b) / 2]
            while (1) {  # 进入内层循环
                print("+TO");  # 打印"+TO"
                a1 = await input();  # 从输入中获取值并赋给变量a1
                b1 = parseInt(a1.substr(a1.indexOf(",") + 1));  # 从a1中获取逗号后的部分并转换为整数赋给变量b1
                a1 = parseInt(a1);  // 将变量 a1 转换为整数类型
                if (a1 < 0)  // 如果 a1 小于 0
                    break;  // 跳出循环
                if (s[a1][b1] == 0 && Math.abs(a1 - a) == 2 && Math.abs(b1 - b) == 2)  // 如果 s[a1][b1] 等于 0 并且 a1 和 a 的绝对值差为 2 且 b1 和 b 的绝对值差为 2
                    break;  // 跳出循环
            }
            if (a1 < 0)  // 如果 a1 小于 0
                break;  // 跳出循环
            e = a;  // 将变量 a 的值赋给 e
            h = b;  // 将变量 b 的值赋给 h
            a = a1;  // 将 a1 的值赋给 a
            b = b1;  // 将 b1 的值赋给 b
            i += 15;  // 将 i 的值增加 15
        } while (1);  // 无限循环
        if (b == 7)  // 如果 b 等于 7
            s[a][b] = 2;  // 将 s[a][b] 的值设为 2，表示玩家达到顶部成为国王
    }
}

main();  // 调用主函数
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```