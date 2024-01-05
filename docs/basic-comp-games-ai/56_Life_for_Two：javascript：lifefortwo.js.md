# `56_Life_for_Two\javascript\lifefortwo.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上显示提示符 "? "
// 设置 INPUT 元素的类型为文本输入类型
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

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回处理后的字符串

var na = [];  # 声明一个空数组
var ka = [, 3,102,103,120,130,121,112,111,12,  # 声明一个数组并初始化
          21,30,1020,1030,1011,1021,1003,1002,1012];
var aa = [,-1,0,1,0,0,-1,0,1,-1,-1,1,-1,-1,1,1,1];  # 声明一个数组并初始化
var xa = [];  # 声明一个空数组
var ya = [];  # 声明一个空数组
var j;  # 声明变量j
var k;  # 声明变量k
var m2;  # 声明变量m2
var m3;  # 声明变量m3

function show_data()  # 定义函数show_data
{
    k = 0;  # 变量k赋值为0
    m2 = 0;  # 变量m2赋值为0
    m3 = 0;  # 变量m3赋值为0
    for (j = 0; j <= 6; j++) {  # 循环变量 j 从 0 到 6
        print("\n");  # 打印换行符
        for (k = 0; k <= 6; k++) {  # 嵌套循环变量 k 从 0 到 6
            if (j == 0 || j == 6) {  # 如果 j 等于 0 或者等于 6
                if (k == 6)  # 如果 k 等于 6
                    print(" 0 ");  # 打印空格和 0
                else
                    print(" " + k + " ");  # 否则打印空格、k、空格
            } else if (k == 0 || k == 6) {  # 如果 k 等于 0 或者等于 6
                if (j == 6)  # 如果 j 等于 6
                    print(" 0\n");  # 打印空格和 0 并换行
                else
                    print(" " + j + " ");  # 否则打印空格、j、空格
            } else {  # 否则
                if (na[j][k] >= 3) {  # 如果 na[j][k] 大于等于 3
                    for (o1 = 1; o1 <= 18; o1++) {  # 循环变量 o1 从 1 到 18
                        if (na[j][k] == ka[o1])  # 如果 na[j][k] 等于 ka[o1]
                            break;  # 跳出循环
                    }
                    if (o1 <= 18) {  # 如果 o1 小于等于 18
# 如果 o1 小于等于 9，则将 na[j][k] 设置为 100，增加 m2 的值，打印 " * "
# 否则，将 na[j][k] 设置为 1000，增加 m3 的值，打印 " # "
# 如果不满足上述条件，则将 na[j][k] 设置为 0，打印 "   "
}

function process_board()
{
    // 循环遍历二维数组na的每个元素
    for (j = 1; j <= 5; j++) {
        for (k = 1; k <= 5; k++) {
            // 如果当前元素的值大于99
            if (na[j][k] > 99) {
                // 设置变量b的初始值为1
                b = 1;
                // 如果当前元素的值大于999，将变量b的值设为10
                if (na[j][k] > 999)
                    b = 10;
                // 循环遍历数组aa，每次增加2
                for (o1 = 1; o1 <= 15; o1 += 2) {
                    // 对当前元素周围的8个元素进行操作，将它们的值增加b
                    na[j + aa[o1]][k + aa[o1 + 1]] = na[j + aa[o1]][k + aa[o1 + 1]] + b;
                }
            }
        }
    }
    // 调用show_data函数
    show_data();
}

// 主程序
async function main()
{
    # 打印游戏标题
    print(tab(33) + "LIFE2\n");
    # 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    # 打印游戏名称
    print(tab(10) + "U.B. LIFE GAME\n");
    # 初始化变量m2和m3
    m2 = 0;
    m3 = 0;
    # 初始化二维数组na
    for (j = 0; j <= 6; j++) {
        na[j] = [];
        for (k = 0; k <= 6; k++)
            na[j][k] = 0;
    }
    # 循环两次，分别为两个玩家
    for (b = 1; b <= 2; b++) {
        # 根据玩家不同设置p1的值
        p1 = (b == 2) ? 30 : 3;
        # 打印玩家信息
        print("\n");
        print("PLAYER " + b + " - 3 LIVE PIECES.\n");
        # 循环3次，表示每个玩家有3个活着的棋子
        for (k1 = 1; k1 <= 3; k1++) {
# 进入一个无限循环，直到满足条件才会跳出循环
            while (1) {
                # 打印字符串 "X,Y\n"
                print("X,Y\n");
                # 等待用户输入，并将输入的字符串赋值给变量 str
                str = await input();
                # 将输入的字符串转换为整数并赋值给数组 ya 的第 b 个元素
                ya[b] = parseInt(str);
                # 从逗号后面的位置开始截取字符串并转换为整数，赋值给数组 xa 的第 b 个元素
                xa[b] = parseInt(str.substr(str.indexOf(",") + 1));
                # 如果满足条件，则跳出循环
                if (xa[b] > 0 && xa[b] < 6 && ya[b] > 0 && ya[b] < 5 && na[xa[b]][ya[b]] == 0)
                    break;
                # 如果不满足条件，则打印字符串 "ILLEGAL COORDS. RETYPE\n"，并继续循环
                print("ILLEGAL COORDS. RETYPE\n");
            }
            # 如果 b 不等于 1
            if (b != 1) {
                # 如果两个坐标相同
                if (xa[1] == xa[2] && ya[1] == ya[2]) {
                    # 打印字符串 "SAME COORD.  SET TO 0\n"
                    print("SAME COORD.  SET TO 0\n");
                    # 将数组 na 中的特定位置设置为 0
                    na[xa[b] + 1][ya[b] + 1] = 0;
                    # 将 b 设置为 99
                    b = 99;
                }
            }
            # 将数组 na 中的特定位置设置为 p1
            na[xa[b]][ya[b]] = p1;
        }
    }
    # 调用函数 show_data()
    show_data();
    while (1):  # 进入无限循环
        print("\n")  # 打印空行
        process_board()  # 调用 process_board() 函数处理游戏板
        if (m2 == 0 and m3 == 0):  # 如果 m2 和 m3 都等于 0
            print("\n")  # 打印空行
            print("A DRAW\n")  # 打印平局信息
            break  # 退出循环
        if (m3 == 0):  # 如果 m3 等于 0
            print("\n")  # 打印空行
            print("PLAYER 1 IS THE WINNER\n")  # 打印玩家 1 获胜信息
            break  # 退出循环
        if (m2 == 0):  # 如果 m2 等于 0
            print("\n")  # 打印空行
            print("PLAYER 2 IS THE WINNER\n")  # 打印玩家 2 获胜信息
            break  # 退出循环
        for (b = 1; b <= 2; b++):  # 循环两次
            print("\n")  # 打印空行
# 打印换行符
print("\n");
# 打印玩家编号
print("PLAYER " + b + " ");
# 进入无限循环，直到满足条件跳出循环
while (1) {
    # 打印提示信息
    print("X,Y\n");
    # 等待用户输入
    str = await input();
    # 将输入的字符串转换为整数并赋值给ya[b]
    ya[b] = parseInt(str);
    # 从逗号后的位置开始截取字符串并转换为整数，赋值给xa[b]
    xa[b] = parseInt(str.substr(str.indexOf(",") + 1));
    # 如果输入的坐标在合法范围内且对应位置的值为0，则跳出循环
    if (xa[b] > 0 && xa[b] < 6 && ya[b] > 0 && ya[b] < 5 && na[xa[b]][ya[b]] == 0)
        break;
    # 打印提示信息
    print("ILLEGAL COORDS. RETYPE\n");
}
# 如果玩家编号不为1
if (b != 1) {
    # 如果两个玩家的坐标相同
    if (xa[1] == xa[2] && ya[1] == ya[2]) {
        # 打印提示信息
        print("SAME COORD.  SET TO 0\n");
        # 将对应位置的值设为0
        na[xa[b] + 1][ya[b] + 1] = 0;
        # 将玩家编号设为99
        b = 99;
    }
}
# 如果玩家编号为99，则跳出循环
if (b == 99)
    break;
        }  # 结束 if 语句块
        if (b <= 2) {  # 如果 b 小于等于 2
            na[x[1]][y[1]] = 100;  # 将数组 na 中索引为 x[1] 和 y[1] 的位置赋值为 100
            na[x[2]][y[2]] = 1000;  # 将数组 na 中索引为 x[2] 和 y[2] 的位置赋值为 1000
        }  # 结束 if 语句块
    }  # 结束 for 循环
}

main();  # 调用 main 函数
```