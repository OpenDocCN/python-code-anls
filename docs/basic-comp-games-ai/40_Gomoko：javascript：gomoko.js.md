# `d:/src/tocomm/basic-computer-games\40_Gomoko\javascript\gomoko.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  # 将空格字符添加到字符串变量str的末尾
    return str;  # 返回处理后的字符串

}

function reset_stats()  # 定义名为reset_stats的函数
{
    for (var j = 1; j <= 4; j++)  # 循环4次，将f数组的元素全部置为0
        f[j] = 0;
}

var a = [];  # 定义一个空数组a
var x;  # 定义变量x
var y;  # 定义变量y
var n;  # 定义变量n

// *** PRINT THE BOARD ***  # 打印棋盘的注释
function print_board()  # 定义名为print_board的函数
{
    for (i = 1; i <= n; i++) {  # 循环n次，i从1到n
        for (j = 1; j <= n; j++) {  # 嵌套循环，循环n次，j从1到n
            print(" " + a[i][j] + " ");  // 打印二维数组a中索引为i和j的元素，并在前后加上空格
        }
        print("\n");  // 打印换行符
    }
    print("\n");  // 打印两个换行符，表示两行之间的间隔
}

// Is valid the movement
function is_valid()
{
    if (x < 1 || x > n || y < 1 || y > n)  // 如果x或y小于1，或者大于n，则返回false
        return false;
    return true;  // 否则返回true
}

// Main program
async function main()
{
    print(tab(33) + "GOMOKO\n");  // 打印一个长度为33的制表符，后面跟着字符串"GOMOKO"和换行符
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印一个长度为15的制表符，后面跟着字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"和换行符
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 初始化一个 20x20 的二维数组，并将所有元素初始化为 0
    for (i = 0; i <= 19; i++) {
        a[i] = [];
        for (j = 0; j <= 19; j++)
            a[i][j] = 0;
    }
    # 打印游戏的欢迎信息和规则
    print("WELCOME TO THE ORIENTAL GAME OF GOMOKO.\n");
    print("\n");
    print("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE\n");
    print("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID\n");
    print("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET\n");
    print("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR\n");
    print("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED\n");
    print("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.\n");
    print("\n");
    print("THE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.\n");
    print("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n");
    print("\n");
    while (1):  # 进入无限循环
        print("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)");  # 打印提示信息，询问用户棋盘大小
        while (1):  # 进入内层无限循环
            n = parseInt(await input());  # 从用户输入中获取棋盘大小并转换为整数赋值给变量n
            if (n >= 7 && n<= 19):  # 判断棋盘大小是否在7到19之间
                break;  # 如果是，则跳出内层循环
            print("I SAID, THE MINIMUM IS 7, THE MAXIMUM IS 19.\n");  # 如果不是，打印错误提示信息
        for (i = 1; i <= n; i++):  # 循环遍历棋盘的行
            for (j = 1; j <= n; j++):  # 循环遍历棋盘的列
                a[i][j] = 0;  # 将棋盘上每个位置的值初始化为0
        print("\n");  # 打印换行
        print("WE ALTERNATE MOVES.  YOU GO FIRST...\n");  # 打印提示信息，提示玩家先行
        print("\n");  # 打印换行
        while (1):  # 进入内层无限循环
            print("YOUR PLAY (I,J)");  # 打印提示信息，询问玩家下棋的位置
            str = await input();  # 从用户输入中获取下棋位置的字符串赋值给变量str
            i = parseInt(str);  # 将字符串转换为整数赋值给变量i
            # 从字符串中提取逗号后的数字，并转换为整数
            j = parseInt(str.substr(str.indexOf(",") + 1));
            # 打印换行符
            print("\n");
            # 如果 i 等于 -1，则跳出循环
            if (i == -1)
                break;
            # 将 i 赋值给 x，将 j 赋值给 y
            x = i;
            y = j;
            # 如果坐标不合法，则打印提示信息并继续下一次循环
            if (!is_valid()) {
                print("ILLEGAL MOVE.  TRY AGAIN...\n");
                continue;
            }
            # 如果坐标对应的方格已经被占据，则打印提示信息并继续下一次循环
            if (a[i][j] != 0) {
                print("SQUARE OCCUPIED.  TRY AGAIN...\n");
                continue;
            }
            # 将坐标对应的方格标记为已占据
            a[i][j] = 1;
            # *** 计算机尝试智能移动 ***
            found = false;
            # 遍历周围的方格
            for (e = -1; e <= 1; e++) {
                for (f = -1; f <= 1; f++) {
                    # 如果条件成立
                    if (e + f - e * f == 0)
                    continue;  // 跳过当前循环，继续下一次循环
                    x = i + f;  // 计算新的 x 坐标
                    y = j + f;  // 计算新的 y 坐标
                    if (!is_valid())  // 如果新的坐标不合法
                        continue;  // 跳过当前循环，继续下一次循环
                    if (a[x][y] == 1) {  // 如果新的坐标上的值为 1
                        x = i - e;  // 计算新的 x 坐标
                        y = j - f;  // 计算新的 y 坐标
                        if (is_valid() || a[x][y] == 0)  // 如果新的坐标合法或者新的坐标上的值为 0
                            found = true;  // 设置 found 为 true
                        break;  // 跳出循环
                    }
                }
            }
            if (!found) {  // 如果 found 为 false
                // *** Computer tries a random move ***  // 计算机尝试随机移动
                do {
                    x = Math.floor(n * Math.random() + 1);  // 生成随机的 x 坐标
                    y = Math.floor(n * Math.random() + 1);  // 生成随机的 y 坐标
                } while (!is_valid() || a[x][y] != 0) ;  // 当坐标不合法或者坐标上的值不为 0 时继续循环
            }  # 结束内层循环
            a[x][y] = 2;  # 将当前玩家的位置标记为2
            print_board();  # 打印游戏棋盘
        }  # 结束当前回合
        print("\n");  # 打印空行
        print("THANKS FOR THE GAME!!\n");  # 打印感谢信息
        print("PLAY AGAIN (1 FOR YES, 0 FOR NO)");  # 提示是否再玩一局
        q = parseInt(await input());  # 获取用户输入的是否再玩一局的选择
        if (q != 1)  # 如果用户选择不再玩一局
            break;  # 退出游戏循环
    }  # 结束外层循环
}

main();  # 调用主函数开始游戏
```