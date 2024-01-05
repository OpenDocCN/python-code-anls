# `d:/src/tocomm/basic-computer-games\65_Nim\javascript\nim.js`

```
// NIM
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    // 在页面输出指定的字符串
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    // 返回一个 Promise 对象，用于异步处理用户输入
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 在页面输出提示符
                       print("? ");
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键事件监听器
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入字符串设置为输入元素的值
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析输入字符串
        resolve(input_str);
    }
});
# 结束事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        a[i] += 1;
        b[i][j] = 0;
    }
    d[i] = 0;
}

// Function to print a tab
function tab(n) {
    var str = "";
    for (var i = 0; i < n; i++) {
        str += " ";
    }
    return str;
}

var a = [];
var b = [];
var d = [];

// Main program
async function main()
{
    print(tab(33) + "NIM\n");  // 打印 33 个空格，然后打印 "NIM"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印 15 个空格，然后打印 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印一个空行
    print("\n");  // 打印一个空行
    print("\n");  // 打印一个空行
    for (i = 1; i <= 100; i++) {  // 循环 100 次
        a[i] = 0;  // 将 a[i] 的值设为 0
        b[i] = [];  // 将 b[i] 初始化为空数组
        for (j = 0; j <= 10; j++)  // 循环 11 次
        {
            a[i] += 1;  // a[i] 的值加 1
            b[i][j] = 0;  // 将 b[i][j] 的值设为 0
        }
        d[i] = 0;  // 将 d[i] 的值设为 0
    }
}
            b[i][j] = 0;  # 将二维数组 b 的第 i 行第 j 列的元素赋值为 0
    }
    d[0] = 0;  # 将数组 d 的第一个元素赋值为 0
    d[1] = 0;  # 将数组 d 的第二个元素赋值为 0
    d[2] = 0;  # 将数组 d 的第三个元素赋值为 0
    print("DO YOU WANT INSTRUCTIONS");  # 打印提示信息
    while (1) {  # 进入无限循环
        str = await input();  # 等待用户输入并将输入的字符串赋值给变量 str
        str = str.toUpperCase();  # 将 str 转换为大写
        if (str == "YES" || str == "NO")  # 如果 str 等于 "YES" 或者 "NO"，则跳出循环
            break;
        print("PLEASE ANSWER YES OR NO\n");  # 打印提示信息
    }
    if (str == "YES") {  # 如果 str 等于 "YES"
        print("THE GAME IS PLAYED WITH A NUMBER OF PILES OF OBJECTS.\n");  # 打印游戏说明
        print("ANY NUMBER OF OBJECTS ARE REMOVED FROM ONE PILE BY YOU AND\n");  # 打印游戏说明
        print("THE MACHINE ALTERNATELY.  ON YOUR TURN, YOU MAY TAKE\n");  # 打印游戏说明
        print("ALL THE OBJECTS THAT REMAIN IN ANY PILE, BUT YOU MUST\n");  # 打印游戏说明
        print("TAKE AT LEAST ONE OBJECT, AND YOU MAY TAKE OBJECTS FROM\n");  # 打印游戏说明
        print("ONLY ONE PILE ON A SINGLE TURN.  YOU MUST SPECIFY WHETHER\n");  # 打印游戏说明
        # 打印游戏规则说明
        print("WINNING IS DEFINED AS TAKING OR NOT TAKING THE LAST OBJECT,\n");
        print("THE NUMBER OF PILES IN THE GAME, AND HOW MANY OBJECTS ARE\n");
        print("ORIGINALLY IN EACH PILE.  EACH PILE MAY CONTAIN A\n");
        print("DIFFERENT NUMBER OF OBJECTS.\n");
        print("THE MACHINE WILL SHOW ITS MOVE BY LISTING EACH PILE AND THE\n");
        print("NUMBER OF OBJECTS REMAINING IN THE PILES AFTER  EACH OF ITS\n");
        print("MOVES.\n");
    }
    # 游戏开始
    while (1) {
        print("\n");
        # 输入玩家选择的胜利条件
        while (1) {
            print("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST");
            w = parseInt(await input());
            if (w == 1 || w == 2)
                break;
        }
        # 输入游戏中的堆数
        while (1) {
            print("ENTER NUMBER OF PILES");
            n = parseInt(await input());
            if (n >= 1 && n <= 100)
        break;  // 结束循环
        }
        print("ENTER PILE SIZES\n");  // 打印提示信息
        for (i = 1; i <= n; i++) {  // 循环n次
            while (1) {  // 进入无限循环
                print(i + " ");  // 打印当前i的值
                a[i] = parseInt(await input());  // 将输入的值转换为整数并赋给a[i]
                if (a[i] >= 1 && a[i] <= 2000)  // 如果a[i]的值在1到2000之间
                    break;  // 结束循环
            }
        }
        print("DO YOU WANT TO MOVE FIRST");  // 打印提示信息
        while (1) {  // 进入无限循环
            str = await input();  // 获取输入的字符串
            str = str.toUpperCase();  // 将字符串转换为大写
            if (str == "YES" || str == "NO")  // 如果输入为"YES"或"NO"
                break;  // 结束循环
            print("PLEASE ANSWER YES OR NO.\n");  // 打印提示信息
        }
        if (str == "YES")  // 如果输入为"YES"
            player_first = true;  # 设置变量 player_first 为 true
        else
            player_first = false;  # 设置变量 player_first 为 false
        while (1) {  # 进入无限循环
            if (!player_first) {  # 如果 player_first 为 false
                if (w != 1) {  # 如果 w 不等于 1
                    c = 0;  # 初始化变量 c 为 0
                    for (i = 1; i <= n; i++) {  # 循环 i 从 1 到 n
                        if (a[i] == 0)  # 如果 a[i] 等于 0
                            continue;  # 跳过本次循环
                        c++;  # 变量 c 自增 1
                        if (c == 3)  # 如果 c 等于 3
                            break;  # 跳出循环
                        d[c] = i;  # 将 i 赋值给数组 d 的第 c 个元素
                    }
                    if (i > n) {  # 如果 i 大于 n
                        if (c == 2) {  # 如果 c 等于 2
                            if (a[d[1]] == 1 || a[d[2]] == 1) {  # 如果 a[d[1]] 等于 1 或者 a[d[2]] 等于 1
                                print("MACHINE WINS\n");  # 打印 "MACHINE WINS" 并换行
                                break;  # 跳出循环
                        }
                    } else {
                        # 初始化变量 c 为 0
                        c = 0;
                        # 遍历 i 从 1 到 n
                        for (i = 1; i <= n; i++) {
                            # 如果 a[i] 大于 1，则跳出循环
                            if (a[i] > 1)
                                break;
                            # 如果 a[i] 等于 0，则继续下一次循环
                            if (a[i] == 0)
                                continue;
                            # 变量 c 自增
                            c++;
                        }
                        # 如果 i 大于 n 并且 c 为奇数
                        if (i > n && c % 2) {
                            # 打印 "MACHINE LOSES"
                            print("MACHINE LOSES\n");
                break;  # 结束当前循环
            }
        }
    }
    for (i = 1; i <= n; i++) {  # 遍历 i 从 1 到 n
        e = a[i];  # 将 a[i] 赋值给 e
        for (j = 0; j <= 10; j++) {  # 遍历 j 从 0 到 10
            f = e / 2;  # 计算 e 除以 2 的结果赋值给 f
            b[i][j] = 2 * (f - Math.floor(f));  # 计算并赋值给 b[i][j]
            e = Math.floor(f);  # 将 f 向下取整后的值赋值给 e
        }
    }
    for (j = 10; j >= 0; j--) {  # 逆序遍历 j 从 10 到 0
        c = 0;  # 初始化 c 为 0
        h = 0;  # 初始化 h 为 0
        for (i = 1; i <= n; i++) {  # 遍历 i 从 1 到 n
            if (b[i][j] == 0)  # 如果 b[i][j] 等于 0
                continue;  # 继续下一次循环
            c++;  # c 自增 1
            if (a[i] <= h)  # 如果 a[i] 小于等于 h
# 继续执行下一次循环
continue;
# 将数组a中索引为i的元素赋值给变量h
h = a[i];
# 将变量i的值赋给变量g
g = i;
# 如果c除以2的余数不为0，则跳出循环
if (c % 2)
    break;
# 如果j小于0，则执行以下操作
if (j < 0) {
    # 执行以下操作直到a中索引为e的元素不为0
    do {
        # 将n乘以一个随机数加1后向下取整，赋值给变量e
        e = Math.floor(n * Math.random() + 1);
    } while (a[e] == 0) ;
    # 将a中索引为e的元素乘以一个随机数加1后向下取整，赋值给变量f
    f = Math.floor(a[e] * Math.random() + 1);
    # 将a中索引为e的元素减去f
    a[e] -= f;
# 否则执行以下操作
} else {
    # 将a中索引为g的元素赋值为0
    a[g] = 0;
    # 对于j从0到10的每一个值，执行以下操作
    for (j = 0; j <= 10; j++) {
        # 将b中索引为g和j的元素赋值为0
        b[g][j] = 0;
        # 将c赋值为0
        c = 0;
        # 对于i从1到n的每一个值，执行以下操作
        for (i = 1; i <= n; i++) {
            # 如果b中索引为i和j的元素为0
            if (b[i][j] == 0)
                                continue;  # 继续执行下一次循环
                            c++;  # 变量 c 自增 1
                        }
                        a[g] = a[g] + (c % 2) * Math.pow(2, j);  # 根据条件计算并赋值给数组 a[g]
                    }
                    if (w != 1) {  # 如果 w 不等于 1
                        c = 0;  # 变量 c 赋值为 0
                        for (i = 1; i <= n; i++) {  # 循环遍历 i 从 1 到 n
                            if (a[i] > 1)  # 如果数组 a[i] 大于 1
                                break;  # 跳出循环
                            if (a[i] == 0)  # 如果数组 a[i] 等于 0
                                continue;  # 继续执行下一次循环
                            c++;  # 变量 c 自增 1
                        }
                        if (i > n && c % 2 == 0)  # 如果 i 大于 n 且 c 除以 2 的余数等于 0
                            a[g] = 1 - a[g];  # 根据条件计算并赋值给数组 a[g]
                    }
                }
                print("PILE  SIZE\n");  # 打印输出字符串 "PILE  SIZE"
                for (i = 1; i <= n; i++)  # 循环遍历 i 从 1 到 n
# 打印当前循环中的 i 和 a[i] 的值
print(" " + i + "  " + a[i] + "\n");
# 如果 w 不等于 2，则执行以下代码块
if (w != 2) {
    # 如果游戏已完成，则打印"MACHINE WINS"并跳出循环
    if (game_completed()) {
        print("MACHINE WINS");
        break;
    }
}
# 如果上述条件不满足，则将 player_first 设为 false
else {
    player_first = false;
}
# 进入无限循环
while (1) {
    # 打印提示信息，等待用户输入
    print("YOUR MOVE - PILE , NUMBER TO BE REMOVED");
    # 获取用户输入的字符串
    str = await input();
    # 将输入的字符串转换为整数并赋值给 x
    x = parseInt(str);
    # 从输入的字符串中获取逗号后面的部分，转换为整数并赋值给 y
    y = parseInt(str.substr(str.indexOf(",") + 1));
    # 如果 x 不在合法范围内，则继续循环
    if (x < 1 || x > n)
        continue;
    # 如果 y 不在合法范围内，则继续循环
    if (y < 1 || y > a[x])
        continue;
    # 如果上述条件都满足，则跳出循环
    break;
            }  # 结束 if 语句块
            a[x] -= y;  # 数组 a 中索引为 x 的元素减去 y
            if (game_completed()) {  # 如果游戏已经完成
                print("MACHINE LOSES");  # 打印 "MACHINE LOSES"
                break;  # 退出循环
            }
        }  # 结束 while 循环
        print("DO YOU WANT TO PLAY ANOTHER GAME");  # 打印 "DO YOU WANT TO PLAY ANOTHER GAME"
        while (1) {  # 进入无限循环
            str = await input();  # 等待用户输入并将输入赋值给变量 str
            str = str.toUpperCase();  # 将 str 转换为大写
            if (str == "YES" || str == "NO")  # 如果 str 等于 "YES" 或 "NO"
                break;  # 退出循环
            print("PLEASE ANSWER YES OR NO.\n");  # 打印提示信息
        }
        if (str == "NO")  # 如果 str 等于 "NO"
            break;  # 退出循环
    }  # 结束外层 while 循环
}  # 结束函数
# 定义一个名为 game_completed 的函数，用于检查游戏是否完成
def game_completed():
    # 使用 for 循环遍历数组 a 中的元素
    for i in range(1, n+1):
        # 如果数组 a 中的元素不等于 0，则返回 false
        if a[i] != 0:
            return False
    # 如果数组 a 中的所有元素都等于 0，则返回 true
    return True

# 调用主函数 main
main()
```