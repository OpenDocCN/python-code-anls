# `d:/src/tocomm/basic-computer-games\18_Bullseye\javascript\bullseye.js`

```
// 定义一个名为print的函数，用于向页面输出内容
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input() {
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
```

在这个示例中，我们为JavaScript代码添加了注释，解释了每个语句的作用。这样做可以帮助其他程序员更容易地理解代码，并且在以后需要修改或维护代码时也能更快地找到需要的部分。
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
        # 解析输入字符串
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
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

var as = [];  // 声明一个空数组as
var s = [];  // 声明一个空数组s
var w = [];  // 声明一个空数组w

// Main program
async function main()  // 声明一个异步函数main
{
    print(tab(32) + "BULLSEYE\n");  // 打印带有32个空格的字符串和"BULLSEYE"，并换行
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行
    print("\n");  // 打印一个空行
    print("\n");  // 打印一个空行
    print("\n");  // 打印一个空行
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET\n");  // 打印游戏说明
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS\n");  // 打印游戏说明
    print("TO GET 200 POINTS.\n");  // 打印游戏说明
    print("\n");  // 打印一个空行
    # 打印标题行
    print("THROW\t\tDESCRIPTION\t\tPROBABLE SCORE\n");
    # 打印不同投掷方式的描述和可能得分
    print("1\t\tFAST OVERARM\t\tBULLSEYE OR COMPLETE MISS\n");
    print("2\t\tCONTROLLED OVERARM\t10, 20 OR 30 POINTS\n");
    print("3\t\tUNDERARM\t\tANYTHING\n");
    print("\n");
    # 初始化变量 m 和 r
    m = 0;
    r = 0;
    # 初始化数组 s 的值为 0
    for (i = 1; i <= 20; i++)
        s[i] = 0;
    # 获取玩家数量并存储在变量 n 中
    print("HOW MANY PLAYERS");
    n = parseInt(await input());
    print("\n");
    # 获取每个玩家的名字并存储在数组 as 中
    for (i = 1; i <= n; i++) {
        print("NAME OF PLAYER #" + i);
        as[i] = await input();
    }
    # 开始游戏循环
    do {
        r++;
        # 打印当前回合数
        print("\n");
        print("ROUND " + r + "\n");
        # 打印分隔线
        print("---------\n")
        # 循环遍历 i 从 1 到 n
        for (i = 1; i <= n; i++) {
            # 执行以下操作直到 t 的值在 1 到 3 之间
            do {
                # 打印换行
                print("\n")
                # 打印 as[i] 的值加上 "'S THROW"
                print(as[i] + "'S THROW")
                # 将输入转换为整数并赋值给 t
                t = parseInt(await input())
                # 如果 t 小于 1 或者大于 3，则打印提示信息
                if (t < 1 || t > 3)
                    print("INPUT 1, 2, OR 3!\n")
            } while (t < 1 || t > 3) ;
            # 如果 t 等于 1，则设置 p1、p2、p3、p4 的值
            if (t == 1) {
                p1 = 0.65
                p2 = 0.55
                p3 = 0.5
                p4 = 0.5
            # 如果 t 等于 2，则设置 p1、p2、p3、p4 的值
            } else if (t == 2) {
                p1 = 0.99
                p2 = 0.77
                p3 = 0.43
                p4 = 0.01
            # 如果 t 不等于 1 或 2，则执行以下操作
            } else {
                p1 = 0.95;  // 设置变量 p1 的值为 0.95
                p2 = 0.75;  // 设置变量 p2 的值为 0.75
                p3 = 0.45;  // 设置变量 p3 的值为 0.45
                p4 = 0.05;  // 设置变量 p4 的值为 0.05
            }
            u = Math.random();  // 生成一个 0 到 1 之间的随机数，并赋值给变量 u
            if (u >= p1) {  // 如果 u 大于等于 p1
                print("BULLSEYE!!  40 POINTS!\n");  // 打印字符串 "BULLSEYE!!  40 POINTS!"
                b = 40;  // 设置变量 b 的值为 40
            } else if (u >= p2) {  // 如果 u 大于等于 p2
                print("30-POINT ZONE!\n");  // 打印字符串 "30-POINT ZONE!"
                b = 30;  // 设置变量 b 的值为 30
            } else if (u >= p3) {  // 如果 u 大于等于 p3
                print("20-POINT ZONE\n");  // 打印字符串 "20-POINT ZONE"
                b = 20;  // 设置变量 b 的值为 20
            } else if (u >= p4) {  // 如果 u 大于等于 p4
                print("WHEW!  10 POINT.\n");  // 打印字符串 "WHEW!  10 POINT."
                b = 10;  // 设置变量 b 的值为 10
            } else {  // 如果以上条件都不满足
                print("MISSED THE TARGET!  TOO BAD.\n");  // 打印字符串 "MISSED THE TARGET!  TOO BAD."
                b = 0;  # 初始化变量 b 为 0
            }
            s[i] += b;  # 将变量 b 的值加到数组 s 的第 i 个元素上
            print("TOTAL SCORE = " + s[i] + "\n");  # 打印输出总分
        }
        for (i = 1; i <= n; i++) {  # 循环遍历数组 s
            if (s[i] >= 200) {  # 如果数组 s 的第 i 个元素大于等于 200
                m++;  # 变量 m 自增 1
                w[m] = i;  # 将 i 赋值给数组 w 的第 m 个元素
            }
        }
    } while (m == 0) ;  # 当 m 等于 0 时继续循环
    print("\n");  # 打印输出空行
    print("WE HAVE A WINNER!!\n");  # 打印输出“我们有一个赢家！”
    print("\n");  # 打印输出空行
    for (i = 1; i <= m; i++)  # 循环遍历数组 w
        print(as[w[i]] + " SCORED " + s[w[i]] + " POINTS.\n");  # 打印输出每个赢家的得分
    print("\n");  # 打印输出空行
    print("THANKS FOR THE GAME.\n");  # 打印输出“谢谢参与游戏。”
}
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，可能是一个错误。
```