# `30_Cube\javascript\cube.js`

```
// 定义一个名为print的函数，用于向页面输出文本
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在页面上输出问号提示
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
        # 解析输入字符串
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
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "CUBE\n");  // 打印带有缩进的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("DO YOU WANT TO SEE THE INSTRUCTIONS? (YES--1,NO--0)");  // 打印提示信息
    b7 = parseInt(await input());  // 将用户输入的字符串转换为整数并赋值给变量b7
    if (b7 != 0) {  // 如果b7不等于0
        print("THIS IS A GAME IN WHICH YOU WILL BE PLAYING AGAINST THE\n");  // 打印游戏说明
        print("RANDOM DECISION OF THE COMPUTER. THE FIELD OF PLAY IS A\n");  // 打印游戏说明
        print("CUBE OF SIDE 3. ANY OF THE 27 LOCATIONS CAN BE DESIGNATED\n");  // 打印游戏说明
        print("BY INPUTING THREE NUMBERS SUCH AS 2,3,1. AT THE START,\n");  // 打印游戏说明
        print("YOU ARE AUTOMATICALLY AT LOCATION 1,1,1. THE OBJECT OF\n");  // 打印游戏说明
# 打印游戏规则和提示信息
print("THE GAME IS TO GET TO LOCATION 3,3,3. ONE MINOR DETAIL:\n");
print("THE COMPUTER WILL PICK, AT RANDOM, 5 LOCATIONS AT WHICH\n");
print("IT WILL PLANT LAND MINES. IF YOU HIT ONE OF THESE LOCATIONS\n");
print("YOU LOSE. ONE OTHER DETAIL: YOU MAY MOVE ONLY ONE SPACE \n");
print("IN ONE DIRECTION EACH MOVE. FOR  EXAMPLE: FROM 1,1,2 YOU\n");
print("MAY MOVE TO 2,1,2 OR 1,1,3. YOU MAY NOT CHANGE\n");
print("TWO OF THE NUMBERS ON THE SAME MOVE. IF YOU MAKE AN ILLEGAL\n");
print("MOVE, YOU LOSE AND THE COMPUTER TAKES THE MONEY YOU MAY\n");
print("HAVE BET ON THAT ROUND.\n");
print("\n");
print("\n");
print("ALL YES OR NO QUESTIONS WILL BE ANSWERED BY A 1 FOR YES\n");
print("OR A 0 (ZERO) FOR NO.\n");
print("\n");
print("WHEN STATING THE AMOUNT OF A WAGER, PRINT ONLY THE NUMBER\n");
print("OF DOLLARS (EXAMPLE: 250)  YOU ARE AUTOMATICALLY STARTED WITH\n");
print("500 DOLLARS IN YOUR ACCOUNT.\n");
print("\n");
print("GOOD LUCK!\n");
    a1 = 500;  // 初始化变量a1为500

    while (1) {  // 进入无限循环
        a = Math.floor(3 * Math.random());  // 生成一个0到2之间的随机整数赋值给变量a
        if (a == 0)  // 如果a等于0
            a = 3;  // 将a赋值为3
        b = Math.floor(3 * Math.random());  // 生成一个0到2之间的随机整数赋值给变量b
        if (b == 0)  // 如果b等于0
            b = 2;  // 将b赋值为2
        c = Math.floor(3 * Math.random());  // 生成一个0到2之间的随机整数赋值给变量c
        if (c == 0)  // 如果c等于0
            c = 3;  // 将c赋值为3
        d = Math.floor(3 * Math.random());  // 生成一个0到2之间的随机整数赋值给变量d
        if (d == 0)  // 如果d等于0
            d = 1;  // 将d赋值为1
        e = Math.floor(3 * Math.random());  // 生成一个0到2之间的随机整数赋值给变量e
        if (e == 0)  // 如果e等于0
            e = 3;  // 将e赋值为3
        f = Math.floor(3 * Math.random());  // 生成一个0到2之间的随机整数赋值给变量f
        if (f == 0)  // 如果f等于0
            f = 3;  // 将f赋值为3
# 生成一个 0-2 之间的随机整数，并赋值给变量 g
g = Math.floor(3 * Math.random());
# 如果 g 的值为 0，则将其重新赋值为 3
if (g == 0)
    g = 3;
# 生成一个 0-2 之间的随机整数，并赋值给变量 h
h = Math.floor(3 * Math.random());
# 如果 h 的值为 0，则将其重新赋值为 3
if (h == 0)
    h = 3;
# 生成一个 0-2 之间的随机整数，并赋值给变量 i
i = Math.floor(3 * Math.random());
# 如果 i 的值为 0，则将其重新赋值为 2
if (i == 0)
    i = 2;
# 生成一个 0-2 之间的随机整数，并赋值给变量 j
j = Math.floor(3 * Math.random());
# 如果 j 的值为 0，则将其重新赋值为 3
if (j == 0)
    j = 3;
# 生成一个 0-2 之间的随机整数，并赋值给变量 k
k = Math.floor(3 * Math.random());
# 如果 k 的值为 0，则将其重新赋值为 2
if (k == 0)
    k = 2;
# 生成一个 0-2 之间的随机整数，并赋值给变量 l
l = Math.floor(3 * Math.random());
# 如果 l 的值为 0，则将其重新赋值为 3
if (l == 0)
    l = 3;
# 生成一个 0-2 之间的随机整数，并赋值给变量 m
m = Math.floor(3 * Math.random());
# 如果 m 的值为 0，则将其重新赋值为 3
if (m == 0)
    m = 3;
            m = 3;  // 初始化变量 m 为 3
        n = Math.floor(3 * Math.random());  // 生成一个 0 到 2 之间的随机整数赋值给变量 n
        if (n == 0)  // 如果 n 等于 0
            n = 1;  // 将 n 的值设为 1
        o = Math.floor(3 * Math.random());  // 生成一个 0 到 2 之间的随机整数赋值给变量 o
        if (o == 0)  // 如果 o 等于 0
            o = 3;  // 将 o 的值设为 3
        print("WANT TO MAKE A WAGER?");  // 打印提示信息
        z = parseInt(await input());  // 从输入中获取一个整数值并赋给变量 z
        if (z != 0) {  // 如果 z 不等于 0
            print("HOW MUCH ");  // 打印提示信息
            while (1) {  // 进入循环，条件为永远为真
                z1 = parseInt(await input());  // 从输入中获取一个整数值并赋给变量 z1
                if (a1 < z1) {  // 如果 a1 小于 z1
                    print("TRIED TO FOOL ME; BET AGAIN");  // 打印提示信息
                } else {  // 否则
                    break;  // 退出循环
                }
            }
        }
        w = 1;  # 初始化变量 w 为 1
        x = 1;  # 初始化变量 x 为 1
        y = 1;  # 初始化变量 y 为 1
        print("\n");  # 打印换行符
        print("IT'S YOUR MOVE:  ");  # 打印提示信息
        while (1) {  # 进入无限循环
            str = await input();  # 从输入中获取字符串
            p = parseInt(str);  # 将字符串转换为整数并赋值给变量 p
            q = parseInt(str.substr(str.indexOf(",") + 1));  # 从字符串中截取子字符串并转换为整数赋值给变量 q
            r = parseInt(str.substr(str.lastIndexOf(",") + 1));  # 从字符串中截取子字符串并转换为整数赋值给变量 r
            if (p > w + 1 || q > x + 1 || r > y + 1 || (p == w + 1 && (q >= x + 1 || r >= y + 1)) || (q == x + 1 && r >= y + 1)) {  # 判断条件
                print("\n");  # 打印换行符
                print("ILLEGAL MOVE, YOU LOSE.\n");  # 打印提示信息
                break;  # 跳出循环
            }
            w = p;  # 更新变量 w 的值为 p
            x = q;  # 更新变量 x 的值为 q
            y = r;  # 更新变量 y 的值为 r
            if (p == 3 && q == 3 && r == 3) {  # 判断条件
                won = true;  # 将变量 won 设置为 true
                break;  # 结束当前循环，跳出循环体
            }
            if (p == a && q == b && r == c  # 如果 p、q、r 分别等于 a、b、c
             || p == d && q == e && r == f  # 或者 p、q、r 分别等于 d、e、f
             || p == g && q == h && r == i  # 或者 p、q、r 分别等于 g、h、i
             || p == j && q == k && r == l  # 或者 p、q、r 分别等于 j、k、l
             || p == m && q == n && r == o):  # 或者 p、q、r 分别等于 m、n、o
                print("******BANG******");  # 打印******BANG******
                print("YOU LOSE!");  # 打印YOU LOSE!
                print("\n");  # 打印换行
                print("\n");  # 打印换行
                won = false;  # 将won变量设为false
                break;  # 结束当前循环，跳出循环体
            }
            print("NEXT MOVE: ");  # 打印NEXT MOVE:
        }
        if (won) {  # 如果won为true
            print("CONGRATULATIONS!\n");  # 打印CONGRATULATIONS!
            if (z != 0) {  # 如果z不等于0
                z2 = a1 + z1;  # 计算z2的值为a1加上z1
                print("YOU NOW HAVE " + z2 + " DOLLARS.\n");  # 打印玩家当前的资金余额
                a1 = z2;  # 更新玩家的资金余额
            }
        } else {
            if (z != 0) {  # 如果庄家不是0
                print("\n");  # 打印空行
                z2 = a1 - z1;  # 计算玩家的新资金余额
                if (z2 <= 0) {  # 如果玩家的资金余额小于等于0
                    print("YOU BUST.\n");  # 打印玩家爆牌
                    break;  # 结束游戏
                } else {
                    print(" YOU NOW HAVE " + z2 + " DOLLARS.\n");  # 打印玩家当前的资金余额
                    a1 = z2;  # 更新玩家的资金余额
                }
            }
        }
        print("DO YOU WANT TO TRY AGAIN ");  # 打印询问玩家是否想再试一次
        s = parseInt(await input());  # 获取玩家的输入并转换为整数
        if (s != 1)  # 如果玩家输入不等于1
            break;  # 结束游戏
    }  # 结束 main 函数的定义

    print("TOUGH LUCK!\n");  # 打印 "TOUGH LUCK!" 到控制台
    print("\n");  # 打印一个空行到控制台
    print("GOODBYE.\n");  # 打印 "GOODBYE." 到控制台
}  # 结束程序的主体部分

main();  # 调用 main 函数作为程序的入口点
```