# `d:/src/tocomm/basic-computer-games\80_Slots\javascript\slots.js`

```
// 定义一个名为print的函数，用于在页面上输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input()
{
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
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键按下事件监听器
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
# 结束添加事件监听器的函数
});
}

# 定义一个函数 tab，参数为 space
function tab(space)
{
    # 初始化一个空字符串 str
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

var figures = [, "BAR", "BELL", "ORANGE", "LEMON", "PLUM", "CHERRY"];  // 定义一个包含不同图案的数组

// 主程序
async function main()
{
    print(tab(30) + "SLOTS\n");  // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印信息
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    // Produced by Fred Mirabelle and Bob Harper on Jan 29, 1973
    // It simulates the slot machine.
    print("YOU ARE IN THE H&M CASINO,IN FRONT ON ONE OF OUR\n");  // 打印信息
    print("ONE-ARM BANDITS. BET FROM $1 TO $100.\n");  // 打印信息
    print("TO PULL THE ARM, PUNCH THE RETURN KEY AFTER MAKING YOUR BET.\n");  // 打印信息
    p = 0;  // 初始化变量p为0
}
    while (1) {  # 进入无限循环
        while (1) {  # 进入内部无限循环
            print("\n");  # 打印空行
            print("YOUR BET");  # 打印提示信息
            m = parseInt(await input());  # 从输入中获取赌注并转换为整数赋值给变量m
            if (m > 100) {  # 如果赌注大于100
                print("HOUSE LIMITS ARE $100\n");  # 打印提示信息
            } else if (m < 1) {  # 如果赌注小于1
                print("MINIMUM BET IS $1\n");  # 打印提示信息
            } else {  # 如果赌注在1到100之间
                break;  # 跳出内部循环
            }
        }
        // Not implemented: GOSUB 1270 ten chimes  # 未实现的功能，调用子程序1270，发出十次钟声
        print("\n");  # 打印空行
        x = Math.floor(6 * Math.random() + 1);  # 生成1到6之间的随机整数并赋值给变量x
        y = Math.floor(6 * Math.random() + 1);  # 生成1到6之间的随机整数并赋值给变量y
        z = Math.floor(6 * Math.random() + 1);  # 生成1到6之间的随机整数并赋值给变量z
        print("\n");  # 打印空行
        // Not implemented: GOSUB 1310 seven chimes after figure x and y  # 未实现的功能，调用子程序1310，在x和y之后发出七次钟声
        # 打印出三个数字的组合
        print(figures[x] + " " + figures[y] + " " + figures[z] + "\n");
        # 重置 lost 变量为 false
        lost = false;
        # 如果三个数字相等
        if (x == y && y == z) {  // Three figure
            # 打印空行
            print("\n");
            # 如果 z 不等于 1
            if (z != 1) {
                # 打印 **TOP DOLLAR**
                print("**TOP DOLLAR**\n");
                # 更新玩家赢得的奖金
                p += ((10 * m) + m);
            } else {
                # 打印 ***JACKPOT***
                print("***JACKPOT***\n");
                # 更新玩家赢得的奖金
                p += ((100 * m) + m);
            }
            # 打印 YOU WON!
            print("YOU WON!\n");
        } else if (x == y || y == z || x == z) {
            # 如果有两个数字相等
            if (x == y)
                # 将 c 设置为相等的数字
                c = x;
            else
                c = z;
            # 如果 c 等于 1
            if (c == 1) {
                # 打印空行
                print("\n");
                # 打印 *DOUBLE BAR*
                print("YOU WON\n");  # 打印出玩家赢了的消息
                p += ((5 * m) + m);  # 玩家的赌注增加了5倍加上原本的赌注
            } else if (x != z) {  # 如果玩家的选择不等于骰子的点数
                print("\n");  # 打印一个空行
                print("DOUBLE!!\n");  # 打印出玩家获得了双倍的消息
                print("YOU WON!\n");  # 打印出玩家赢了的消息
                p += ((2 * m) + m);  # 玩家的赌注增加了2倍加上原本的赌注
            } else {  # 如果玩家的选择和骰子的点数相等
                lost = true;  # 玩家输了
            }
        } else {  # 如果玩家的选择不在1到6之间
            lost = true;  # 玩家输了
        }
        if (lost) {  # 如果玩家输了
            print("\n");  # 打印一个空行
            print("YOU LOST.\n");  # 打印出玩家输了的消息
            p -= m;  # 玩家的赌注减去原本的赌注
        }
        print("YOUR STANDINGS ARE $" + p + "\n");  # 打印出玩家的赌注余额
        print("AGAIN");  # 打印出再玩一次的提示
# 从输入中获取字符串
str = await input();
# 如果字符串的第一个字符不是 "Y"，则跳出循环
if (str.substr(0, 1) != "Y")
    break;
# 打印换行符
print("\n");
# 如果 p 小于 0，则打印支付信息
if (p < 0) {
    print("PAY UP!  PLEASE LEAVE YOUR MONEY ON THE TERMINAL.\n");
# 如果 p 等于 0，则打印平局信息
} else if (p == 0) {
    print("HEY, YOU BROKE EVEN.\n");
# 如果 p 大于 0，则打印赢取奖金信息
} else {
    print("COLLECT YOUR WINNINGS FROM THE H&M CASHIER.\n");
}

# 调用主函数
main();
```