# `basic-computer-games\28_Combat\javascript\combat.js`

```py
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
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素的类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入元素的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
{
    // 打印游戏名称
    print(tab(33) + "COMBAT\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏提示信息
    print("I AM AT WAR WITH YOU.\n");
    print("WE HAVE 72000 SOLDIERS APIECE.\n");
    # 执行循环，直到输入的军队总人数不超过72000
    do {
        # 打印空行
        print("\n");
        # 打印提示信息
        print("DISTRIBUTE YOUR FORCES.\n");
        # 打印表头
        print("\tME\t  YOU\n");
        # 打印军队类型和数量
        print("ARMY\t30000\t");
        # 将输入的字符串转换为整数
        a = parseInt(await input());
        # 打印军队类型和数量
        print("NAVY\t20000\t");
        # 将输入的字符串转换为整数
        b = parseInt(await input());
        # 打印军队类型和数量
        print("A. F.\t22000\t");
        # 将输入的字符串转换为整数
        c = parseInt(await input());
    # 当输入的军队总人数超过72000时继续执行循环
    } while (a + b + c > 72000) ;
    # 初始化变量d、e、f
    d = 30000;
    e = 20000;
    f = 22000;
    # 打印提示信息
    print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;\n");
    # 打印提示信息
    print("AND (3) FOR AIR FORCE.\n");
    # 将输入的字符串转换为整数
    y = parseInt(await input());
    # 执行循环，直到输入的攻击人数不超过相应军队的数量
    do {
        # 打印提示信息
        print("HOW MANY MEN\n");
        # 将输入的字符串转换为整数
        x = parseInt(await input());
    # 当输入的攻击人数超过相应军队的数量时继续执行循环
    } while ((y == 1 && x > a) || (y == 2 && x > b) || (y == 3 && x > c)) ;
    # 打印空行
    print("\n");
    # 打印表头
    print("\tYOU\tME\n");
    # 打印军队类型和数量
    print("ARMY\t" + a + "\t" + d + "\n");
    # 打印军队类型和数量
    print("NAVY\t" + b + "\t" + e + "\n");
    # 打印军队类型和数量
    print("A. F.\t" + c + "\t" + f + "\n");
    # 打印提示信息
    print("WHAT IS YOUR NEXT MOVE?\n");
    # 打印提示信息
    print("ARMY=1  NAVY=2  AIR FORCE=3\n");
    # 将输入的字符串转换为整数
    g = parseInt(await input());
    # 执行循环，直到输入的调动人数不超过相应军队的数量或小于0
    do {
        # 打印提示信息
        print("HOW MANY MEN\n");
        # 将输入的字符串转换为整数
        t = parseInt(await input());
    # 当输入的调动人数小于0或超过相应军队的数量时继续执行循环
    } while (t < 0 || (g == 1 && t > a) || (g == 2 && t > b) || (g == 3 && t > c)) ;
    # 初始化变量crashed
    crashed = false;
    # 根据变量 g 的值进行不同的操作
    switch (g) {
        # 如果 g 等于 1
        case 1:
            # 如果 t 小于 d 的一半
            if (t < d / 2) {
                # 打印信息并更新变量 a
                print("I WIPED OUT YOUR ATTACK!\n");
                a -= t;
            } else {
                # 否则打印信息并更新变量 d
                print("YOU DESTROYED MY ARMY!\n");
                d = 0;
            }
            # 结束 case 1
            break;
        # 如果 g 等于 2
        case 2:
            # 如果 t 小于 e 的一半
            if (t < e / 2) {
                # 打印信息并更新变量 a 和 b
                print("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE\n");
                print("WIPED OUT YOUR UNGUARDED CAPITOL.\n");
                a /= 4.0;
                b /= 2.0;
                # 结束 case 2
                break;
            }
            # 否则打印信息并更新变量 f 和 e
            print("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES.\n");
            print("AND SUNK THREE BATTLESHIPS.\n");
            f = 2 * f / 3;
            e /= 2;
            # 结束 case 2
            break;
        # 如果 g 等于 3
        case 3:
            # 如果 t 大于 f 的一半
            if (t > f / 2) {
                # 打印信息并更新变量 a, b, 和 c
                print("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT\n");
                print("YOUR COUNTRY IN SHAMBLES.\n");
                a /= 3.0;
                b /= 3.0;
                c /= 3.0;
                # 结束 case 3
                break;
            }
            # 否则打印信息并更新变量 crashed 和 won
            print("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.\n");
            print("MY COUNTRY FELL APART.\n");
            crashed = true;
            won = 1;
            # 结束 case 3
            break;
    }
    # 如果没有发生坠毁
    if (!crashed) {
        # 重置变量 won
        won = 0;
        # 打印信息
        print("\n");
        print("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,\n");
        # 根据条件更新变量 won
        if (a + b + c > 3.0 / 2.0 * (d + e + f))
            won = 1;
        if (a + b + c < 2.0 / 3.0 * (d + e + f))
            won = 2;
    }
    # 根据 won 的值打印不同的信息
    if (won == 0) {
        print("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR\n");
        print("RESPECTIVE COUNTRIES AND LIVE IN PEACE.\n");
    } else if (won == 1) {
        print("YOU WON, OH! SHUCKS!!!!\n");
    } else if (won == 2) {
        print("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU\n");
        print("RIGHT FOR PLAYING THIS STUPID GAME!!!\n");
    }
# 结束当前的代码块
}

# 调用名为main的函数
main();
```