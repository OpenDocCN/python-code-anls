# `basic-computer-games\18_Bullseye\javascript\bullseye.js`

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
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
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

// 定义三个空数组
var as = [];
var s = [];
var w = [];

// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(32) + "BULLSEYE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET\n");
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS\n");
    print("TO GET 200 POINTS.\n");
    print("\n");
}
    # 打印标题行
    print("THROW\t\tDESCRIPTION\t\tPROBABLE SCORE\n");
    # 打印第一种投掷方式的描述和可能得分
    print("1\t\tFAST OVERARM\t\tBULLSEYE OR COMPLETE MISS\n");
    # 打印第二种投掷方式的描述和可能得分
    print("2\t\tCONTROLLED OVERARM\t10, 20 OR 30 POINTS\n");
    # 打印第三种投掷方式的描述
    print("3\t\tUNDERARM\t\tANYTHING\n");
    # 打印空行
    print("\n");
    # 初始化变量 m 和 r
    m = 0;
    r = 0;
    # 初始化数组 s 的前 20 个元素为 0
    for (i = 1; i <= 20; i++)
        s[i] = 0;
    # 提示输入玩家数量，并将输入转换为整数赋给变量 n
    print("HOW MANY PLAYERS");
    n = parseInt(await input());
    # 打印空行
    print("\n");
    # 循环输入每个玩家的名字，并存储在数组 as 中
    for (i = 1; i <= n; i++) {
        print("NAME OF PLAYER #" + i);
        as[i] = await input();
    }
    # 循环开始，每次循环增加 r 的值
    do {
        r++;
        # 打印换行符
        print("\n");
        # 打印当前回合数
        print("ROUND " + r + "\n");
        # 打印分隔线
        print("---------\n");
        # 遍历每个玩家
        for (i = 1; i <= n; i++) {
            # 内部循环，直到输入合法的投掷值
            do {
                # 打印换行符
                print("\n");
                # 打印当前玩家的投掷提示
                print(as[i] + "'S THROW");
                # 获取用户输入的投掷值
                t = parseInt(await input());
                # 如果投掷值不在 1 到 3 之间，打印错误提示
                if (t < 1 || t > 3)
                    print("INPUT 1, 2, OR 3!\n");
            } while (t < 1 || t > 3) ;
            # 根据投掷值设置不同的概率
            if (t == 1) {
                p1 = 0.65;
                p2 = 0.55;
                p3 = 0.5;
                p4 = 0.5;
            } else if (t == 2) {
                p1 = 0.99;
                p2 = 0.77;
                p3 = 0.43;
                p4 = 0.01;
            } else {
                p1 = 0.95;
                p2 = 0.75;
                p3 = 0.45;
                p4 = 0.05;
            }
            # 生成一个随机数
            u = Math.random();
            # 根据随机数和概率判断得分
            if (u >= p1) {
                print("BULLSEYE!!  40 POINTS!\n");
                b = 40;
            } else if (u >= p2) {
                print("30-POINT ZONE!\n");
                b = 30;
            } else if (u >= p3) {
                print("20-POINT ZONE\n");
                b = 20;
            } else if (u >= p4) {
                print("WHEW!  10 POINT.\n");
                b = 10;
            } else {
                print("MISSED THE TARGET!  TOO BAD.\n");
                b = 0;
            }
            # 累加当前玩家的得分
            s[i] += b;
            # 打印当前玩家的总得分
            print("TOTAL SCORE = " + s[i] + "\n");
        }
        # 遍历每个玩家，检查是否达到 200 分
        for (i = 1; i <= n; i++) {
            if (s[i] >= 200) {
                # 如果达到 200 分，增加 m 的值，并记录玩家编号
                m++;
                w[m] = i;
            }
        }
    # 当没有玩家达到 200 分时继续循环
    } while (m == 0) ;
    # 打印游戏结束提示
    print("\n");
    print("WE HAVE A WINNER!!\n");
    print("\n");
    # 遍历每个达到 200 分的玩家，打印他们的得分
    for (i = 1; i <= m; i++)
        print(as[w[i]] + " SCORED " + s[w[i]] + " POINTS.\n");
    # 打印结束语
    print("\n");
    print("THANKS FOR THE GAME.\n");
# 结束当前的代码块
}

# 调用名为main的函数
main();
```