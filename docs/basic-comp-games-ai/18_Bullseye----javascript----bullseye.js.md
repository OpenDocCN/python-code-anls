# `basic-computer-games\18_Bullseye\javascript\bullseye.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
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
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值返回
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
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

// 初始化变量
var as = [];
var s = [];
var w = [];

// 主程序
async function main()
{
    // 打印游戏标题和介绍
    print(tab(32) + "BULLSEYE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET\n");
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS\n");
    print("TO GET 200 POINTS.\n");
    print("\n");
    print("THROW\t\tDESCRIPTION\t\tPROBABLE SCORE\n");
    print("1\t\tFAST OVERARM\t\tBULLSEYE OR COMPLETE MISS\n");
    print("2\t\tCONTROLLED OVERARM\t10, 20 OR 30 POINTS\n");
    print("3\t\tUNDERARM\t\tANYTHING\n");
    print("\n");
    m = 0;
    r = 0;
    for (i = 1; i <= 20; i++)
        s[i] = 0;
    print("HOW MANY PLAYERS");
    // 获取玩家数量
    n = parseInt(await input());
    print("\n");
    for (i = 1; i <= n; i++) {
        // 获取每个玩家的名字
        print("NAME OF PLAYER #" + i);
        as[i] = await input();
    }
    do {
        r++;
        print("\n");
        print("ROUND " + r + "\n");
        print("---------\n");
        for (i = 1; i <= n; i++) {
            do {
                print("\n");
                print(as[i] + "'S THROW");
                // 获取玩家投掷的方式
                t = parseInt(await input());
                if (t < 1 || t > 3)
                    print("INPUT 1, 2, OR 3!\n");
            } while (t < 1 || t > 3) ;
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
            u = Math.random();
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
            s[i] += b;
            print("TOTAL SCORE = " + s[i] + "\n");
        }
        for (i = 1; i <= n; i++) {
            if (s[i] >= 200) {
                m++;
                w[m] = i;
            }
        }
    } while (m == 0) ;
    print("\n");
    print("WE HAVE A WINNER!!\n");
    print("\n");
    for (i = 1; i <= m; i++)
        print(as[w[i]] + " SCORED " + s[w[i]] + " POINTS.\n");
    print("\n");
    print("THANKS FOR THE GAME.\n");
}

// 调用主程序
main();

```