# `basic-computer-games\17_Bullfight\javascript\bullfight.js`

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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入元素
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

var a;
var b;
var c;
var l;
var t;
var as;
var bs;
// 定义一个空数组
var d = [];
// 定义一个字符串数组
var ls = [, "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"];

// 定义一个返回 1 或 2 的随机整数的函数
function af(k)
{
    return Math.floor(Math.random() * 2 + 1);
}

// 定义一个返回 0 到 q 之间的随机数的函数
function cf(q)
{
    return df(q) * Math.random();
}

// 定义一个根据参数计算结果的函数
function df(q)
{
    return (4.5 + l / 6 - (d[1] + d[2]) * 2.5 + 4 * d[4] + 2 * d[5] - Math.pow(d[3], 2) / 120 - a);
}

// 设置辅助变量
function setup_helpers()
{
    // 计算 b 的值
    b = 3 / a * Math.random();
    # 如果b小于0.37，则c等于0.5
    if (b < 0.37)
        c = 0.5;
    # 如果b大于等于0.37且小于0.5，则c等于0.4
    else if (b < 0.5)
        c = 0.4;
    # 如果b大于等于0.5且小于0.63，则c等于0.3
    else if (b < 0.63)
        c = 0.3;
    # 如果b大于等于0.63且小于0.87，则c等于0.2
    else if (b < 0.87)
        c = 0.2;
    # 如果b大于等于0.87，则c等于0.1
    else
        c = 0.1;
    # t等于10 * c + 0.2的向下取整
    t = Math.floor(10 * c + 0.2);
    # 打印结果字符串
    print("THE " + as + bs + " DID A " + ls[t] + " JOB.\n");
    # 如果t大于等于4
    if (4 <= t) {
        # 如果t不等于5
        if (5 != t) {
            # 执行switch语句
            switch (af(0)) {
                # 如果af(0)返回1
                case 1:
                    # 打印结果字符串
                    print("ONE OF THE " + as + bs + " WAS KILLED.\n");
                    break;
                # 如果af(0)返回2
                case 2:
                    # 打印结果字符串
                    print("NO " + as + b + " WERE KILLED.\n");
                    break;
            }
        } else {
            # 如果as不等于"TOREAD"
            if (as != "TOREAD")
                # 打印结果字符串
                print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n");
            # 打印结果字符串
            print(af(0) + " OF THE " + as + bs + " KILLED.\n");
        }
    }
    # 打印空行
    print("\n");
// 主程序
async function main()
{
    // 打印字符串并在末尾添加换行符
    print(tab(34) + "BULL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量 l
    l = 1;
    // 打印提示信息并等待用户输入
    print("DO YOU WANT INSTRUCTIONS");
    str = await input();
    // 如果用户输入不是 "NO"，则打印以下信息
    if (str != "NO") {
        print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.\n");
        print("HERE IS YOUR BIG CHANCE TO KILL A BULL.\n");
        print("\n");
        // 打印游戏规则
        print("ON EACH PASS OF THE BULL, YOU MAY TRY\n");
        print("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)\n");
        print("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE\n");
        print("2 - ORDINARY SWIRL OF THE CAPE.\n");
        print("\n");
        print("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL\n");
        print("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).\n");
        print("BUT IF I WERE YOU,\n");
        print("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.\n");
        print("\n");
        print("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE\n");
        print("(POSTHUMOUSLY IF NECESSARY).\n");
        print("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.\n");
        print("\n");
        print("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,\n");
        print("THE BETTER YOUR CHANCES ARE.\n");
    }
    print("\n");
    print("\n");
    // 初始化数组 d 的值
    d[5] = 1;
    d[4] = 1;
    d[3] = 0;
    // 生成一个随机数并赋值给变量 a
    a = Math.floor(Math.random() * 5 + 1);
    // 打印信息
    print("YOU HAVE DRAWN A " + ls[a] + " BULL.\n");
    // 根据 a 的值打印不同的信息
    if (a > 4) {
        print("YOU'RE LUCKY.\n");
    } else if (a < 2) {
        print("GOOD LUCK.  YOU'LL NEED IT.\n");
        print("\n");
    }
    print("\n");
    // 设置变量 as 和 bs 的值，并调用 setup_helpers 函数
    as = "PICADO";
    bs = "RES";
    setup_helpers();
    d[1] = c;
    as = "TOREAD";
    bs = "ORES";
    setup_helpers();
    d[2] = c;
    print("\n");
    print("\n");
    // 初始化变量 z
    z = 0;
    }
    print("\n");
    print("\n");
    print("\n");
}
    # 如果 d[4] 等于 0，则执行以下代码块
    if (d[4] == 0) {
        # 打印以下内容
        print("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW\n");
        print("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--\n");
        print("UNLESS THE BULL DOES FIRST.\n");
    } else {
        # 如果 d[4] 不等于 0，则执行以下代码块
        if (d[4] == 2) {
            # 打印以下内容
            print("THE CROWD CHEERS WILDLY!\n");
        } else if (d[5] == 2) {
            # 如果 d[4] 不等于 2 且 d[5] 等于 2，则执行以下代码块
            print("THE CROWD CHEERS!\n");
            print("\n");
        }
        # 打印以下内容
        print("THE CROWD AWARDS YOU\n");
        # 如果 cf(0) 小于 2.4，则执行以下代码块
        if (cf(0) < 2.4) {
            # 打印以下内容
            print("NOTHING AT ALL.\n");
        } else if (cf(0) < 4.9) {
            # 如果 cf(0) 大于等于 2.4 且小于 4.9，则执行以下代码块
            print("ONE EAR OF THE BULL.\n");
        } else if (cf(0) < 7.4) {
            # 如果 cf(0) 大于等于 4.9 且小于 7.4，则执行以下代码块
            print("BOTH EARS OF THE BULL!\n");
            print("OLE!\n");
        } else {
            # 如果 cf(0) 大于等于 7.4，则执行以下代码块
            print("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!\n");
        }
        print("\n");
        print("ADIOS\n");
        print("\n");
        print("\n");
        print("\n");
    }
# 调用名为main的函数
main();
```