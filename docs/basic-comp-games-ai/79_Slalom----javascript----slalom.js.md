# `basic-computer-games\79_Slalom\javascript\slalom.js`

```
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

// 定义一个速度数组
var speed = [,14,18,26,29,18,
             25,28,32,29,20,
             29,29,25,21,26,
             29,20,21,20,18,
             26,25,33,31,22];

// 定义一个显示指令的函数
function show_instructions()
{
    // 打印游戏说明
    print("\n");
    print("*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE\n");
    print("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.\n");
    print("\n");
    print("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.\n");
}
    # 打印提示信息，让用户选择加速程度
    print("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.\n");
    print("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.\n");
    print("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.\n");
    print("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.\n");
    print("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.\n");
    print("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.\n");
    print("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.\n");
    print("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.\n");
    print("\n");
    # 打印提示信息，告诉用户在计算机询问时使用这些选项
    print(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:\n");
    print("\n");
    # 打印计算机询问时的提示信息
    print("OPTION?\n");
    print("\n");
    # 打印祝福信息
    print("                GOOD LUCK!\n");
    print("\n");
// 显示速度信息
function show_speeds()
{
    // 打印标题
    print("GATE MAX\n");
    // 打印表头
    print(" #  M.P.H.\n");
    // 打印分隔线
    print("----------\n");
    // 遍历速度数组，打印门数和对应的速度
    for (var b = 1; b <= v; b++) {
        print(" " + b + "  " + speed[b] + "\n");
    }
}

// 主程序
async function main()
{
    // 初始化金牌、银牌、铜牌数量
    var gold = 0;
    var silver = 0;
    var bronze = 0;

    // 打印标题
    print(tab(33) + "SLALOM\n");
    // 打印作者信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 循环直到输入合法的门数
    while (1) {
        print("HOW MANY GATES DOES THIS COURSE HAVE (1 TO 25)");
        v = parseInt(await input());
        if (v >= 25) {
            print("25 IS THE LIMIT\n");
            v = 25;
        } else if (v < 1) {
            print("TRY AGAIN.\n");
        } else {
            break;
        }
    }
    // 打印提示信息
    print("\n");
    print("TYPE \"INS\" FOR INSTRUCTIONS\n");
    print("TYPE \"MAX\" FOR APPROXIMATE MAXIMUM SPEEDS\n");
    print("TYPE \"RUN\" FOR THE BEGINNING OF THE RACE\n");
    // 循环直到输入合法的指令
    while (1) {
        print("COMMAND--");
        str = await input();
        if (str == "INS") {
            show_instructions();
        } else if (str == "MAX") {
            show_speeds();
        } else if (str == "RUN") {
            break;
        } else {
            print("\"" + str + "\" IS AN ILLEGAL COMMAND--RETRY");
        }
    }
    // 循环直到输入合法的评分
    while (1) {
        print("RATE YOURSELF AS A SKIER, (1=WORST, 3=BEST)");
        a = parseInt(await input());
        if (a < 1 || a > 3)
            print("THE BOUNDS ARE 1-3\n");
        else
            break;
    }
    // 打印感谢信息
    print("THANKS FOR THE RACE\n");
    // 打印奖牌信息
    if (gold >= 1)
        print("GOLD MEDALS: " + gold + "\n");
    if (silver >= 1)
        print("SILVER MEDALS: " + silver + "\n");
    if (bronze >= 1)
        print("BRONZE MEDALS: " + bronze + "\n");
}

// 调用主程序
main();
```