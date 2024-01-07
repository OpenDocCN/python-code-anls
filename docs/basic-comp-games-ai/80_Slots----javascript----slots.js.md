# `basic-computer-games\80_Slots\javascript\slots.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
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
                       // 输入框获取焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行
                                                      print("\n");
                                                      // 解析输入的字符串并返回
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定长度的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个包含不同图案的数组
var figures = [, "BAR", "BELL", "ORANGE", "LEMON", "PLUM", "CHERRY"];

// 主程序
async function main()
{
    // 打印标题
    print(tab(30) + "SLOTS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印提示信息
    print("YOU ARE IN THE H&M CASINO,IN FRONT ON ONE OF OUR\n");
    print("ONE-ARM BANDITS. BET FROM $1 TO $100.\n");
    print("TO PULL THE ARM, PUNCH THE RETURN KEY AFTER MAKING YOUR BET.\n");
    // 初始化变量
    p = 0;
    // 循环
    while (1) {
        while (1) {
            // 打印提示信息
            print("\n");
            print("YOUR BET");
            // 获取输入的赌注
            m = parseInt(await input());
            // 检查赌注是否符合要求
            if (m > 100) {
                print("HOUSE LIMITS ARE $100\n");
            } else if (m < 1) {
                print("MINIMUM BET IS $1\n");
            } else {
                break;
            }
        }
        // 生成随机数
        x = Math.floor(6 * Math.random() + 1);
        y = Math.floor(6 * Math.random() + 1);
        z = Math.floor(6 * Math.random() + 1);
        // 打印图案
        print("\n");
        print(figures[x] + " " + figures[y] + " " + figures[z] + "\n");
        // 初始化变量
        lost = false;
        // 判断输赢
        if (x == y && y == z) {  // 三个相同图案
            print("\n");
            if (z != 1) {
                print("**TOP DOLLAR**\n");
                p += ((10 * m) + m);
            } else {
                print("***JACKPOT***\n");
                p += ((100 * m) + m);
            }
            print("YOU WON!\n");
        } else if (x == y || y == z || x == z) {
            if (x == y)
                c = x;
            else
                c = z;
            if (c == 1) {
                print("\n");
                print("*DOUBLE BAR*\n");
                print("YOU WON\n");
                p += ((5 * m) + m);
            } else if (x != z) {
                print("\n");
                print("DOUBLE!!\n");
                print("YOU WON!\n");
                p += ((2 * m) + m);
            } else {
                lost = true;
            }
        } else {
            lost = true;
        }
        // 处理输赢
        if (lost) {
            print("\n");
            print("YOU LOST.\n");
            p -= m;
        }
        // 打印当前金额
        print("YOUR STANDINGS ARE $" + p + "\n");
        print("AGAIN");
        // 获取输入的字符串
        str = await input();
        // 如果输入的字符串不是以"Y"开头，则跳出循环
        if (str.substr(0, 1) != "Y")
            break;
    }
    // 打印最终结果
    print("\n");
    if (p < 0) {
        print("PAY UP!  PLEASE LEAVE YOUR MONEY ON THE TERMINAL.\n");
    } else if (p == 0) {
        print("HEY, YOU BROKE EVEN.\n");
    } else {
        print("COLLECT YOUR WINNINGS FROM THE H&M CASHIER.\n");
    }
}

// 调用主程序
main();

```