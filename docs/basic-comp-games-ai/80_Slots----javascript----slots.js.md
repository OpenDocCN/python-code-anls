# `basic-computer-games\80_Slots\javascript\slots.js`

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
                       // 设置输入元素类型和长度
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

// 定义一个数组，包含了几种水果的名称
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
    // 打印说明信息
    print("YOU ARE IN THE H&M CASINO,IN FRONT ON ONE OF OUR\n");
}
    # 打印赌博游戏的提示信息
    print("ONE-ARM BANDITS. BET FROM $1 TO $100.\n");
    # 打印拉杆机的提示信息
    print("TO PULL THE ARM, PUNCH THE RETURN KEY AFTER MAKING YOUR BET.\n");
    # 初始化玩家的赌注
    p = 0;
    // 无限循环，直到用户选择退出
    while (1) {
        // 无限循环，直到用户输入有效的赌注
        while (1) {
            // 打印空行和提示用户输入赌注
            print("\n");
            print("YOUR BET");
            // 将用户输入的赌注转换为整数
            m = parseInt(await input());
            // 如果赌注大于100，提示用户超出了赌注上限
            if (m > 100) {
                print("HOUSE LIMITS ARE $100\n");
            } else if (m < 1) {  // 如果赌注小于1，提示用户低于最低赌注
                print("MINIMUM BET IS $1\n");
            } else {
                break;  // 跳出循环，赌注有效
            }
        }
        // 未实现的功能：GOSUB 1270 十次钟声
        print("\n");
        // 生成1到6之间的随机整数
        x = Math.floor(6 * Math.random() + 1);
        y = Math.floor(6 * Math.random() + 1);
        z = Math.floor(6 * Math.random() + 1);
        print("\n");
        // 未实现的功能：GOSUB 1310 在x和y之后七次钟声
        // 打印三个骰子的结果
        print(figures[x] + " " + figures[y] + " " + figures[z] + "\n");
        lost = false;
        if (x == y && y == z) {  // 三个相同的数字
            print("\n");
            if (z != 1) {
                print("**TOP DOLLAR**\n");
                // 计算赢得的金额并更新玩家的余额
                p += ((10 * m) + m);
            } else {
                print("***JACKPOT***\n");
                // 计算赢得的金额并更新玩家的余额
                p += ((100 * m) + m);
            }
            print("YOU WON!\n");
        } else if (x == y || y == z || x == z) {  // 两个相同的数字
            if (x == y)
                c = x;
            else
                c = z;
            if (c == 1) {
                print("\n");
                print("*DOUBLE BAR*\n");
                print("YOU WON\n");
                // 计算赢得的金额并更新玩家的余额
                p += ((5 * m) + m);
            } else if (x != z) {
                print("\n");
                print("DOUBLE!!\n");
                print("YOU WON!\n");
                // 计算赢得的金额并更新玩家的余额
                p += ((2 * m) + m);
            } else {
                lost = true;
            }
        } else {
            lost = true;
        }
        if (lost) {
            print("\n");
            print("YOU LOST.\n");
            // 扣除赌注金额
            p -= m;
        }
        // 打印玩家的余额
        print("YOUR STANDINGS ARE $" + p + "\n");
        print("AGAIN");
        // 等待用户输入是否再次进行游戏
        str = await input();
        if (str.substr(0, 1) != "Y")
            break;  // 如果用户输入不是以Y开头，则退出循环
    }
    print("\n");
    # 如果赌注小于0，则输出要求支付赌注的信息
    if (p < 0) {
        print("PAY UP!  PLEASE LEAVE YOUR MONEY ON THE TERMINAL.\n");
    # 如果赌注等于0，则输出平局的信息
    } else if (p == 0) {
        print("HEY, YOU BROKE EVEN.\n");
    # 如果赌注大于0，则输出赢得赌注的信息
    } else {
        print("COLLECT YOUR WINNINGS FROM THE H&M CASHIER.\n");
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```