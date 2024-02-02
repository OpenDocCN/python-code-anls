# `basic-computer-games\29_Craps\javascript\craps.js`

```py
// CRAPS
//
// 由Oscar Toledo G. (nanochess)从BASIC转换为Javascript

// 打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个Promise对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 生成指定数量空格的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 掷骰子函数，返回两个骰子点数之和
function roll()
{
    return Math.floor(6 * Math.random())+1 + Math.floor(6 * Math.random())+1;
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "CRAPS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量
    r = 0;
    // 打印游戏规则
    print("2,3,12 ARE LOSERS: 4,5,6,8,9,10 ARE POINTS: 7,11 ARE NATURAL WINNERS.\n");
    # 进入游戏循环
    while (1) {
        # 打印提示信息，输入赌注金额
        print("INPUT THE AMOUNT OF YOUR WAGER.");
        f = parseInt(await input());
        # 打印提示信息，掷骰子
        print("I WILL NOW THROW THE DICE\n");
        x = roll();
        # 判断骰子点数是否为7或11，是则赢得赌注金额
        if (x == 7 || x == 11) {
            print(x + " - NATURAL....A WINNER!!!!\n");
            print(x + " PAYS EVEN MONEY, YOU WIN " + f + " DOLLARS\n");
            r += f;
        } else if (x == 2) {
            # 判断骰子点数是否为2，是则输掉赌注金额
            print(x + " - SNAKE EYES....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else if (x == 3 || x == 12) { // Original duplicates comparison in line 70
            # 判断骰子点数是否为3或12，是则输掉赌注金额
            print(x + " - CRAPS....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else {
            # 骰子点数不为特殊值，进入循环直到点数为7或者再次出现初始点数
            print(x + " IS THE POINT. I WILL ROLL AGAIN\n");
            while (1) {
                o = roll();
                if (o == 7) {
                    # 如果点数为7，则输掉赌注金额
                    print(o + " - CRAPS, YOU LOSE.\n");
                    print("YOU LOSE $" + f + "\n");
                    r -= f;
                    break;
                }
                if (o == x) {
                    # 如果点数再次出现初始点数，则赢得赌注金额
                    print(x + " - A WINNER.........CONGRATS!!!!!!!!\n");
                    print(x + " AT 2 TO 1 ODDS PAYS YOU...LET ME SEE..." + 2 * f + " DOLLARS\n");
                    r += f * 2;
                    break;
                }
                # 否则继续循环
                print(o + " - NO POINT. I WILL ROLL AGAIN\n");
            }
        }
        # 打印提示信息，询问是否继续游戏
        print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2");
        m = parseInt(await input());
        # 根据赌注金额情况打印相应信息
        if (r < 0) {
            print("YOU ARE NOW UNDER $" + -r + "\n");
        } else if (r > 0) {
            print("YOU ARE NOW AHEAD $" + r + "\n");
        } else {
            print("YOU ARE NOW EVEN AT 0\n");
        }
        # 如果不想继续游戏，则跳出循环
        if (m != 5)
            break;
    }
    # 根据赌注金额情况打印最终结果信息
    if (r < 0) {
        print("TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.\n");
    } else if (r > 0) {
        print("CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!\n");
    } else {
        # 如果条件不满足，则打印恭喜消息
        print("CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR\n");
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```