# `basic-computer-games\29_Craps\javascript\craps.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
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

                       // 输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串并返回
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个掷骰子的函数，返回两个骰子点数之和
function roll()
{
    return Math.floor(6 * Math.random())+1 + Math.floor(6 * Math.random())+1;
}

// 主程序，使用 async 函数定义
async function main()
{
    // 输出标题
    print(tab(33) + "CRAPS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量 r
    r = 0;
    // 输出游戏规则
    print("2,3,12 ARE LOSERS: 4,5,6,8,9,10 ARE POINTS: 7,11 ARE NATURAL WINNERS.\n");
    // 循环进行游戏
    while (1) {
        // 获取下注金额
        print("INPUT THE AMOUNT OF YOUR WAGER.");
        f = parseInt(await input());
        // 输出掷骰子信息
        print("I WILL NOW THROW THE DICE\n");
        x = roll();
        // 根据掷骰子结果进行判断
        if (x == 7 || x == 11) {
            print(x + " - NATURAL....A WINNER!!!!\n");
            print(x + " PAYS EVEN MONEY, YOU WIN " + f + " DOLLARS\n");
            r += f;
        } else if (x == 2) {
            print(x + " - SNAKE EYES....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else if (x == 3 || x == 12) { // Original duplicates comparison in line 70
            print(x + " - CRAPS....YOU LOSE.\n");
            print("YOU LOSE " + f + " DOLLARS.\n");
            r -= f;
        } else {
            print(x + " IS THE POINT. I WILL ROLL AGAIN\n");
            // 循环掷骰子，直到出现点数或者 7
            while (1) {
                o = roll();
                if (o == 7) {
                    print(o + " - CRAPS, YOU LOSE.\n");
                    print("YOU LOSE $" + f + "\n");
                    r -= f;
                    break;
                }
                if (o == x) {
                    print(x + " - A WINNER.........CONGRATS!!!!!!!!\n");
                    print(x + " AT 2 TO 1 ODDS PAYS YOU...LET ME SEE..." + 2 * f + " DOLLARS\n");
                    r += f * 2;
                    break;
                }
                print(o + " - NO POINT. I WILL ROLL AGAIN\n");
            }
        }
        // 询问是否继续游戏
        print(" IF YOU WANT TO PLAY AGAIN PRINT 5 IF NOT PRINT 2");
        m = parseInt(await input());
        // 根据赢钱情况输出信息
        if (r < 0) {
            print("YOU ARE NOW UNDER $" + -r + "\n");
        } else if (r > 0) {
            print("YOU ARE NOW AHEAD $" + r + "\n");
        } else {
            print("YOU ARE NOW EVEN AT 0\n");
        }
        // 如果不想继续游戏，则跳出循环
        if (m != 5)
            break;
    }
    // 根据最终赢钱情况输出信息
    if (r < 0) {
        print("TOO BAD, YOU ARE IN THE HOLE. COME AGAIN.\n");
    } else if (r > 0) {
        print("CONGRATULATIONS---YOU CAME OUT A WINNER. COME AGAIN!\n");
    } else {
        print("CONGRATULATIONS---YOU CAME OUT EVEN, NOT BAD FOR AN AMATEUR\n");
    }

}

// 调用主程序
main();

```