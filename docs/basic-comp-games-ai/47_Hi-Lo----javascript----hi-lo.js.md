# `basic-computer-games\47_Hi-Lo\javascript\hi-lo.js`

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
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
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
    // 打印游戏标题
    print(tab(34) + "HI LO\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE GAME OF HI LO.\n");
    print("\n");
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE\n");
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU\n");
}
    # 打印游戏规则提示信息
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!\n");
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,\n");
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n");
    print("\n");
    # 初始化变量 r 为 0
    r = 0;
    # 进入游戏循环
    while (1) {
        # 初始化变量 b 为 0
        b = 0;
        print("\n");
        # 生成一个 0 到 99 的随机数
        y = Math.floor(100 * Math.random());
        # 进行 6 次猜数游戏
        for (b = 1; b <= 6; b++) {
            # 提示用户输入猜测的数字
            print("YOUR GUESS");
            # 获取用户输入的数字
            a = parseInt(await input());
            # 判断用户猜测的数字与随机数的大小关系
            if (a < y) {
                print("YOUR GUESS IS TOO LOW.\n");
            } else if (a > y) {
                print("YOUR GUESS IS TOO HIGH.\n");
            } else {
                break;
            }
            print("\n");
        }
        # 判断用户是否猜对了数字
        if (b > 6) {
            print("YOU BLEW IT...TOO BAD...THE NUMBER WAS " + y + "\n");
            r = 0;
        } else {
            print("GOT IT!!!!!!!!!!   YOU WIN " + y + " DOLLARS.\n");
            # 累加用户赢得的金额
            r += y;
            print("YOUR TOTAL WINNINGS ARE NOW " + r + " DOLLARS.\n");
        }
        print("\n");
        # 询问用户是否继续游戏
        print("PLAY AGAIN (YES OR NO)");
        # 获取用户输入的字符串并转换为大写
        str = await input();
        str = str.toUpperCase();
        # 如果用户输入不是 "YES" 则结束游戏循环
        if (str != "YES")
            break;
    }
    print("\n");
    print("SO LONG.  HOPE YOU ENJOYED YOURSELF!!!\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```