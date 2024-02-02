# `basic-computer-games\75_Roulette\javascript\roulette.js`

```py
// ROULETTE
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

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
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并解析 Promise
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

// 定义一些空数组和数字数组
var ba = [];
var ca = [];
var ta = [];
var xa = [];
var aa = [];
var numbers = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36];

// 主程序，使用 async 函数定义
async function main()
{
    // 打印标题
    print(tab(32) + "ROULETTE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印欢迎信息
    print("WELCOME TO THE ROULETTE TABLE\n");
    print("\n");
    # 打印提示信息，询问是否需要说明
    print("DO YOU WANT INSTRUCTIONS");
    # 等待用户输入
    str = await input();
    # 如果用户输入的第一个字符不是"N"
    if (str.substr(0, 1) != "N") {
        # 打印赌注布局
        print("\n");
        print("THIS IS THE BETTING LAYOUT\n");
        print("  (*=RED)\n");
        print("\n");
        print(" 1*    2     3*\n");
        print(" 4     5*    6 \n");
        print(" 7*    8     9*\n");
        print("10    11    12*\n");
        print("---------------\n");
        print("13    14*   15 \n");
        print("16*   17    18*\n");
        print("19*   20    21*\n");
        print("22    23*   24 \n");
        print("---------------\n");
        print("25*   26    27*\n");
        print("28    29    30*\n");
        print("31    32*   33 \n");
        print("34*   35    36*\n");
        print("---------------\n");
        print("    00    0    \n");
        print("\n");
        # 打印赌注类型
        print("TYPES OF BETS\n");
        print("\n");
        print("THE NUMBERS 1 TO 36 SIGNIFY A STRAIGHT BET\n");
        print("ON THAT NUMBER.\n");
        print("THESE PAY OFF 35:1\n");
        print("\n");
        print("THE 2:1 BETS ARE:\n");
        print(" 37) 1-12     40) FIRST COLUMN\n");
        print(" 38) 13-24    41) SECOND COLUMN\n");
        print(" 39) 25-36    42) THIRD COLUMN\n");
        print("\n");
        print("THE EVEN MONEY BETS ARE:\n");
        print(" 43) 1-18     46) ODD\n");
        print(" 44) 19-36    47) RED\n");
        print(" 45) EVEN     48) BLACK\n");
        print("\n");
        print(" 49)0 AND 50)00 PAY OFF 35:1\n");
        print(" NOTE: 0 AND 00 DO NOT COUNT UNDER ANY\n");
        print("       BETS EXCEPT THEIR OWN.\n");
        print("\n");
        print("WHEN I ASK FOR EACH BET, TYPE THE NUMBER\n");
        print("AND THE AMOUNT, SEPARATED BY A COMMA.\n");
        print("FOR EXAMPLE: TO BET $500 ON BLACK, TYPE 48,500\n");
        print("WHEN I ASK FOR A BET.\n");
        print("\n");
        print("THE MINIMUM BET IS $5, THE MAXIMUM IS $500.\n");
        print("\n");
    }
    # 程序从这里开始
    // 初始化赌注、赔率和其他变量
    for (i = 1; i <= 100; i++) {
        ba[i] = 0;  // 初始化赌注数组
        ca[i] = 0;  // 初始化赔率数组
        ta[i] = 0;  // 初始化其他数组
    }
    for (i = 1; i <= 38; i++)
        xa[i] = 0;  // 初始化其他数组
    p = 1000;  // 设置赌注上限
    d = 100000;  // 设置赔率上限
    }
    // 如果赌注小于1，则输出信息并结束程序
    if (p < 1) {
        print("THANKS FOR YOUR MONEY.\n");
        print("I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL\n");
    } else {
        // 否则，要求输入收款人信息
        print("TO WHOM SHALL I MAKE THE CHECK");
        str = await input();
        print("\n");
        // 输出格式化的支票信息
        for (i = 1; i <= 72; i++)
            print("-");
        print("\n");
        print(tab(50) + "CHECK NO. " + Math.floor(Math.random() * 100) + "\n");
        print("\n");
        print(tab(40) + new Date().toDateString());
        print("\n");
        print("\n");
        print("PAY TO THE ORDER OF-----" + str + "-----$ " + p + "\n");
        print("\n");
        print("\n");
        print(tab(10) + "\tTHE MEMORY BANK OF NEW YORK\n");
        print("\n");
        print(tab(40) + "\tTHE COMPUTER\n");
        print(tab(40) + "----------X-----\n");
        print("\n");
        for (i = 1; i <= 72; i++)
            print("-");
        print("\n");
        print("COME BACK SOON!\n");
    }
    print("\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```