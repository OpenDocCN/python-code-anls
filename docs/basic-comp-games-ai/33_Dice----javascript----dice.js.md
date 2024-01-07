# `basic-computer-games\33_Dice\javascript\dice.js`

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
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析为字符串，然后返回
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

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序，使用 async 函数定义
async function main()
{
    // 打印标题
    print(tab(34) + "DICE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    f = [];
    // 打印说明
    print("THIS PROGRAM SIMULATES THE ROLLING OF A\n");
    print("PAIR OF DICE.\n");
    print("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO\n");
    print("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE\n");
    print("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.\n");
    do {
        // 初始化数组 f
        for (q = 1; q <= 12; q++)
            f[q] = 0;
        print("\n");
        print("HOW MANY ROLLS");
        // 获取用户输入的次数
        x = parseInt(await input());
        for (s = 1; s <= x; s++) {
            // 模拟掷骰子，计算点数并统计次数
            a = Math.floor(Math.random() * 6 + 1);
            b = Math.floor(Math.random() * 6 + 1);
            r = a + b;
            f[r]++;
        }
        print("\n");
        print("TOTAL SPOTS\tNUMBER OF TIMES\n");
        // 打印点数和次数
        for (v = 2; v <= 12; v++) {
            print("\t" + v + "\t" + f[v] + "\n");
        }
        print("\n");
        print("\n");
        print("TRY AGAIN");
        // 获取用户输入，如果以 "Y" 开头则继续循环
        str = await input();
    } while (str.substr(0, 1) == "Y") ;
}

// 调用主程序
main();

```