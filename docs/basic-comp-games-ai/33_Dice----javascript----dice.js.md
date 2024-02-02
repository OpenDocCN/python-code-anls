# `basic-computer-games\33_Dice\javascript\dice.js`

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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
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

// 定义一个制表符函数，返回指定数量的空格字符串
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
    // 打印标题
    print(tab(34) + "DICE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组 f
    f = [];
    // 打印提示信息
    print("THIS PROGRAM SIMULATES THE ROLLING OF A\n");
    print("PAIR OF DICE.\n");
    print("YOU ENTER THE NUMBER OF TIMES YOU WANT THE COMPUTER TO\n");
    print("'ROLL' THE DICE.  WATCH OUT, VERY LARGE NUMBERS TAKE\n");
}
    # 打印提示信息
    print("A LONG TIME.  IN PARTICULAR, NUMBERS OVER 5000.\n");
    # 循环开始
    do {
        # 初始化数组 f 的值为 0
        for (q = 1; q <= 12; q++)
            f[q] = 0;
        # 打印提示信息，获取用户输入并转换为整数
        print("\n");
        print("HOW MANY ROLLS");
        x = parseInt(await input());
        # 循环生成随机数并计算点数总和
        for (s = 1; s <= x; s++) {
            a = Math.floor(Math.random() * 6 + 1);
            b = Math.floor(Math.random() * 6 + 1);
            r = a + b;
            f[r]++;
        }
        # 打印点数和出现次数
        print("\n");
        print("TOTAL SPOTS\tNUMBER OF TIMES\n");
        for (v = 2; v <= 12; v++) {
            print("\t" + v + "\t" + f[v] + "\n");
        }
        # 打印提示信息，获取用户输入
        print("\n");
        print("\n");
        print("TRY AGAIN");
        str = await input();
    # 当用户输入以 "Y" 开头时继续循环
    } while (str.substr(0, 1) == "Y") ;
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```