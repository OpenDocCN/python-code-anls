# `basic-computer-games\25_Chief\javascript\chief.js`

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

// 主程序
async function main()
{
    // 打印标题
    print(tab(30) + "CHIEF\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.\n");
    print("ARE YOU READY TO TAKE THE TEST YOU CALLED ME OUT FOR");
    // 等待用户输入
    a = await input();
    // 如果输入的字符串第一个字符不是 "Y"，则打印提示信息
    if (a.substr(0, 1) != "Y")
        print("SHUT UP, PALE FACE WITH WIE TONGUE.\n");
}
    # 打印提示信息
    print(" TAKE A NUMBER AND ADD 3. DIVIDE THIS NUMBER BY 5 AND\n");
    # 打印提示信息
    print("MULTIPLY BY 8. DIVIDE BY 5 AND ADD THE SAME. SUBTRACT 1.\n");
    # 打印提示信息
    print("  WHAT DO YOU HAVE");
    # 获取用户输入的数字并转换为浮点数
    b = parseFloat(await input());
    # 根据给定的数学运算公式计算结果
    c = (b + 1 - 5) * 5 / 8 * 5 - 3;
    # 打印猜测用户输入的数字
    print("I BET YOUR NUMBER WAS " + Math.floor(c + 0.5) + ". AM I RIGHT");
    # 获取用户输入的回答
    d = await input();
    # 如果用户回答不是以"Y"开头，则执行以下代码块
    if (d.substr(0, 1) != "Y") {
        # 打印提示信息
        print("WHAT WAS YOUR ORIGINAL NUMBER");
        # 获取用户输入的数字并转换为浮点数
        k = parseFloat(await input());
        # 根据给定的数学运算公式计算结果
        f = k + 3;
        g = f / 5;
        h = g * 8;
        i = h / 5 + 5;
        j = i - 1;
        # 打印一系列数学运算的结果
        print("SO YOU THINK YOU'RE SO SMART, EH?\n");
        print("NOW WATCH.\n");
        print(k + " PLUS 3 EQUALS " + f + ". THIS DIVIDED BY 5 EQUALS " + g + ";\n");
        print("THIS TIMES 8 EQUALS " + h + ". IF WE DIVIDE BY 5 AND ADD 5,\n");
        print("WE GET " + i + ", WHICH, MINUS 1, EQUALS " + j + ".\n");
        print("NOW DO YOU BELIEVE ME");
        # 获取用户输入的回答
        z = await input();
        # 如果用户回答不是以"Y"开头，则执行以下代码块
        if (z.substr(0, 1) != "Y") {
            # 打印愤怒的提示信息
            print("YOU HAVE MADE ME MAD!!!\n");
            print("THERE MUST BE A GREAT LIGHTNING BOLT!\n");
            print("\n");
            print("\n");
            # 打印闪电图案
            for (x = 30; x >= 22; x--)
                print(tab(x) + "X X\n");
            print(tab(21) + "X XXX\n");
            print(tab(20) + "X   X\n");
            print(tab(19) + "XX X\n");
            for (y = 20; y >= 13; y--)
                print(tab(y) + "X X\n");
            print(tab(12) + "XX\n");
            print(tab(11) + "X\n");
            print(tab(10) + "*\n");
            print("\n");
            print("#########################\n");
            print("\n");
            # 打印警告信息
            print("I HOPE YOU BELIEVE ME NOW, FOR YOUR SAKE!!\n");
            # 返回结束程序
            return;
        }
    }
    # 打印结束信息
    print("BYE!!!\n");
# 调用名为main的函数
main();
```