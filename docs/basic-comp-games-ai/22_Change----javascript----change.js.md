# `basic-computer-games\22_Change\javascript\change.js`

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
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
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
    // 打印指定数量的空格和文本
    print(tab(33) + "CHANGE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE\n");
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n");
    print("\n");
    print("\n");
    # 进入无限循环，直到用户手动中断程序
    while (1) {
        # 打印提示信息，要求输入商品的价格
        print("COST OF ITEM");
        # 将用户输入的价格转换为浮点数
        a = parseFloat(await input());
        # 打印提示信息，要求输入付款金额
        print("AMOUNT OF PAYMENT");
        # 将用户输入的付款金额转换为浮点数
        p = parseFloat(await input());
        # 计算找零金额
        c = p - a;
        # 将找零金额赋值给临时变量m
        m = c;
        # 如果找零金额为0
        if (c == 0) {
            # 打印正确的付款金额提示信息
            print("CORRECT AMOUNT, THANK YOU.\n");
        } else {
            # 打印找零金额
            print("YOUR CHANGE, $" + c + "\n");
            # 计算并打印十美元纸币的数量
            d = Math.floor(c / 10);
            if (d)
                print(d + " TEN DOLLAR BILL(S)\n");
            # 更新找零金额
            c -= d * 10;
            # 计算并打印五美元纸币的数量
            e = Math.floor(c / 5);
            if (e)
                print(e + " FIVE DOLLAR BILL(S)\n");
            # 更新找零金额
            c -= e * 5;
            # 计算并打印一美元纸币的数量
            f = Math.floor(c);
            if (f)
                print(f + " ONE DOLLAR BILL(S)\n");
            # 更新找零金额
            c -= f;
            # 将找零金额转换为分
            c *= 100;
            # 计算并打印半美元硬币的数量
            g = Math.floor(c / 50);
            if (g)
                print(g + " ONE HALF DOLLAR(S)\n");
            # 更新找零金额
            c -= g * 50;
            # 计算并打印25美分硬币的数量
            h = Math.floor(c / 25);
            if (h)
                print(h + " QUARTER(S)\n");
            # 更新找零金额
            c -= h * 25;
            # 计算并打印10美分硬币的数量
            i = Math.floor(c / 10);
            if (i)
                print(i + " DIME(S)\n");
            # 更新找零金额
            c -= i * 10;
            # 计算并打印5美分硬币的数量
            j = Math.floor(c / 5);
            if (j)
                print(j + " NICKEL(S)\n");
            # 更新找零金额
            c -= j * 5;
            # 计算并打印1美分硬币的数量
            k = Math.floor(c + 0.5);
            if (k)
                print(k + " PENNY(S)\n");
            # 打印感谢信息
            print("THANK YOU, COME AGAIN.\n");
            # 打印空行
            print("\n");
            # 打印空行
            print("\n");
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```