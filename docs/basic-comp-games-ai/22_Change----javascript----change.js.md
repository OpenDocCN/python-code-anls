# `basic-computer-games\22_Change\javascript\change.js`

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
                       // 监听键盘事件，当按下回车键时，获取输入值并解析为字符串
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入值并传递给 resolve 函数
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

// 主程序，使用 async 函数定义，可以使用 await 关键字等待 Promise 对象的解析
async function main()
{
    // 打印标题
    print(tab(33) + "CHANGE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE\n");
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n");
    print("\n");
    print("\n");
    while (1) {
        // 获取商品价格
        print("COST OF ITEM");
        a = parseFloat(await input());
        // 获取支付金额
        print("AMOUNT OF PAYMENT");
        p = parseFloat(await input());
        // 计算找零金额
        c = p - a;
        m = c;
        if (c == 0) {
            print("CORRECT AMOUNT, THANK YOU.\n");
        } else {
            print("YOUR CHANGE, $" + c + "\n");
            // 计算各种面额的钞票/硬币数量
            d = Math.floor(c / 10);
            if (d)
                print(d + " TEN DOLLAR BILL(S)\n");
            c -= d * 10;
            e = Math.floor(c / 5);
            if (e)
                print(e + " FIVE DOLLAR BILL(S)\n");
            c -= e * 5;
            f = Math.floor(c);
            if (f)
                print(f + " ONE DOLLAR BILL(S)\n");
            c -= f;
            c *= 100;
            g = Math.floor(c / 50);
            if (g)
                print(g + " ONE HALF DOLLAR(S)\n");
            c -= g * 50;
            h = Math.floor(c / 25);
            if (h)
                print(h + " QUARTER(S)\n");
            c -= h * 25;
            i = Math.floor(c / 10);
            if (i)
                print(i + " DIME(S)\n");
            c -= i * 10;
            j = Math.floor(c / 5);
            if (j)
                print(j + " NICKEL(S)\n");
            c -= j * 5;
            k = Math.floor(c + 0.5);
            if (k)
                print(k + " PENNY(S)\n");
            print("THANK YOU, COME AGAIN.\n");
            print("\n");
            print("\n");
        }
    }
}

// 调用主程序
main();

```