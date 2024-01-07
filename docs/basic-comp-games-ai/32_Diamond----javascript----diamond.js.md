# `basic-computer-games\32_Diamond\javascript\diamond.js`

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
    // 打印标题
    print(tab(33) + "DIAMOND\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("FOR A PRETTY DIAMOND PATTERN,\n");
    print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21");
    // 获取用户输入的奇数
    r = parseInt(await input());
    // 计算每行的重复次数
    q = Math.floor(60 / r);
    as = "CC"
    x = 1;
    y = r;
    z = 2;
    for (l = 1; l <= q; l++) {
        for (n = x; z < 0 ? n >= y : n <= y; n += z) {
            str = "";
            // 添加空格以居中
            while (str.length < (r - n) / 2)
                str += " ";
            for (m = 1; m <= q; m++) {
                c = 1;
                for (a = 1; a <= n; a++) {
                    if (c > as.length)
                        str += "!";
                    else
                        str += as[c++ - 1];
                }
                if (m == q)
                    break;
                // 添加空格以居中
                while (str.length < r * m + (r - n) / 2)
                    str += " ";
            }
            // 打印每行的字符串
            print(str + "\n");
        }
        if (x != 1) {
            x = 1;
            y = r;
            z = 2;
        } else {
            x = r - 2;
            y = 1;
            z = -2;
            l--;
        }
    }
}

// 调用主程序
main();

```