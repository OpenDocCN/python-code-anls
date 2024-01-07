# `basic-computer-games\06_Banner\javascript\banner.js`

```

// BANNER
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
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
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义字符数组
var letters = [" ",0,0,0,0,0,0,0,
               // ... 省略部分字符数据 ...
               ];

f = [];
j = [];
s = [];

// 主程序
async function main()
{
    // 打印提示信息
    print("HORIZONTAL");
    // 获取水平位置
    x = parseInt(await input());
    // 打印提示信息
    print("VERTICAL");
    // 获取垂直位置
    y = parseInt(await input());
    // 打印提示信息
    print("CENTERED");
    // 获取居中字符
    ls = await input();
    g1 = 0;
    if (ls > "P")
        g1 = 1;
    // 打印提示信息
    print("CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)");
    // 获取字符
    ms = await input();
    // 打印提示信息
    print("STATEMENT");
    // 获取语句
    as = await input();
    // 打印提示信息
    print("SET PAGE");	// This means to prepare printer, just press Enter
    // 获取页面设置
    os = await input();

    for (t = 0; t < as.length; t++) {
        ps = as.substr(t, 1);
        for (o = 0; o < 50 * 8; o += 8) {
            if (letters[o] == ps) {
                for (u = 1; u <= 7; u++)
                    s[u] = letters[o + u];
                break;
            }
        }
        if (o == 50 * 8) {
            ps = " ";
            o = 0;
        }
        // 打印调试信息
//      print("Doing " + o + "\n");
        if (o == 0) {
            for (h = 1; h <= 7 * x; h++)
                print("\n");
        } else {
            xs = ms;
            if (ms == "ALL")
                xs = ps;
            for (u = 1; u <= 7; u++) {
                // 一个低效的提取位的方法
                // 但在BASIC中足够好，因为没有位移运算符。
                for (k = 8; k >= 0; k--) {
                    if (Math.pow(2, k) >= s[u]) {
                        j[9 - k] = 0;
                    } else {
                        j[9 - k] = 1;
                        s[u] -= Math.pow(2, k);
                        if (s[u] == 1) {
                            f[u] = 9 - k;
                            break;
                        }
                    }
                }
                for (t1 = 1; t1 <= x; t1++) {
                    str = tab((63 - 4.5 * y) * g1 / xs.length + 1);
                    for (b = 1; b <= f[u]; b++) {
                        if (j[b] == 0) {
                            for (i = 1; i <= y; i++)
                                str += tab(xs.length);
                        } else {
                            for (i = 1; i <= y; i++)
                                str += xs;
                        }
                    }
                    print(str + "\n");
                }
            }
            for (h = 1; h <= 2 * x; h++)
                print("\n");
        }
    }
}

// 调用主程序
main();

```