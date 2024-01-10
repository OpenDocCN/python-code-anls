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
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并移除输入框，然后解析输入值并返回
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
    // 获取用户输入并解析为整数
    r = parseInt(await input());
    // 计算 q 的值
    q = Math.floor(60 / r);
    // 初始化变量
    as = "CC"
    x = 1;
    y = r;
    z = 2;
}
    # 外层循环，控制打印的行数
    for (l = 1; l <= q; l++) {
        # 内层循环，控制每行的字符数和方向
        for (n = x; z < 0 ? n >= y : n <= y; n += z) {
            # 初始化空字符串
            str = "";
            # 在字符串长度达到一定值之前，向字符串添加空格
            while (str.length < (r - n) / 2)
                str += " ";
            # 内层循环，控制每行中的字符内容
            for (m = 1; m <= q; m++) {
                # 初始化计数器
                c = 1;
                # 根据n的值向字符串添加特定字符
                for (a = 1; a <= n; a++) {
                    # 如果计数器超出了字符数组的长度，向字符串添加"!"
                    if (c > as.length)
                        str += "!";
                    # 否则向字符串添加字符数组中的字符
                    else
                        str += as[c++ - 1];
                }
                # 如果m等于q，跳出内层循环
                if (m == q)
                    break;
                # 在字符串长度达到一定值之前，向字符串添加空格
                while (str.length < r * m + (r - n) / 2)
                    str += " ";
            }
            # 打印字符串并换行
            print(str + "\n");
        }
        # 根据x的值更新x、y、z的值
        if (x != 1) {
            x = 1;
            y = r;
            z = 2;
        } else {
            x = r - 2;
            y = 1;
            z = -2;
            # l减一，使外层循环再次执行当前行
            l--;
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```