# `basic-computer-games\55_Life\javascript\life.js`

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
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值
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

// 定义两个空数组
var bs = [];
var a = [];

// 主程序
async function main()
{
    // 打印标题
    print(tab(34) + "LIFE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("ENTER YOUR PATTERN:\n");
    // 初始化坐标
    x1 = 1;
    y1 = 1;
    x2 = 24;
    y2 = 70;
    // 初始化数组
    for (c = 1; c <= 24; c++) {
        bs[c] = "";
        a[c] = [];
        for (d = 1; d <= 70; d++)
            a[c][d] = 0;
    }
    c = 1;
}
    # 进入无限循环，直到条件不满足
    while (1) {
        # 从输入中获取数据，存入数组 bs 的第 c 个位置
        bs[c] = await input();
        # 如果输入为 "DONE"，则清空 bs[c] 并跳出循环
        if (bs[c] == "DONE") {
            bs[c] = "";
            break;
        }
        # 如果 bs[c] 的第一个字符是 "."，则将其替换为空格
        if (bs[c].substr(0, 1) == ".")
            bs[c] = " " + bs[c].substr(1);
        # c 自增
        c++;
    }
    # c 减一
    c--;
    # 初始化变量 l 为 0
    l = 0;
    # 遍历 bs 数组，找出最长的字符串长度，存入 l
    for (x = 1; x <= c - 1; x++) {
        if (bs[x].length > l)
            l = bs[x].length;
    }
    # 根据 c 和 l 计算出 x1 和 y1 的值
    x1 = 11 - (c >> 1);
    y1 = 33 - (l >> 1);
    # 初始化变量 p 为 0
    p = 0;
    # 遍历 bs 数组，将非空格字符的位置标记为 1，并统计非空格字符的个数
    for (x = 1; x <= c; x++) {
        for (y = 1; y <= bs[x].length; y++) {
            if (bs[x][y - 1] != " ") {
                a[x1 + x][y1 + y] = 1;
                p++;
            }
        }
    }
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 初始化变量 i9 为 false
    i9 = false;
    # 初始化变量 g 为 0
    g = 0;
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```