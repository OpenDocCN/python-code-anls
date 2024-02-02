# `basic-computer-games\05_Bagels\javascript\bagels.js`

```py
// 定义一个打印函数，将字符串输出到指定元素
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

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
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
                                                      // 输出输入的字符串
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

// 输出标题
print(tab(33) + "BAGELS\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

// 初始化数组
a1 = [0,0,0,0];
a = [0,0,0,0];
b = [0,0,0,0];

y = 0;
t = 255;

// 输出空行
print("\n");
print("\n");
print("\n");

// 主程序
async function main()
{
    // 如果 y 等于 0，则输出提示信息
    if (y == 0)
        print("HOPE YOU HAD FUN.  BYE.\n");
}
    # 如果条件不满足，则执行以下代码
    else
        # 打印特定格式的字符串
        print("\nA " + y + " POINT BAGELS BUFF!!\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```