# `basic-computer-games\36_Flip_Flop\javascript\flipflop.js`

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
                                                      // 解析 Promise 对象
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

// 定义一个空数组
var as = [];

// 主程序
async function main()
{
    // 打印标题
    print(tab(32) + "FLIPFLOP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    // 打印提示信息
    print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:\n");
    print("\n");
    print("X X X X X X X X X X\n");
    print("\n");
    print("TO THIS:\n");
    print("\n");
    print("O O O O O O O O O O\n");
    print("\n");
}
    # 打印提示信息，告知用户如何进行操作
    print("BY TYPING THE NUMBER CORRESPONDING TO THE POSITION OF THE\n");
    # 打印提示信息，告知用户如何进行操作
    print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON\n");
    # 打印提示信息，告知用户如何进行操作
    print("OTHERS, TWO WILL CHANGE.  TO RESET LINE TO ALL X'S, TYPE 0\n");
    # 打印提示信息，告知用户如何进行操作
    print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE \n");
    # 打印提示信息，告知用户如何进行操作
    print("11 (ELEVEN).\n");
    # 打印空行
    print("\n");
    # 打印空行
    }
    # 打印空行
    print("\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```