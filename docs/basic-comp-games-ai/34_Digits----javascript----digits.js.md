# `basic-computer-games\34_Digits\javascript\digits.js`

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
                                                      // 解析输入的字符串并返回
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

// 主程序，使用 async 关键字定义一个异步函数
async function main()
{
    // 打印游戏名称
    print(tab(33) + "DIGITS\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏说明
    print("THIS IS A GAME OF GUESSING.\n");
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0'");
    // 获取用户输入并转换为整数
    e = parseInt(await input());
}
    # 如果 e 不等于 0，则执行以下操作
    if (e != 0) {
        # 打印空行
        print("\n");
        # 打印提示信息
        print("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN\n");
        print("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.\n");
        print("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.\n");
        print("I WILL ASK FOR THEN TEN AT A TIME.\n");
        print("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR\n");
        print("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,\n");
        print("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER\n");
        print("THAN THAT *****\n");
        print("\n");
        print("\n");
    }
    # 初始化变量 a, b, c, m, k, l, n
    a = 0;
    b = 1;
    c = 3;
    m = [];
    k = [];
    l = [];
    n = [];
    # 打印空行
    print("\n");
    # 打印感谢信息
    print("THANKS FOR THE GAME.\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```