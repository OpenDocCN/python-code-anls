# `basic-computer-games\67_One_Check\javascript\onecheck.js`

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
                                                      // 从输出元素中移除输入框
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

// 定义一个空数组
var a = [];

// 主程序
async function main()
{
    // 打印标题
    print(tab(30) + "ONE CHECK\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组
    for (i = 0; i <= 64; i++)
        a[i] = 0;
    // 打印游戏说明
    print("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n");
    print("\n");
    print("48 CHECKERS ARE PLACED ON THE 2 OUTSIDE SPACES OF A\n");
}
    # 打印标准的 64 方格棋盘，目标是通过对角跳吃掉尽可能多的棋子
    print("STANDARD 64-SQUARE CHECKERBOARD.  THE OBJECT IS TO\n");
    # 打印游戏规则说明
    print("REMOVE AS MANY CHECKERS AS POSSIBLE BY DIAGONAL JUMPS\n");
    print("(AS IN STANDARD CHECKERS).  USE THE NUMBERED BOARD TO\n");
    print("INDICATE THE SQUARE YOU WISH TO JUMP FROM AND TO.  ON\n");
    print("THE BOARD PRINTED OUT ON EACH TURN '1' INDICATES A\n");
    print("CHECKER AND '0' AN EMPTY SQUARE.  WHEN YOU HAVE NO\n");
    print("POSSIBLE JUMPS REMAINING, INPUT A '0' IN RESPONSE TO\n");
    print("QUESTION 'JUMP FROM ?'\n");
    print("\n");
    # 打印数字棋盘
    print("HERE IS THE NUMERICAL BOARD:\n");
    print("\n");
    }
    # 打印结束语
    print("\n");
    print("O.K.  HOPE YOU HAD FUN!!\n");
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```