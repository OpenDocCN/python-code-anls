# `basic-computer-games\65_Nim\javascript\nim.js`

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
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素的类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
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

// 初始化数组变量
var a = [];
var b = [];
var d = [];

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "NIM\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组 a 和 b
    for (i = 1; i <= 100; i++) {
        a[i] = 0;
        b[i] = [];
        for (j = 0; j <= 10; j++)
            b[i][j] = 0;
    }
    // 初始化数组 d
    d[0] = 0;
    d[1] = 0;
    d[2] = 0;
    // 打印提示信息
    print("DO YOU WANT INSTRUCTIONS");
}
    # 进入无限循环，直到条件为假
    while (1) {
        # 等待输入，并将输入字符串转换为大写
        str = await input();
        str = str.toUpperCase();
        # 如果输入为"YES"或者"NO"，则跳出循环
        if (str == "YES" || str == "NO")
            break;
        # 如果输入既不是"YES"也不是"NO"，则打印提示信息并继续循环
        print("PLEASE ANSWER YES OR NO\n");
    }
    # 如果输入为"YES"
    if (str == "YES") {
        # 打印游戏规则说明
        print("THE GAME IS PLAYED WITH A NUMBER OF PILES OF OBJECTS.\n");
        print("ANY NUMBER OF OBJECTS ARE REMOVED FROM ONE PILE BY YOU AND\n");
        print("THE MACHINE ALTERNATELY.  ON YOUR TURN, YOU MAY TAKE\n");
        print("ALL THE OBJECTS THAT REMAIN IN ANY PILE, BUT YOU MUST\n");
        print("TAKE AT LEAST ONE OBJECT, AND YOU MAY TAKE OBJECTS FROM\n");
        print("ONLY ONE PILE ON A SINGLE TURN.  YOU MUST SPECIFY WHETHER\n");
        print("WINNING IS DEFINED AS TAKING OR NOT TAKING THE LAST OBJECT,\n");
        print("THE NUMBER OF PILES IN THE GAME, AND HOW MANY OBJECTS ARE\n");
        print("ORIGINALLY IN EACH PILE.  EACH PILE MAY CONTAIN A\n");
        print("DIFFERENT NUMBER OF OBJECTS.\n");
        print("THE MACHINE WILL SHOW ITS MOVE BY LISTING EACH PILE AND THE\n");
        print("NUMBER OF OBJECTS REMAINING IN THE PILES AFTER  EACH OF ITS\n");
        print("MOVES.\n");
    }
    }
# 定义游戏完成的函数
function game_completed()
{
    # 遍历数组 a，检查是否所有元素都为 0
    for (var i = 1; i <= n; i++) {
        # 如果存在非零元素，则游戏未完成，返回 false
        if (a[i] != 0)
            return false;
    }
    # 如果所有元素都为 0，则游戏完成，返回 true
    return true;
}

# 调用主函数
main();
```