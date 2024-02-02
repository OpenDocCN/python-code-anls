# `basic-computer-games\35_Even_Wins\javascript\evenwins.js`

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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义两个空数组
var ma = [];
var ya = [];

// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(31) + "EVEN WINS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    y1 = 0;
    m1 = 0;
    // 打印游戏说明
    print("     THIS IS A TWO PERSON GAME CALLED 'EVEN WINS.'\n");
    print("TO PLAY THE GAME, THE PLAYERS NEED 27 MARBLES OR\n");
    print("OTHER OBJECTS ON A TABLE.\n");
    print("\n");
    print("\n");
}
    # 打印游戏规则说明
    print("     THE 2 PLAYERS ALTERNATE TURNS, WITH EACH PLAYER\n");
    print("REMOVING FROM 1 TO 4 MARBLES ON EACH MOVE.  THE GAME\n");
    print("ENDS WHEN THERE ARE NO MARBLES LEFT, AND THE WINNER\n");
    print("IS THE ONE WITH AN EVEN NUMBER OF MARBLES.\n");
    print("\n");
    print("\n");
    # 打印游戏规则说明
    print("     THE ONLY RULES ARE THAT (1) YOU MUST ALTERNATE TURNS,\n");
    print("(2) YOU MUST TAKE BETWEEN 1 AND 4 MARBLES EACH TURN,\n");
    print("AND (3) YOU CANNOT SKIP A TURN.\n");
    print("\n");
    print("\n");
    print("\n");
    }
    # 打印结束语
    print("\n");
    print("OK.  SEE YOU LATER\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```