# `basic-computer-games\40_Gomoko\javascript\gomoko.js`

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
                       // 设置输入框类型为文本，长度为50
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
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表函数，返回指定空格数的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 重置统计数据
function reset_stats()
{
    for (var j = 1; j <= 4; j++)
        f[j] = 0;
}

// 定义一个数组和三个变量
var a = [];
var x;
var y;
var n;

// *** 打印棋盘 ***
function print_board()
{
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n; j++) {
            // 打印棋盘上每个位置的值
            print(" " + a[i][j] + " ");
        }
        // 换行
        print("\n");
    }
    // 打印空行
    print("\n");
}

// 检查移动是否有效
function is_valid()
{
    if (x < 1 || x > n || y < 1 || y > n)
        return false;
    # 返回布尔值 True
    return True;
// 主程序
async function main()
{
    // 打印游戏标题
    print(tab(33) + "GOMOKO\n");
    // 打印游戏信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化游戏棋盘
    for (i = 0; i <= 19; i++) {
        a[i] = [];
        for (j = 0; j <= 19; j++)
            a[i][j] = 0;
    }
    // 打印游戏规则
    print("WELCOME TO THE ORIENTAL GAME OF GOMOKO.\n");
    print("\n");
    print("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE\n");
    print("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID\n");
    print("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET\n");
    print("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR\n");
    print("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED\n");
    print("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.\n");
    print("\n");
    print("THE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.\n");
    print("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n");
    print("\n");
    }
}

main();
```