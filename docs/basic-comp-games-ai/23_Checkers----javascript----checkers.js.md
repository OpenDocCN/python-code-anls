# `basic-computer-games\23_Checkers\javascript\checkers.js`

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
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析输入的值
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

// 尝试计算机移动的函数
function try_computer()
{
    u = x + a;
    v = y + b;
    // 如果超出棋盘范围，则返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果目标位置为空，则执行移动评估函数
    if (s[u][v] == 0) {
        eval_move();
        return;
    }
    // 如果目标位置是自己的棋子，则返回
    if (s[u][v] < 0)    // Cannot jump over own pieces
        return;
    u += a;
    u += b;
    // 如果超出棋盘范围，则返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果目标位置为空，则执行移动评估函数
    if (s[u][v] == 0)
        eval_move();
}
// 计算移动的评估值
function eval_move()
{
    // 如果目标方格为0且原方格为-1，则增加2分
    if (v == 0 && s[x][y] == -1)
        q += 2;
    // 如果y和v的绝对值为2，则增加5分
    if (Math.abs(y - v) == 2)
        q += 5;
    // 如果y等于7，则减少2分
    if (y == 7)
        q -= 2;
    // 如果u等于0或7，则增加1分
    if (u == 0 || u == 7)
        q++;
    // 遍历周围的方格
    for (c = -1; c <= 1; c += 2) {
        // 如果u+c或v+g超出边界，则继续下一次循环
        if (u + c < 0 || u + c > 7 || v + g < 0)
            continue;
        // 如果周围的方格为计算机的棋子，则增加1分
        if (s[u + c][v + g] < 0) {    // Computer piece
            q++;
            continue;
        }
        // 如果周围的方格为对方的棋子，则减少2分
        if (u - c < 0 || u - c > 7 || v - g > 7)
            continue;
        if (s[u + c][v + g] > 0 && (s[u - c][v - g] == 0 || (u - c == x && v - g == y)))
            q -= 2;
    }
    // 如果评估值大于最佳分数，则更新最佳分数和相关坐标
    if (q > r[0]) {    // Best movement so far?
        r[0] = q;    // Take note of score
        r[1] = x;    // Origin square
        r[2] = y;
        r[3] = u;    // Target square
        r[4] = v;
    }
    q = 0;
}

// 检查是否有更多的跳吃
function more_captures() {
    u = x + a;
    v = y + b;
    // 如果目标方格超出边界，则返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果目标方格为空且跳吃的中间方格为对方的棋子，则调用eval_move函数
    if (s[u][v] == 0 && s[x + a / 2][y + b / 2] > 0)
        eval_move();
}

// 初始化变量
var r = [-99, 0, 0, 0, 0];
var s = [];

// 初始化棋盘
for (x = 0; x <= 7; x++)
    s[x] = [];

var g = -1;
var data = [1, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, -1, 15];
var p = 0;
var q = 0;

// 主程序
async function main()
{
    // 输出游戏标题
    print(tab(32) + "CHECKERS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,\n");
    print("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.\n");
    print("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.\n");
    print("(0,0) IS THE LOWER LEFT CORNER\n");
    print("(0,7) IS THE UPPER LEFT CORNER\n");
    print("(7,0) IS THE LOWER RIGHT CORNER\n");
    print("(7,7) IS THE UPPER RIGHT CORNER\n");
    print("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER\n");
    print("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.\n");
    print("\n");
    print("\n");
    print("\n");
}
    # 循环遍历 x 的取值范围为 0 到 7
    for (x = 0; x <= 7; x++) {
        # 循环遍历 y 的取值范围为 0 到 7
        for (y = 0; y <= 7; y++) {
            # 如果 data[p] 的值为 15，则将 p 的值设为 0
            if (data[p] == 15)
                p = 0;
            # 将 data[p] 的值赋给二维数组 s 的元素 s[x][y]
            s[x][y] = data[p];
            # p 自增 1
            p++;
        }
    }
    # 循环结束
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```