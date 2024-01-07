# `basic-computer-games\46_Hexapawn\javascript\hexapawn.js`

```

// 定义一个打印函数，用于在页面上输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象，用于获取用户输入
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 在页面上输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到页面上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从页面上移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 在页面上输出输入的字符串
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

// 初始化棋盘和棋子的数组
var ba = [,
          [,-1,-1,-1,1,0,0,0,1,1],
          ...
          ];
var ma = [,
          [,24,25,36,0],
          ...
          ];
var s = [];
var t = [];
var ps = "X.O";

// 显示棋盘的函数
function show_board()
{
    print("\n");
    for (var i = 1; i <= 3; i++) {
        print(tab(10));
        for (var j = 1; j <= 3; j++) {
            print(ps[s[(i - 1) * 3 + j] + 1]);
        }
        print("\n");
    }
}

// 镜像函数，用于计算棋子的镜像位置
function mirror(x)
{
    if (x == 1)
        return 3;
    if (x == 3)
        return 1;
    if (x == 6)
        return 4;
    if (x == 4)
        return 6;
    if (x == 9)
        return 7;
    if (x == 7)
        return 9;
    return x;
}

// 主程序，使用 async 函数声明，表示其中可能包含异步操作
async function main()
}

// 调用主程序
main();

```