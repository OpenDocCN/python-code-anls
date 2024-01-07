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
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 在输出元素中显示提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 换行
                                                      print("\n");
                                                      // 解析输入的字符串
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

// 尝试计算机移动
function try_computer()
{
    // 计算目标位置
    u = x + a;
    v = y + b;
    // 如果目标位置超出边界，返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果目标位置为空，执行移动
    if (s[u][v] == 0) {
        eval_move();
        return;
    }
    // 如果目标位置有自己的棋子，返回
    if (s[u][v] < 0)	
        return;
    // 计算跳跃后的位置
    u += a;
    u += b;
    // 如果跳跃后的位置超出边界，返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果跳跃后的位置为空，执行移动
    if (s[u][v] == 0)
        eval_move();
}

// 评估移动
function eval_move()
{
    // 根据条件对移动进行评分
    // ...
}

// 检查是否有更多的跳跃
function more_captures() {
    // 计算目标位置
    u = x + a;
    v = y + b;
    // 如果目标位置超出边界，返回
    if (u < 0 || u > 7 || v < 0 || v > 7)
        return;
    // 如果目标位置为空且中间有对方的棋子，执行移动
    if (s[u][v] == 0 && s[x + a / 2][y + b / 2] > 0)
        eval_move();
}

// 初始化变量
var r = [-99, 0, 0, 0, 0];
var s = [];
// ...

// 主程序
async function main()
}

// 调用主程序
main();

```