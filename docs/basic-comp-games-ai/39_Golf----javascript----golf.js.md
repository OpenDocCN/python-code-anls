# `basic-computer-games\39_Golf\javascript\golf.js`

```py
// 定义一个打印函数，将字符串输出到指定的元素上
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
                       // 将输入框添加到指定元素上
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
                                                      // 输出输入的字符串
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

// 定义一些变量
var la = [];
var f;
var s1;
var g2;
var g3;
var x;

// 定义一个包含障碍物数据的数组
var hole_data = [
    361,4,4,2,389,4,3,3,206,3,4,2,500,5,7,2,
    408,4,2,4,359,4,6,4,424,4,4,2,388,4,4,4,
    196,3,7,2,400,4,7,2,560,5,7,2,132,3,2,2,
    357,4,4,4,294,4,2,4,475,5,2,3,375,4,4,2,
    180,3,6,2,550,5,6,6,
];

// 定义一个显示障碍物的函数
function show_obstacle()
{
    # 根据 la[x] 的值进行不同的操作
    switch (la[x]) {
        # 如果 la[x] 的值为 1，则打印 "FAIRWAY."
        case 1:
            print("FAIRWAY.\n");
            # 结束 switch 语句
            break;
        # 如果 la[x] 的值为 2，则打印 "ROUGH."
        case 2:
            print("ROUGH.\n");
            # 结束 switch 语句
            break;
        # 如果 la[x] 的值为 3，则打印 "TREES."
        case 3:
            print("TREES.\n");
            # 结束 switch 语句
            break;
        # 如果 la[x] 的值为 4，则打印 "ADJACENT FAIRWAY."
        case 4:
            print("ADJACENT FAIRWAY.\n");
            # 结束 switch 语句
            break;
        # 如果 la[x] 的值为 5，则打印 "TRAP."
        case 5:
            print("TRAP.\n");
            # 结束 switch 语句
            break;
        # 如果 la[x] 的值为 6，则打印 "WATER."
        case 6:
            print("WATER.\n");
            # 结束 switch 语句
            break;
    }
// 显示分数的函数
function show_score()
{
    // 将当前分数累加到总分上
    g2 += s1;
    // 打印总杆数和总分数
    print("TOTAL PAR FOR " + (f - 1) + " HOLES IS " + g3 + "  YOUR TOTAL IS " + g2 + "\n");
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(34) + "GOLF\n");
    // 打印创意计算的信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印欢迎词
    print("WELCOME TO THE CREATIVE COMPUTING COUNTRY CLUB,\n");
    print("AN EIGHTEEN HOLE CHAMPIONSHIP LAYOUT LOCATED A SHORT\n");
    print("DISTANCE FROM SCENIC DOWNTOWN MORRISTOWN.  THE\n");
    print("COMMENTATOR WILL EXPLAIN THE GAME AS YOU PLAY.\n");
    print("ENJOY YOUR GAME; SEE YOU AT THE 19TH HOLE...\n");
    print("\n");
    print("\n");
    // 初始化变量
    next_hole = 0;
    g1 = 18;
    g2 = 0;
    g3 = 0;
    a = 0;
    n = 0.8;
    s2 = 0;
    f = 1;
    // 循环直到输入合法的手球杆数
    while (1) {
        print("WHAT IS YOUR HANDICAP");
        h = parseInt(await input());
        print("\n");
        if (h < 0 || h > 30) {
            print("PGA HANDICAPS RANGE FROM 0 TO 30.\n");
        } else {
            break;
        }
    }
    // 循环直到输入合法的困难程度
    do {
        print("DIFFICULTIES AT GOLF INCLUDE:\n");
        print("0=HOOK, 1=SLICE, 2=POOR DISTANCE, 4=TRAP SHOTS, 5=PUTTING\n");
        print("WHICH ONE (ONLY ONE) IS YOUR WORST");
        t = parseInt(await input());
        print("\n");
    } while (t > 5) ;
    // 初始化变量
    s1 = 0;
    first_routine = true;
    // 调用主程序
}

main();
```