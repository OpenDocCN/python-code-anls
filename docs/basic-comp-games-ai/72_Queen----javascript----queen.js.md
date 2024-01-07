# `basic-computer-games\72_Queen\javascript\queen.js`

```

// 定义一个打印函数，用于在页面上输出字符串
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
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 在页面上输出输入的值
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个函数，用于生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化变量
var sa = [,81,  71,  61,  51,  41,  31,  21,  11,
           92,  82,  72,  62,  52,  42,  32,  22,
          103,  93,  83,  73,  63,  53,  43,  33,
          114, 104,  94,  84,  74,  64,  54,  44,
          125, 115, 105,  95,  85,  75,  65,  55,
          136, 126, 116, 106,  96,  86,  76,  66,
          147, 137, 127, 117, 107,  97,  87,  77,
          158, 148, 138, 128, 118, 108,  98,  88];

var m;
var m1;
var u;
var t;
var u1;
var t1;

// 显示游戏说明
function show_instructions()
{
    // 在页面上输出游戏说明
    print("WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS\n");
    print("MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,\n");
    print("DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.\n");
    // ... 其他说明内容
}

// 显示游戏地图
function show_map()
{
    // 输出游戏地图
    // ... 输出地图内容
}

// 测试移动是否合法
function test_move()
{
    // 判断移动是否合法
    // ... 判断逻辑
}

// 随机移动
function random_move()
{
    // 随机移动
    // ... 随机移动逻辑
}

// 计算机移动
function computer_move()
{
    // 计算机移动
    // ... 计算机移动逻辑
}

// 主程序
async function main()
{
    // 在页面上输出标题
    print(tab(33) + "QUEEN\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");

    // 循环获取用户是否需要游戏说明
    // ... 获取用户输入逻辑

    // 如果需要游戏说明，则显示游戏说明
    if (str == "YES")
        show_instructions();

    // 循环进行游戏
    // ... 游戏逻辑
}

// 调用主程序
main();

```