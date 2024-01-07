# `basic-computer-games\73_Reverse\javascript\reverse.js`

```

// 定义一个打印函数，用于在页面上输出字符串
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
                       input_element = document.createElement("INPUT");

                       // 在页面上输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并解析为字符串
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

// 定义一个函数，用于生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个空数组和一个变量
var a = [];
var n;

// 打印游戏规则的子程序
function print_rules()
{
    // 打印游戏规则
    // ...
}

// 打印列表的子程序
function print_list()
{
    // 打印列表
    // ...
}

// 主程序
async function main()
{
    // 打印游戏标题和信息
    // ...

    // 初始化数组 a
    for (i = 0; i <= 20; i++)
        a[i] = 0;
    // 设置数字数量
    n = 9;
    // 询问是否需要打印游戏规则
    str = await input();
    if (str.toUpperCase() === "YES" || str.toUpperCase() === "Y")
        print_rules();
    while (1) {
        // 生成随机列表
        // ...
        // 打印原始列表并开始游戏
        // ...
        while (1) {
            // 获取用户输入并进行游戏操作
            // ...
        }
        // 询问是否再玩一次
        // ...
    }
    // 结束游戏
    print("\n");
    print("O.K. HOPE YOU HAD FUN!!\n");
}

// 调用主程序
main();

```