# `basic-computer-games\35_Even_Wins\javascript\gameofevenwins.js`

```

// 定义一个打印函数，用于在页面上输出字符串
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象，用于获取用户输入
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
                       // 监听用户输入，当按下回车键时，获取输入值并返回
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

// 初始化一个二维数组
var r = [[], []];

// 主程序，使用 async/await 来处理异步操作
async function main()
{
    // 输出游戏标题和说明
    print(tab(28) + "GAME OF EVEN WINS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("DO YOU WANT INSTRUCTIONS (YES OR NO)");
    // 获取用户输入
    str = await input();
    print("\n");
    if (str != "NO") {
        // 输出游戏规则说明
        print("THE GAME IS PLAYED AS FOLLOWS:\n");
        // ... (以下类似，输出游戏规则和提示信息)
    }
    // 初始化变量
    l = 0;
    b = 0;
    for (i = 0; i <= 5; i++) {
        r[1][i] = 4;
        r[0][i] = 4;
    }
    // 游戏循环
    while (1) {
        // ... (以下为游戏逻辑，包括计算和输出结果)
    }
}

// 调用主程序
main();

```