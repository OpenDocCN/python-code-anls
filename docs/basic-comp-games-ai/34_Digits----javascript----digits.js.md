# `basic-computer-games\34_Digits\javascript\digits.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象，当用户输入完成后 resolve
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听用户输入，当按下回车键时，将输入的值返回并 resolve
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
{
    // 输出标题
    print(tab(33) + "DIGITS\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS IS A GAME OF GUESSING.\n");
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0'");
    // 获取用户输入，转换为整数
    e = parseInt(await input());
    if (e != 0) {
        // 输出游戏说明
        // ...
    }
    // 初始化变量
    a = 0;
    b = 1;
    c = 3;
    m = [];
    k = [];
    l = [];
    n = [];
    // 进入游戏循环
    while (1) {
        // ...
    }
    // ...
}

// 调用主程序
main();

```