# `basic-computer-games\67_One_Check\javascript\onecheck.js`

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
                       // 监听键盘事件，当按下回车键时，将输入的值添加到输出元素中，并解析为输入字符串
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

// 定义一个空数组
var a = [];

// 主程序，使用 async 函数定义
async function main()
{
    // 打印标题
    print(tab(30) + "ONE CHECK\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组 a
    for (i = 0; i <= 64; i++)
        a[i] = 0;
    // 打印游戏说明
    print("SOLITAIRE CHECKER PUZZLE BY DAVID AHL\n");
    // ...（以下为打印游戏规则和操作提示的部分，略）

    // 主循环
    while (1) {
        // ...（以下为游戏逻辑的部分，略）
    }
    // ...（以下为游戏结束后的部分，略）
}

// 调用主程序
main();

```