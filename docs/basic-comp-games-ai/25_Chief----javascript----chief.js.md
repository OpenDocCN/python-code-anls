# `basic-computer-games\25_Chief\javascript\chief.js`

```

// 定义一个打印函数，将字符串输出到指定的元素中
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序，使用 async 函数定义
async function main()
{
    // 输出标题
    print(tab(30) + "CHIEF\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I AM CHIEF NUMBERS FREEK, THE GREAT INDIAN MATH GOD.\n");
    print("ARE YOU READY TO TAKE THE TEST YOU CALLED ME OUT FOR");
    // 等待用户输入，如果输入不以"Y"开头，则输出指定字符串
    a = await input();
    if (a.substr(0, 1) != "Y")
        print("SHUT UP, PALE FACE WITH WIE TONGUE.\n");
    // 输出一系列数学问题，并等待用户输入
    // 根据用户输入计算结果并输出
    // 再次等待用户输入，根据输入判断输出指定字符串
    // 如果输入不以"Y"开头，则再次等待用户输入，并进行一系列计算和输出
    // 最后根据用户输入输出指定字符串
    // 如果输入不以"Y"开头，则输出一系列字符串和图形
    // 最后输出指定字符串
    // 如果输入以"Y"开头，则输出指定字符串
    // 最后输出指定字符串
}

// 调用主程序
main();

```