# `basic-computer-games\15_Boxing\javascript\boxing.js`

```

// BOWLING
// 保龄球游戏的Javascript版本，由Oscar Toledo G. (nanochess)从BASIC转换而来

// 打印输出
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数
function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象
    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
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

// 缩进函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序
async function main()
}

// 调用主程序
main();

```