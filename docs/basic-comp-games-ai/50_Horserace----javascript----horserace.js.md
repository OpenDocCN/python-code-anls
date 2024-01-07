# `basic-computer-games\50_Horserace\javascript\horserace.js`

```

// HORSERACE
// 赛马游戏的Javascript版本，由Oscar Toledo G. (nanochess)从BASIC转换而来

// 打印函数，将字符串输出到指定的元素上
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个Promise对象，当输入完成时resolve
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
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

// 生成指定数量空格的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化变量数组
var sa = [];
var ws = [];
var da = [];
var qa = [];
var pa = [];
var ma = [];
var ya = [];
var vs = [];

// 主程序
async function main()
}

main();

```