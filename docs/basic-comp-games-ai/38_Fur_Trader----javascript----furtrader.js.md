# `basic-computer-games\38_Fur_Trader\javascript\furtrader.js`

```

// FUR TRADER
// 皮毛交易程序

// 将 BASIC 转换为 Javascript 由 Oscar Toledo G. (nanochess) 完成

// 打印输出
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入
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

// 缩进
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 变量初始化
var f = [];
var bs = [, "MINK", "BEAVER", "ERMINE", "FOX"];

// 重置统计数据
function reset_stats()
{
    for (var j = 1; j <= 4; j++)
        f[j] = 0;
}

// 主程序
async function main()
}

main();

```