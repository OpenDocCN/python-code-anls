# `basic-computer-games\90_Tower\javascript\tower.js`

```

// TOWER
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，解析输入的字符串并返回
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

// 生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化数组
var ta = [];

// 打印函数，显示塔的状态
function show_towers()
{
    var z;

    for (var k = 1; k <= 7; k++) {
        z = 10;
        str = "";
        for (var j = 1; j <= 3; j++) {
            if (ta[k][j] != 0) {
                while (str.length < z - Math.floor(ta[k][j] / 2))
                    str += " ";
                for (v = 1; v <= ta[k][j]; v++)
                    str += "*";
            } else {
                while (str.length < z)
                    str += " ";
                str += "*";
            }
            z += 21;
        }
        print(str + "\n");
    }
}

// 主控制部分，使用 async 函数
async function main()
}

// 调用主函数
main();

```