# `basic-computer-games\17_Bullfight\javascript\bullfight.js`

```

// BULLFIGHT
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串输出到指定元素
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，移除输入框，输出输入的字符串，并解析 Promise
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

// 定义缩进函数，返回指定数量空格组成的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义全局变量
var a;
var b;
var c;
var l;
var t;
var as;
var bs;
var d = [];
var ls = [, "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"];

// 定义函数 af，返回一个随机整数
function af(k)
{
    return Math.floor(Math.random() * 2 + 1);
}

// 定义函数 cf，返回一个随机数
function cf(q)
{
    return df(q) * Math.random();
}

// 定义函数 df，返回一个计算结果
function df(q)
{
    return (4.5 + l / 6 - (d[1] + d[2]) * 2.5 + 4 * d[4] + 2 * d[5] - Math.pow(d[3], 2) / 120 - a);
}

// 定义设置辅助函数
function setup_helpers()
{
    // 根据全局变量 a 计算 b 和 c 的值
    b = 3 / a * Math.random();
    if (b < 0.37)
        c = 0.5;
    else if (b < 0.5)
        c = 0.4;
    else if (b < 0.63)
        c = 0.3;
    else if (b < 0.87)
        c = 0.2;
    else
        c = 0.1;
    // 根据 c 的值计算 t 的值
    t = Math.floor(10 * c + 0.2);
    // 输出结果
    print("THE " + as + bs + " DID A " + ls[t] + " JOB.\n");
    if (4 <= t) {
        if (5 != t) {
            // Lines 1800 and 1810 of original program are unreachable
            switch (af(0)) {
                case 1:
                    print("ONE OF THE " + as + bs + " WAS KILLED.\n");
                    break;
                case 2:
                    print("NO " + as + b + " WERE KILLED.\n");
                    break;
            }
        } else {
            if (as != "TOREAD")
                print(af(0) + " OF THE HORSES OF THE " + as + bs + " KILLED.\n");
            print(af(0) + " OF THE " + as + bs + " KILLED.\n");
        }
    }
    print("\n");
}

// 主程序
async function main()
}


main();

```