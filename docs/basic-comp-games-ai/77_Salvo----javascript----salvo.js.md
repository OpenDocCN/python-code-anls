# `basic-computer-games\77_Salvo\javascript\salvo.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
        // 创建一个输入框元素
        input_element = document.createElement("INPUT");

        // 在输出元素中添加提示符
        print("? ");

        // 设置输入框的类型和长度
        input_element.setAttribute("type", "text");
        input_element.setAttribute("length", "50");

        // 将输入框添加到输出元素中
        document.getElementById("output").appendChild(input_element);

        // 让输入框获得焦点
        input_element.focus();

        // 初始化输入字符串
        input_str = undefined;

        // 监听输入框的键盘事件
        input_element.addEventListener("keydown", function (event) {
            // 如果按下的是回车键
            if (event.keyCode == 13) {
                // 获取输入的字符串
                input_str = input_element.value;

                // 从输出元素中移除输入框
                document.getElementById("output").removeChild(input_element);

                // 将输入的字符串打印到输出元素中
                print(input_str);
                print("\n");

                // 解析输入的字符串并返回
                resolve(input_str);
            }
        });
    });
}

// 定义一个生成指定空格数的字符串的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 声明一系列变量
var aa = [];
var ba = [];
var ca = [];
var da = [];
var ea = [];
var fa = [];
var ga = [];
var ha = [];
var ka = [];
var w;
var r3;
var x;
var y;
var v;
var v2;

// 定义一个返回数值的符号函数
function sgn(k)
{
    if (k < 0)
        return -1;
    if (k > 0)
        return 1;
    return 0;
}

// 定义一个数学函数
function fna(k)
{
    return (5 - k) * 3 - 2 * Math.floor(k / 4) + sgn(k - 1) - 1;
}

// 定义另一个数学函数
function fnb(k)
{
    return k + Math.floor(k / 4) - sgn(k - 1);
}

// 定义一个生成随机数的函数
function generate_random()
{
    x = Math.floor(Math.random() * 10 + 1);
    y = Math.floor(Math.random() * 10 + 1);
    v = Math.floor(3 * Math.random() - 1);
    v2 = Math.floor(3 * Math.random() - 1);
}

// 主程序
async function main()
{
    // 调用主函数
}

// 调用主程序
main();

```