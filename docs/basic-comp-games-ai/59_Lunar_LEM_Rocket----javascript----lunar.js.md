# `basic-computer-games\59_Lunar_LEM_Rocket\javascript\lunar.js`

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
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的字符串
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

// 定义一系列变量
var l;
var t;
var m;
var s;
var k;
var a;
var v;
var i;
var j;
var q;
var g;
var z;
var d;

// 定义一个函数，设置一组公式
function formula_set_1()
{
    l = l + s;
    t = t - s;
    m = m - s * k;
    a = i;
    v = j;
}

// 定义另一组公式
function formula_set_2()
{
    q = s * k / m;
    j = v + g * s + z * (-q - q * q / 2 - Math.pow(q, 3) / 3 - Math.pow(q, 4) / 4 - Math.pow(q, 5) / 5);
    # 计算变量 i 的值
    i = a - g * s * s / 2 - v * s + z * s * (q / 2 + Math.pow(q, 2) / 6 + Math.pow(q, 3) / 12 + Math.pow(q, 4) / 20 + Math.pow(q, 5) / 30);
// 定义函数 formula_set_3，计算并更新变量 s 直到其小于 5e-3
function formula_set_3()
{
    // 当 s 大于等于 5e-3 时执行循环
    while (s >= 5e-3) {
        // 根据给定公式计算 d
        d = v + Math.sqrt(v * v + 2 * a * (g - z * k / m));
        // 根据给定公式计算 s
        s = 2 * a / d;
        // 调用 formula_set_2 函数
        formula_set_2();
        // 调用 formula_set_1 函数
        formula_set_1();
    }
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "LUNAR\n");
    // 打印信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印模拟信息
    print("THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR\n");
    print("LANDING CAPSULE.\n");
    // 打印空行
    print("\n");
    print("\n");
    // 打印故障信息
    print("THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY\n");
    print("XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY.\n");
}

// 调用主程序
main();
```