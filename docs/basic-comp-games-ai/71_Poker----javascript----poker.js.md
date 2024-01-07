# `basic-computer-games\71_Poker\javascript\poker.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    // 创建输入元素
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并移除输入元素，然后解析输入值
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

// 定义一个生成空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一系列变量
var aa = [];
var ba = [];
var b;
var c;
var d;
var g;
var i;
var k;
var m;
var n;
var p;
var s;
var u;
var v;
var x;
var z;
var hs;
var is;
var js;
var ks;

// 定义一个生成随机数的函数
function fna(x)
{
    return Math.floor(10 * Math.random());
}

// 定义一个取余数的函数
function fnb(x)
{
    return x % 100;
}

// 定义一个函数，打印"I'M BUSTED.  CONGRATULATIONS!"
function im_busted()
{
    print("I'M BUSTED.  CONGRATULATIONS!\n");
}

// 1740
// 定义发牌函数
function deal_card()
{
    // 循环直到发出一张合法的牌
    while (1) {
        // 生成一张牌
        aa[z] = 100 * Math.floor(4 * Math.random()) + Math.floor(100 * Math.random());
        // 检查花色是否合法
        if (Math.floor(aa[z] / 100) > 3)    // Invalid suit
            continue;
        // 检查数字是否合法
        if (aa[z] % 100 > 12) // Invalid number
            continue;
        // 检查是否有重复的牌
        if (z != 1) {
            for (k = 1; k <= z - 1; k++) {
                if (aa[z] == aa[k])
                    break;
            }
            if (k <= z - 1) // Repeated card
                continue;
            // 如果发出的牌数大于10，则交换牌
            if (z > 10) {
                n = aa[u];
                aa[u] = aa[z];
                aa[z] = n;
            }
        }
        return;
    }
}

// 1850
// 展示手牌
function show_cards()
{
    for (z = n; z <= n + 4; z++) {
        print(" " + z + "--  ");
        k = fnb(aa[z]);
        show_number();
        print(" OF");
        k = Math.floor(aa[z] / 100);
        show_suit();
        if (z % 2 == 0)
            print("\n");
    }
    print("\n");
}

// 1950
// 展示牌的数字
function show_number()
{
    // 根据数字打印对应的牌面
    if (k == 9)
        print("JACK");
    if (k == 10)
        print("QUEEN");
    if (k == 11)
        print("KING");
    if (k == 12)
        print("ACE");
    if (k < 9)
        print(" " + (k + 2));
}

// 2070
// 展示牌的花色
function show_suit()
{
    // 根据花色打印对应的花色名称
    if (k == 0)
        print(" CLUBS\t");
    if (k == 1)
        print(" DIAMONDS\t");
    if (k == 2)
        print(" HEARTS\t");
    if (k == 3)
        print(" SPADES\t");
}

// 2170
// 评估手牌
function evaluate_hand()
{
    // 初始化变量
    u = 0;
    // 遍历手牌，计算相同花色的数量
    for (z = n; z <= n + 4; z++) {
        ba[z] = fnb(aa[z]);
        if (z != n + 4) {
            if (Math.floor(aa[z] / 100) == Math.floor(aa[z + 1] / 100))
                u++;
        }
    }
    // 根据相同花色的数量判断手牌类型
    // ...
}

// 其他函数和变量的作用需要根据具体代码逻辑进行分析

```