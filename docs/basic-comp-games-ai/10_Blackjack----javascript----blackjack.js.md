# `basic-computer-games\10_Blackjack\javascript\blackjack.js`

```

// BLACKJACK
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 输入函数，返回一个 Promise 对象，当用户输入完成时解析
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

// 生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化牌堆和玩家手牌
var da = [];
var pa = [];
var qa = [];
var ca = [];
var ta = [];
var sa = [];
var ba = [];
var za = [];
var ra = [];

// 牌面和花色
var ds = "N A  2  3  4  5  6  7N 8  9 10  J  Q  K";
var is = "H,S,D,/,"

var q;
var aa;
var ab;
var ac;
var h;
var h1;

// 计算牌面点数，如果大于 22 则减去 11
function af(q) {
    return q >= 22 ? q - 11 : q;
}

// 重新洗牌
function reshuffle()
{
    print("RESHUFFLING\n");
    for (; d >= 1; d--)
        ca[--c] = da[d];
    for (c1 = 52; c1 >= c; c1--) {
        c2 = Math.floor(Math.random() * (c1 - c + 1)) + c;
        c3 = ca[c2];
        ca[c2] = ca[c1];
        ca[c1] = c3;
    }
}

// 获取一张牌
function get_card()
{
    if (c >= 51)
        reshuffle();
    return ca[c++];
}

// 打印牌面
function card_print(x)
{
    print(ds.substr(3 * x - 3, 3) + "  ");
}

// 替代的打印牌面
function alt_card_print(x)
{
    print(" " + ds.substr(3 * x - 2, 2) + "   ");
}

// 添加牌面到总点数
function add_card(which)
{
    x1 = which;
    if (x1 > 10)
        x1 = 10;
    q1 = q + x1;
    if (q < 11) {
        if (which <= 1) {
            q += 11;
            return;
        }
        if (q1 >= 11)
            q = q1 + 11;
        else
            q = q1;
        return;
    }
    if (q <= 21 && q1 > 21)
        q = q1 + 1;
    else
        q = q1;
    if (q >= 33)
        q = -1;
}

// 计算手牌总点数
function evaluate_hand(which)
{
    q = 0;
    for (q2 = 1; q2 <= ra[which]; q2++) {
        add_card(pa[i][q2]);
    }
    qa[which] = q;
}

// 添加一张牌到手牌
function add_card_to_row(i, x) {
    ra[i]++;
    pa[i][ra[i]] = x;
    q = qa[i];
    add_card(x);
    qa[i] = q;
    if (q < 0) {
        print("...BUSTED\n");
        discard_row(i);
    }
}

// 丢弃手牌
function discard_row(i) {
    while (ra[i]) {
        d++;
        da[d] = pa[i][ra[i]];
        ra[i]--;
    }
}

// 打印手牌总点数
function print_total(i) {
    print("\n");
    aa = qa[i];
    total_aa();
    print("TOTAL IS " + aa + "\n");
}

// 计算总点数
function total_aa()
{
    if (aa >= 22)
        aa -= 11;
}

function total_ab()
{
    if (ab >= 22)
        ab -= 11;
}

function total_ac()
{
    if (ac >= 22)
        ac -= 11;
}

// 处理用户输入
function process_input(str)
{
    str = str.substr(0, 1);
    for (h = 1; h <= h1; h += 2) {
        if (str == is.substr(h - 1, 1))
            break;
    }
    if (h <= h1) {
        h = (h + 1) / 2;
        return 0;
    }
    print("TYPE " + is.substr(0, h1 - 1) + " OR " + is.substr(h1 - 1, 2) + " PLEASE");
    return 1;
}

// 主程序
async function main()
}

main();

```