# `basic-computer-games\71_Poker\javascript\poker.js`

```
// POKER
//
// 由 Oscar Toledo G. (nanochess) 将 BASIC 转换为 Javascript
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
                       // 创建输入元素
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
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
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

// 初始化变量
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

// 生成 0-9 之间的随机整数
function fna(x)
{
    return Math.floor(10 * Math.random());
}

// 对输入的数取模 100
function fnb(x)
{
    return x % 100;
}

// 打印玩家爆牌的消息
function im_busted()
{
    print("I'M BUSTED.  CONGRATULATIONS!\n");
}

// 1740
// 发牌函数
function deal_card()
{
    # 进入无限循环，直到条件不满足才退出
    while (1) {
        # 生成一个随机数作为扑克牌的值
        aa[z] = 100 * Math.floor(4 * Math.random()) + Math.floor(100 * Math.random());
        # 如果花色大于3，则为无效花色，重新生成
        if (Math.floor(aa[z] / 100) > 3)    // Invalid suit
            continue;
        # 如果数字大于12，则为无效数字，重新生成
        if (aa[z] % 100 > 12) // Invalid number
            continue;
        # 如果不是第一张牌，则检查是否重复
        if (z != 1) {
            # 遍历之前的牌，检查是否有重复的牌
            for (k = 1; k <= z - 1; k++) {
                if (aa[z] == aa[k])
                    break;
            }
            # 如果有重复的牌，则重新生成
            if (k <= z - 1) // Repeated card
                continue;
            # 如果当前牌的位置大于10，则与第一张牌交换位置
            if (z > 10) {
                n = aa[u];
                aa[u] = aa[z];
                aa[z] = n;
            }
        }
        # 符合条件则退出循环
        return;
    }
// 展示玩家手中的五张牌
function show_cards()
{
    for (z = n; z <= n + 4; z++) {
        // 打印牌的编号
        print(" " + z + "--  ");
        // 获取牌的数字
        k = fnb(aa[z]);
        // 展示牌的数字
        show_number();
        // 打印" OF"
        print(" OF");
        // 获取牌的花色
        k = Math.floor(aa[z] / 100);
        // 展示牌的花色
        show_suit();
        // 如果是偶数张牌，换行
        if (z % 2 == 0)
            print("\n");
    }
    // 换行
    print("\n");
}

// 展示牌的数字
function show_number()
{
    // 如果是9，打印"JACK"
    if (k == 9)
        print("JACK");
    // 如果是10，打印"QUEEN"
    if (k == 10)
        print("QUEEN");
    // 如果是11，打印"KING"
    if (k == 11)
        print("KING");
    // 如果是12，打印"ACE"
    if (k == 12)
        print("ACE");
    // 如果小于9，打印数字
    if (k < 9)
        print(" " + (k + 2));
}

// 展示牌的花色
function show_suit()
{
    // 如果是0，打印" CLUBS\t"
    if (k == 0)
        print(" CLUBS\t");
    // 如果是1，打印" DIAMONDS\t"
    if (k == 1)
        print(" DIAMONDS\t");
    // 如果是2，打印" HEARTS\t"
    if (k == 2)
        print(" HEARTS\t");
    // 如果是3，打印" SPADES\t"
    if (k == 3)
        print(" SPADES\t");
}

// 评估玩家手中的牌
function evaluate_hand()
{
    // 初始化变量
    u = 0;
    // 遍历玩家手中的五张牌
    for (z = n; z <= n + 4; z++) {
        // 获取牌的数字
        ba[z] = fnb(aa[z]);
        // 如果不是最后一张牌
        if (z != n + 4) {
            // 如果相邻两张牌的花色相同，计数加一
            if (Math.floor(aa[z] / 100) == Math.floor(aa[z + 1] / 100))
                u++;
        }
    }
    // 如果计数为4
    if (u == 4) {
        // 设置特殊值
        x = 11111;
        d = aa[n];
        hs = "A FLUS";
        is = "H IN";
        u = 15;
        return;
    }
    // 对玩家手中的牌进行排序
    for (z = n; z <= n + 3; z++) {
        for (k = z + 1; k <= n + 4; k++) {
            if (ba[z] > ba[k]) {
                x = aa[z];
                aa[z] = aa[k];
                ba[z] = ba[k];
                aa[k] = x;
                ba[k] = aa[k] - 100 * Math.floor(aa[k] / 100);
            }
        }
    }
    // 初始化变量
    x = 0;
}
    # 循环遍历数组，从 n 到 n+3
    for (z = n; z <= n + 3; z++) {
        # 如果数组中当前元素等于下一个元素
        if (ba[z] == ba[z + 1]) {
            # 计算 x 的值
            x = x + 11 * Math.pow(10, z - n);
            # 将当前元素赋值给 d
            d = aa[z];
            # 如果 u 小于 11
            if (u < 11) {
                # 更新 u 的值为 11，设置 hs 和 is 的值
                u = 11;
                hs = "A PAIR";
                is = " OF ";
            } else if (u == 11) {
                # 如果 u 等于 11
                if (ba[z] == ba[z - 1]) {
                    # 设置 hs 和 is 的值，更新 u 的值
                    hs = "THREE";
                    is = " ";
                    u = 13;
                } else {
                    # 设置 hs 和 is 的值，更新 u 的值
                    hs = "TWO P";
                    is = "AIR, ";
                    u = 12;
                }
            } else if (u == 12) {
                # 更新 u 的值
                u = 16;
                hs = "FULL H";
                is = "OUSE, ";
            } else if (ba[z] == ba[z - 1]) {
                # 更新 u 的值，设置 hs 和 is 的值
                u = 17;
                hs = "FOUR";
                is = " ";
            } else {
                # 更新 u 的值，设置 hs 和 is 的值
                u = 16;
                hs = "FULL H";
                is = "OUSE. ";
            }
        }
    }
    # 如果 x 的值为 0
    if (x == 0) {
        # 如果数组中 n 和 n+3 的元素相差为 3
        if (ba[n] + 3 == ba[n + 3]) {
            # 更新 x 和 u 的值
            x = 1111;
            u = 10;
        }
        # 如果数组中 n+1 和 n+4 的元素相差为 3
        if (ba[n + 1] + 3 == ba[n + 4]) {
            # 如果 u 的值为 10
            if (u == 10) {
                # 更新 u、hs、is、x 和 d 的值，然后返回
                u = 14;
                hs = "STRAIG";
                is = "HT";
                x = 11111;
                d = aa[n + 4];
                return;
            }
            # 更新 u 和 x 的值
            u = 10;
            x = 11110;
        }
    }
    # 如果 u 小于 10
    if (u < 10) {
        # 更新 d、hs、is、u、x 和 i 的值，然后返回
        d = aa[n + 4];
        hs = "SCHMAL";
        is = "TZ, ";
        u = 9;
        x = 11000;
        i = 6;
        return;
    }
    # 如果 u 的值为 10
    if (u == 10) {
        # 如果 i 的值为 1，更新 i 的值
        if (i == 1)
            i = 6;
        return;
    }
    # 如果 u 大于 12，返回
    if (u > 12)
        return;
    # 如果 fnb(d) 大于 6，返回
    if (fnb(d) > 6)
        return;
    # 更新 i 的值
    i = 6;
# 结束当前的 JavaScript 函数
}

# 定义一个名为 get_prompt 的 JavaScript 函数，接受一个问题和默认值作为参数
function get_prompt(question, def)
{
    # 声明一个变量 str
    var str;

    # 调用 window.prompt 方法，显示问题并返回用户输入的值
    str = window.prompt(question, def);
    # 调用 print 方法，打印问题和用户输入的值
    print(question + "? " + str + "\n");
    # 返回用户输入的值
    return str;
}

# 定义一个名为 player_low_in_money 的 JavaScript 函数
function player_low_in_money()
{
    # 调用 print 方法，打印空行
    print("\n");
    # 调用 print 方法，打印提示信息
    print("YOU CAN'T BET WITH WHAT YOU HAVEN'T GOT.\n");
    # 声明一个变量 str，并赋值为 "N"
    str = "N";
    # 如果 o 除以 2 的余数不等于 0
    if (o % 2 != 0) {
        # 调用 get_prompt 方法，询问用户是否想要卖掉手表，如果用户没有输入则使用默认值 "YES"
        str = get_prompt("WOULD YOU LIKE TO SELL YOUR WATCH", "YES");
        # 如果用户输入的值的第一个字符不是 "N"
        if (str.substr(0, 1) != "N") {
            # 如果调用 fna 方法返回的值小于 7
            if (fna(0) < 7) {
                # 调用 print 方法，打印信息
                print("I'LL GIVE YOU $75 FOR IT.\n");
                # 将 s 的值增加 75
                s += 75;
            } else {
                # 调用 print 方法，打印信息
                print("THAT'S A PRETTY CRUMMY WATCH - I'LL GIVE YOU $25.\n");
                # 将 s 的值增加 25
                s += 25;
            }
            # 将 o 的值乘以 2
            o *= 2;
        }
    }
    # 如果 o 除以 3 的余数等于 0 并且用户输入的值的第一个字符是 "N"
    if (o % 3 == 0 && str.substr(0, 1) == "N") {
        # 调用 get_prompt 方法，询问用户是否想要卖掉领带夹，如果用户没有输入则使用默认值 "YES"
        str = get_prompt("WILL YOU PART WITH THAT DIAMOND TIE TACK", "YES");
        # 如果用户输入的值的第一个字符不是 "N"
        if (str.substr(0, 1) != "N") {
            # 如果调用 fna 方法返回的值小于 6
            if (fna(0) < 6) {
                # 调用 print 方法，打印信息
                print("YOU ARE NOW $100 RICHER.\n");
                # 将 s 的值增加 100
                s += 100;
            } else {
                # 调用 print 方法，打印信息
                print("IT'S PASTE.  $25.\n");
                # 将 s 的值增加 25
                s += 25;
            }
            # 将 o 的值乘以 3
            o *= 3;
        }
    }
    # 如果用户输入的值的第一个字符是 "N"
    if (str.substr(0,1) == "N") {
        # 调用 print 方法，打印信息
        print("YOUR WAD IS SHOT.  SO LONG, SUCKER!\n");
        # 返回 true
        return true;
    }
    # 返回 false
    return false;
}

# 定义一个名为 computer_low_in_money 的 JavaScript 函数
function computer_low_in_money()
{
    # 如果 c 减去 g 减去 v 大于等于 0，则返回 false
    if (c - g - v >= 0)
        return false;
    # 如果 g 的值等于 0
    if (g == 0) {
        # 将 v 的值设置为 c
        v = c;
        # 返回 false
        return false;
    }
    # 如果 c 减去 g 小于 0
    if (c - g < 0) {
        # 调用 print 方法，打印信息
        print("I'LL SEE YOU.\n");
        # 将 k 的值设置为 g
        k = g;
        # 将 s 的值减去 g
        s = s - g;
        # 将 c 的值减去 k
        c = c - k;
        # 将 p 的值增加 g 和 k
        p = p + g + k;
        # 返回 false
        return false;
    }
    # 声明一个变量 js，并赋值为 "N"
    js = "N";
    # 如果 o 除以 2 的余数等于 0
    if (o % 2 == 0) {
        # 调用 get_prompt 方法，询问用户是否想要以 50 美元的价格买回手表，如果用户没有输入则使用默认值 "YES"
        js = get_prompt("WOULD YOU LIKE TO BUY BACK YOUR WATCH FOR $50", "YES");
        # 如果用户输入的值的第一个字符不是 "N"
        if (js.substr(0, 1) != "N") {
            # 将 c 的值增加 50
            c += 50;
            # 将 o 的值除以 2
            o /= 2;
        }
    }
    # 如果用户输入的值的第一个字符是 "N" 并且 o 除以 3 的余数等于 0
    if (js.substr(0, 1) == "N" && o % 3 == 0) {
        # 调用 get_prompt 方法，询问用户是否想要以 50 美元的价格买回领带夹，如果用户没有输入则使用默认值 "YES"
        js = get_prompt("WOULD YOU LIKE TO BUY BACK YOUR TIE TACK FOR $50", "YES");
        # 如果用户输入的值的第一个字符不是 "N"
        if (js.substr(0, 1) != "N") {
            # 将 c 的值增加 50
            c += 50;
            # 将 o 的值除以 3
            o /= 3;
        }
    }
    # 如果字符串 js 的第一个字符是 "N"，则执行以下操作
    if (js.substr(0, 1) == "N") {
        # 打印消息 "I'M BUSTED.  CONGRATULATIONS!"
        print("I'M BUSTED.  CONGRATULATIONS!\n");
        # 返回 true
        return true;
    }
    # 如果字符串 js 的第一个字符不是 "N"，则返回 false
    return false;
# 结束函数定义
}

# 定义询问下注函数
function ask_for_bet()
{
    # 声明变量 forced
    var forced;

    # 如果 t 不是整数
    if (t != Math.floor(t)) {
        # 如果 k 不等于 0 或者 g 不等于 0 或者 t 不等于 0.5
        if (k != 0 || g != 0 || t != 0.5) {
            # 打印消息并返回 0
            print("NO SMALL CHANGE, PLEASE.\n");
            return 0;
        }
        # 返回 1
        return 1;
    }
    # 如果 s - g - t 小于 0
    if (s - g - t < 0) {
        # 如果玩家钱不够，返回 2；否则返回 0
        if (player_low_in_money())
            return 2;
        return 0;
    }
    # 如果 t 等于 0
    if (t == 0) {
        # 设置 i 为 3
        i = 3;
    } else if (g + t < k) {
        # 打印消息并返回 0
        print("IF YOU CAN'T SEE MY BET, THEN FOLD.\n");
        return 0;
    } else {
        # 将 g 加上 t
        g += t;
        # 如果 g 不等于 k
        if (g != k) {
            # 初始化 forced 为 false
            forced = false;
            # 如果 z 不等于 1
            if (z != 1) {
                # 如果 g 小于等于 3 * z，将 forced 设置为 true
                if (g <= 3 * z)
                    forced = true;
            } else {
                # 如果 g 小于等于 5
                if (g <= 5) {
                    # 如果 z 小于 2
                    if (z < 2) {
                        # 设置 v 为 5
                        v = 5;
                        # 如果 g 小于等于 3 * z，将 forced 设置为 true
                        if (g <= 3 * z)
                            forced = true;
                    }
                } else {
                    # 如果 z 等于 1 或者 t 大于 25
                    if (z == 1 || t > 25) {
                        # 设置 i 为 4
                        i = 4;
                        # 打印消息并返回 1
                        print("I FOLD.\n");
                        return 1;
                    }
                }
            }
            # 如果 forced 为 true 或者 z 等于 2
            if (forced || z == 2) {
                # 设置 v 为 g - k + fna(0)
                v = g - k + fna(0);
                # 如果电脑钱不够，返回 2
                if (computer_low_in_money())
                    return 2;
                # 打印消息并返回 0
                print("I'LL SEE YOU, AND RAISE YOU " + v + "\n");
                k = g + v;
                return 0;
            }
            # 打印消息
            print("I'LL SEE YOU.\n");
            # 设置 k 为 g
            k = g;
        }
    }
    # 减去 g 从 s
    s -= g;
    # 减去 k 从 c
    c -= k;
    # 加上 g 和 k 到 p
    p += g + k;
    # 返回 1
    return 1;
}

# 检查胜利类型的函数
function check_for_win(type)
{
    # 如果 type 等于 0 并且 i 等于 3 或者 type 等于 1
    if (type == 0 && i == 3 || type == 1) {
        # 打印消息并将 c 加上 p
        print("\n");
        print("I WIN.\n");
        c += p;
    } else if (type == 0 && i == 4 || type == 2) {
        # 打印消息并将 s 加上 p
        print("\n");
        print("YOU WIN.\n");
        s += p;
    } else {
        # 返回 0
        return 0;
    }
    # 打印消息
    print("NOW I HAVE $" + c + " AND YOU HAVE $" + s + "\n");
    # 返回 1
    return 1;
}

# 展示手牌的函数
function show_hand()
{
    # 打印牌的组合
    print(hs + is);
    # 如果花色为"A FLUS"
    if (hs == "A FLUS") {
        # 对k进行除以100并向下取整
        k = Math.floor(k / 100);
        # 打印换行符
        print("\n");
        # 显示花色
        show_suit();
        # 打印换行符
        print("\n");
    } 
    # 如果花色不为"A FLUS"
    else {
        # 对k进行fnb函数处理
        k = fnb(k);
        # 显示数字
        show_number();
        # 如果花色为"SCHMAL"或者"STRAIG"，打印" HIGH\n"
        if (hs == "SCHMAL" || hs == "STRAIG")
            print(" HIGH\n");
        # 否则打印"'S\n"
        else
            print("'S\n");
    }
// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "POKER\n");
    // 打印创意计算的信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印欢迎词和初始赌注信息
    print("WELCOME TO THE CASINO.  WE EACH HAVE $200.\n");
    print("I WILL OPEN THE BETTING BEFORE THE DRAW; YOU OPEN AFTER.\n");
    print("TO FOLD BET 0; TO CHECK BET .5.\n");
    print("ENOUGH TALK -- LET'S GET DOWN TO BUSINESS.\n");
    print("\n");
    // 初始化玩家和庄家的初始资金
    o = 1;
    c = 200;
    s = 200;
    z = 0;
    // 结束主程序
    }
}

// 调用主程序
main();
```