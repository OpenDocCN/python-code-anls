# `d:/src/tocomm/basic-computer-games\71_Poker\javascript\poker.js`

```
// 定义一个名为print的函数，用于向页面输出内容
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
# 初始化一个空字符串
str = ""

# 遍历数组 aa 中的元素
for a in aa:
    # 将元素转换为字符串并添加到 str 中
    str += str(a) 

# 遍历数组 ba 中的元素
for b in ba:
    # 将元素转换为字符串并添加到 str 中
    str += str(b)

# 将空格添加到 str 中
str += " "

# 返回拼接后的字符串
return str

# 初始化变量
var aa = []
var ba = []
var b
var c
var d
var g
var i
var k
var m
var n
var p
var s
var u
var v
var x
var z
# 声明变量 hs
var hs;
# 声明变量 is
var is;
# 声明变量 js
var js;
# 声明变量 ks
var ks;

# 定义函数 fna，返回一个 0 到 9 之间的随机整数
function fna(x)
{
    return Math.floor(10 * Math.random());
}

# 定义函数 fnb，返回参数 x 对 100 取余的结果
function fnb(x)
{
    return x % 100;
}

# 定义函数 im_busted，打印字符串 "I'M BUSTED.  CONGRATULATIONS!\n"
function im_busted()
{
    print("I'M BUSTED.  CONGRATULATIONS!\n");
}
# 1740
# 发牌
def deal_card():
    while (1):  # 无限循环，直到满足条件跳出
        aa[z] = 100 * Math.floor(4 * Math.random()) + Math.floor(100 * Math.random())  # 生成一个随机的牌
        if (Math.floor(aa[z] / 100) > 3):  # 如果花色大于3，表示无效的花色
            continue  # 继续下一次循环
        if (aa[z] % 100 > 12):  # 如果数字大于12，表示无效的牌号
            continue  # 继续下一次循环
        if (z != 1):  # 如果不是第一张牌
            for (k = 1; k <= z - 1; k++):  # 遍历之前的牌
                if (aa[z] == aa[k]):  # 如果当前牌和之前的某张牌相同
                    break  # 跳出循环
            if (k <= z - 1):  # 如果找到了重复的牌
                continue  # 继续下一次循环
            if (z > 10):  # 如果已经发了10张牌
                n = aa[u]  # 交换第一张和当前张的牌
                aa[u] = aa[z]
                aa[z] = n
// 1850
function show_cards()
{
    // 循环遍历从 n 到 n + 4 的数字
    for (z = n; z <= n + 4; z++) {
        // 打印空格和当前数字
        print(" " + z + "--  ");
        // 调用 fnb 函数并将 aa[z] 作为参数，将结果赋值给 k
        k = fnb(aa[z]);
        // 调用 show_number 函数
        show_number();
        // 打印 " OF"
        print(" OF");
        // 将 aa[z] 除以 100 的结果取整并赋值给 k
        k = Math.floor(aa[z] / 100);
        // 调用 show_suit 函数
        show_suit();
        // 如果 z 是偶数，则换行打印
        if (z % 2 == 0)
            print("\n");
    }
    // 打印换行
    print("\n");
}
}

// 1950
function show_number()
{
    // 如果 k 的值为 9，则打印 "JACK"
    if (k == 9)
        print("JACK");
    // 如果 k 的值为 10，则打印 "QUEEN"
    if (k == 10)
        print("QUEEN");
    // 如果 k 的值为 11，则打印 "KING"
    if (k == 11)
        print("KING");
    // 如果 k 的值为 12，则打印 "ACE"
    if (k == 12)
        print("ACE");
    // 如果 k 的值小于 9，则打印 k+2 的值
    if (k < 9)
        print(" " + (k + 2));
}

// 2070
function show_suit()
{
    if (k == 0)  # 如果 k 的值为 0
        print(" CLUBS\t");  # 打印 " CLUBS\t"
    if (k == 1)  # 如果 k 的值为 1
        print(" DIAMONDS\t");  # 打印 " DIAMONDS\t"
    if (k == 2)  # 如果 k 的值为 2
        print(" HEARTS\t");  # 打印 " HEARTS\t"
    if (k == 3)  # 如果 k 的值为 3
        print(" SPADES\t");  # 打印 " SPADES\t"
}

// 2170
function evaluate_hand()  # 定义名为 evaluate_hand 的函数
{
    u = 0;  # 初始化变量 u 为 0
    for (z = n; z <= n + 4; z++) {  # 循环，z 从 n 到 n + 4
        ba[z] = fnb(aa[z]);  # 将 aa[z] 的值传递给 fnb 函数，并将结果赋给 ba[z]
        if (z != n + 4) {  # 如果 z 不等于 n + 4
            if (Math.floor(aa[z] / 100) == Math.floor(aa[z + 1] / 100))  # 如果 aa[z] 除以 100 的整数部分等于 aa[z+1] 除以 100 的整数部分
                u++;  # 变量 u 自增 1
        }
    }  # 结束 if 语句块
    if (u == 4) {  # 如果 u 的值等于 4，则执行下面的代码块
        x = 11111;  # 将变量 x 的值设为 11111
        d = aa[n];  # 将数组 aa 中索引为 n 的元素赋值给变量 d
        hs = "A FLUS";  # 将字符串 "A FLUS" 赋值给变量 hs
        is = "H IN";  # 将字符串 "H IN" 赋值给变量 is
        u = 15;  # 将变量 u 的值设为 15
        return;  # 返回当前函数
    }
    for (z = n; z <= n + 3; z++) {  # 循环，从 n 开始，直到 n + 3 结束，每次增加 1
        for (k = z + 1; k <= n + 4; k++) {  # 嵌套循环，从 z + 1 开始，直到 n + 4 结束，每次增加 1
            if (ba[z] > ba[k]) {  # 如果数组 ba 中索引为 z 的元素大于索引为 k 的元素
                x = aa[z];  # 将数组 aa 中索引为 z 的元素赋值给变量 x
                aa[z] = aa[k];  # 将数组 aa 中索引为 k 的元素赋值给数组 aa 中索引为 z 的元素
                ba[z] = ba[k];  # 将数组 ba 中索引为 k 的元素赋值给数组 ba 中索引为 z 的元素
                aa[k] = x;  # 将变量 x 的值赋值给数组 aa 中索引为 k 的元素
                ba[k] = aa[k] - 100 * Math.floor(aa[k] / 100);  # 计算并赋值给数组 ba 中索引为 k 的元素
            }
        }
    }
    x = 0;  // 初始化变量 x 为 0
    for (z = n; z <= n + 3; z++) {  // 循环遍历数组 ba 中索引从 n 到 n+3 的元素
        if (ba[z] == ba[z + 1]) {  // 如果数组 ba 中索引为 z 的元素等于索引为 z+1 的元素
            x = x + 11 * Math.pow(10, z - n);  // 计算 x 的值
            d = aa[z];  // 将数组 aa 中索引为 z 的元素赋值给变量 d
            if (u < 11) {  // 如果变量 u 小于 11
                u = 11;  // 将变量 u 赋值为 11
                hs = "A PAIR";  // 将字符串 "A PAIR" 赋值给变量 hs
                is = " OF ";  // 将字符串 " OF " 赋值给变量 is
            } else if (u == 11) {  // 如果变量 u 等于 11
                if (ba[z] == ba[z - 1]) {  // 如果数组 ba 中索引为 z 的元素等于索引为 z-1 的元素
                    hs = "THREE";  // 将字符串 "THREE" 赋值给变量 hs
                    is = " ";  // 将空格赋值给变量 is
                    u = 13;  // 将变量 u 赋值为 13
                } else {  // 如果条件不成立
                    hs = "TWO P";  // 将字符串 "TWO P" 赋值给变量 hs
                    is = "AIR, ";  // 将字符串 "AIR, " 赋值给变量 is
                    u = 12;  // 将变量 u 赋值为 12
                }
            } else if (u == 12) {  // 如果变量 u 等于 12
                u = 16;  # 初始化变量 u 为 16
                hs = "FULL H";  # 初始化变量 hs 为 "FULL H"
                is = "OUSE, ";  # 初始化变量 is 为 "OUSE, "
            } else if (ba[z] == ba[z - 1]) {  # 如果 ba[z] 等于 ba[z - 1]
                u = 17;  # 变量 u 赋值为 17
                hs = "FOUR";  # 变量 hs 赋值为 "FOUR"
                is = " ";  # 变量 is 赋值为 " "
            } else {  # 否则
                u = 16;  # 变量 u 赋值为 16
                hs = "FULL H";  # 变量 hs 赋值为 "FULL H"
                is = "OUSE. ";  # 变量 is 赋值为 "OUSE. "
            }
        }
    }
    if (x == 0) {  # 如果 x 等于 0
        if (ba[n] + 3 == ba[n + 3]) {  # 如果 ba[n] + 3 等于 ba[n + 3]
            x = 1111;  # 变量 x 赋值为 1111
            u = 10;  # 变量 u 赋值为 10
        }
        if (ba[n + 1] + 3 == ba[n + 4]) {  # 如果 ba[n + 1] + 3 等于 ba[n + 4]
            if (u == 10) {  # 如果变量u的值等于10
                u = 14;  # 将变量u的值设为14
                hs = "STRAIG";  # 将变量hs的值设为"STRAIG"
                is = "HT";  # 将变量is的值设为"HT"
                x = 11111;  # 将变量x的值设为11111
                d = aa[n + 4];  # 将变量d的值设为数组aa中索引为n+4的值
                return;  # 返回
            }
            u = 10;  # 将变量u的值设为10
            x = 11110;  # 将变量x的值设为11110
        }
    }
    if (u < 10) {  # 如果变量u的值小于10
        d = aa[n + 4];  # 将变量d的值设为数组aa中索引为n+4的值
        hs = "SCHMAL";  # 将变量hs的值设为"SCHMAL"
        is = "TZ, ";  # 将变量is的值设为"TZ, "
        u = 9;  # 将变量u的值设为9
        x = 11000;  # 将变量x的值设为11000
        i = 6;  # 将变量i的值设为6
        return;  # 返回
    }  # 结束 if 语句块
    if (u == 10) {  # 如果 u 等于 10
        if (i == 1)  # 如果 i 等于 1
            i = 6;  # 将 i 赋值为 6
        return;  # 返回
    }
    if (u > 12)  # 如果 u 大于 12
        return;  # 返回
    if (fnb(d) > 6)  # 如果 fnb(d) 大于 6
        return;  # 返回
    i = 6;  # 将 i 赋值为 6
}

function get_prompt(question, def)  # 定义名为 get_prompt 的函数，接受 question 和 def 两个参数
{
    var str;  # 声明一个名为 str 的变量

    str = window.prompt(question, def);  # 使用 window.prompt 方法获取用户输入的值，赋值给 str
    print(question + "? " + str + "\n");  # 打印问题和用户输入的值
    return str;  # 返回用户输入的值
}

# 定义名为 player_low_in_money 的函数
def player_low_in_money():
    # 打印空行
    print("\n")
    # 打印提示信息
    print("YOU CAN'T BET WITH WHAT YOU HAVEN'T GOT.\n")
    # 初始化字符串变量 str
    str = "N"
    # 如果 o 除以 2 的余数不等于 0
    if (o % 2 != 0):
        # 获取用户输入的字符串并赋值给 str
        str = get_prompt("WOULD YOU LIKE TO SELL YOUR WATCH", "YES")
        # 如果用户输入的字符串的第一个字符不是 "N"
        if (str.substr(0, 1) != "N"):
            # 如果函数 fna 的返回值小于 7
            if (fna(0) < 7):
                # 打印信息并更新变量 s 的值
                print("I'LL GIVE YOU $75 FOR IT.\n")
                s += 75
            else:
                # 打印信息并更新变量 s 的值
                print("THAT'S A PRETTY CRUMMY WATCH - I'LL GIVE YOU $25.\n")
                s += 25
            # 更新变量 o 的值
            o *= 2
    # 如果 o 能被 3 整除且字符串的第一个字符是 "N"
    if (o % 3 == 0 && str.substr(0, 1) == "N") {
        # 获取用户输入，询问是否愿意放弃那个钻石领带夹
        str = get_prompt("WILL YOU PART WITH THAT DIAMOND TIE TACK", "YES");
        # 如果用户输入的第一个字符不是 "N"
        if (str.substr(0, 1) != "N") {
            # 如果 fna(0) 小于 6
            if (fna(0) < 6) {
                # 打印消息并增加 s 的值
                print("YOU ARE NOW $100 RICHER.\n");
                s += 100;
            } else {
                # 打印消息并增加 s 的值
                print("IT'S PASTE.  $25.\n");
                s += 25;
            }
            # o 增加三倍
            o *= 3;
        }
    }
    # 如果字符串的第一个字符是 "N"
    if (str.substr(0,1) == "N") {
        # 打印消息并返回 true
        print("YOUR WAD IS SHOT.  SO LONG, SUCKER!\n");
        return true;
    }
    # 返回 false
    return false;
}
# 定义一个名为computer_low_in_money的函数
def computer_low_in_money():
    # 如果c-g-v大于等于0，返回false
    if (c - g - v >= 0):
        return False
    # 如果g等于0，将v设为c，返回false
    if (g == 0):
        v = c
        return False
    # 如果c-g小于0，打印"I'LL SEE YOU."，并进行一系列操作，最后返回false
    if (c - g < 0):
        print("I'LL SEE YOU.\n")
        k = g
        s = s - g
        c = c - k
        p = p + g + k
        return False
    # 将js设为"N"
    js = "N"
    # 如果o除以2的余数等于0
    if (o % 2 == 0):
        # 调用get_prompt函数，将返回值赋给js
        js = get_prompt("WOULD YOU LIKE TO BUY BACK YOUR WATCH FOR $50", "YES")
        # 如果js的第一个字符不是"N"
        if (js.substr(0, 1) != "N"):
            c += 50;  // 增加 c 的值 50
            o /= 2;   // 将 o 的值除以 2
        }
    }
    if (js.substr(0, 1) == "N" && o % 3 == 0) {  // 如果 js 字符串的第一个字符是 "N" 并且 o 除以 3 的余数等于 0
        js = get_prompt("WOULD YOU LIKE TO BUY BACK YOUR TIE TACK FOR $50", "YES");  // 获取用户输入的提示信息
        if (js.substr(0, 1) != "N") {  // 如果 js 字符串的第一个字符不是 "N"
            c += 50;  // 增加 c 的值 50
            o /= 3;   // 将 o 的值除以 3
        }
    }
    if (js.substr(0, 1) == "N") {  // 如果 js 字符串的第一个字符是 "N"
        print("I'M BUSTED.  CONGRATULATIONS!\n");  // 打印消息
        return true;  // 返回 true
    }
    return false;  // 返回 false
}

function ask_for_bet()
{
    var forced;  // 声明一个变量 forced

    if (t != Math.floor(t)) {  // 如果 t 不等于 t 的向下取整
        if (k != 0 || g != 0 || t != 0.5) {  // 如果 k 不等于 0 或者 g 不等于 0 或者 t 不等于 0.5
            print("NO SMALL CHANGE, PLEASE.\n");  // 打印 "NO SMALL CHANGE, PLEASE."
            return 0;  // 返回 0
        }
        return 1;  // 返回 1
    }
    if (s - g - t < 0) {  // 如果 s - g - t 小于 0
        if (player_low_in_money())  // 如果玩家的钱不够
            return 2;  // 返回 2
        return 0;  // 返回 0
    }
    if (t == 0) {  // 如果 t 等于 0
        i = 3;  // 将 i 赋值为 3
    } else if (g + t < k) {  // 否则如果 g + t 小于 k
        print("IF YOU CAN'T SEE MY BET, THEN FOLD.\n");  // 打印 "IF YOU CAN'T SEE MY BET, THEN FOLD."
        return 0;  // 返回 0
    } else {  // 否则
        g += t;  # 将变量 t 的值加到变量 g 上
        if (g != k) {  # 如果变量 g 不等于变量 k
            forced = false;  # 将变量 forced 设置为 false
            if (z != 1) {  # 如果变量 z 不等于 1
                if (g <= 3 * z)  # 如果变量 g 小于等于 3 倍的 z
                    forced = true;  # 将变量 forced 设置为 true
            } else {  # 否则
                if (g <= 5) {  # 如果变量 g 小于等于 5
                    if (z < 2) {  # 如果变量 z 小于 2
                        v = 5;  # 将变量 v 设置为 5
                        if (g <= 3 * z)  # 如果变量 g 小于等于 3 倍的 z
                            forced = true;  # 将变量 forced 设置为 true
                    }
                } else {  # 否则
                    if (z == 1 || t > 25) {  # 如果变量 z 等于 1 或者变量 t 大于 25
                        i = 4;  # 将变量 i 设置为 4
                        print("I FOLD.\n");  # 打印 "I FOLD."
                        return 1;  # 返回 1
                    }
                }
            }
            # 如果强制比赛或者z等于2，则执行以下代码
            if (forced || z == 2) {
                # 计算v的值
                v = g - k + fna(0);
                # 如果计算机钱不够，则返回2
                if (computer_low_in_money())
                    return 2;
                # 打印信息
                print("I'LL SEE YOU, AND RAISE YOU " + v + "\n");
                # 更新k的值
                k = g + v;
                # 返回0
                return 0;
            }
            # 打印信息
            print("I'LL SEE YOU.\n");
            # 更新k的值
            k = g;
        }
    }
    # 更新s的值
    s -= g;
    # 更新c的值
    c -= k;
    # 更新p的值
    p += g + k;
    # 返回1
    return 1;
}

# 检查是否获胜的函数
function check_for_win(type)
{
    # 如果 type 等于 0 并且 i 等于 3，或者 type 等于 1
    if (type == 0 && i == 3 || type == 1) {
        # 打印换行符
        print("\n");
        # 打印 "I WIN."
        print("I WIN.\n");
        # c 增加 p
        c += p;
    } 
    # 如果 type 等于 0 并且 i 等于 4，或者 type 等于 2
    else if (type == 0 && i == 4 || type == 2) {
        # 打印换行符
        print("\n");
        # 打印 "YOU WIN."
        print("YOU WIN.\n");
        # s 增加 p
        s += p;
    } 
    # 如果以上条件都不满足
    else {
        # 返回 0
        return 0;
    }
    # 打印当前 c 和 s 的值
    print("NOW I HAVE $" + c + " AND YOU HAVE $" + s + "\n");
    # 返回 1
    return 1;
}

function show_hand()
{
    # 打印 hs 和 is 的值
    print(hs + is);
    # 如果 hs 等于 "A FLUS"
    if (hs == "A FLUS") {
        k = Math.floor(k / 100);  # 将 k 除以 100 并向下取整，将结果赋值给 k
        print("\n");  # 打印换行符
        show_suit();  # 调用 show_suit 函数
        print("\n");  # 打印换行符
    } else {
        k = fnb(k);  # 调用 fnb 函数，并将返回值赋给 k
        show_number();  # 调用 show_number 函数
        if (hs == "SCHMAL" || hs == "STRAIG")  # 如果 hs 的值为 "SCHMAL" 或 "STRAIG"
            print(" HIGH\n");  # 打印 " HIGH" 并换行
        else
            print("'S\n");  # 打印 "'S" 并换行
    }
}

// Main program
async function main()
{
    print(tab(33) + "POKER\n");  # 打印 tab(33) 和 "POKER" 并换行
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印 tab(15) 和 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY" 并换行
    print("\n");  # 打印换行符
    # 打印空行
    print("\n");
    print("\n");
    # 打印欢迎信息和初始赌注
    print("WELCOME TO THE CASINO.  WE EACH HAVE $200.\n");
    print("I WILL OPEN THE BETTING BEFORE THE DRAW; YOU OPEN AFTER.\n");
    print("TO FOLD BET 0; TO CHECK BET .5.\n");
    print("ENOUGH TALK -- LET'S GET DOWN TO BUSINESS.\n");
    print("\n");
    # 初始化变量
    o = 1;
    c = 200;
    s = 200;
    z = 0;
    # 进入循环
    while (1) {
        p = 0;
        #
        print("\n");
        # 如果赌注少于等于5，调用im_busted函数并返回
        if (c <= 5) {
            im_busted();
            return;
        }
        # 打印底注信息
        print("THE ANTE IS $5, I WILL DEAL:\n");
        print("\n");  # 打印一个空行
        if (s <= 5) {  # 如果s小于等于5
            if (player_low_in_money())  # 如果玩家的钱不够
                return;  # 返回
        }
        p += 10;  # p增加10
        s -= 5;  # s减少5
        c -= 5;  # c减少5
        for (z = 1; z <= 10; z++)  # 循环10次
            deal_card();  # 发牌
        print("YOUR HAND:\n");  # 打印"YOUR HAND:"
        n = 1;  # n赋值为1
        show_cards();  # 展示卡牌
        n = 6;  # n赋值为6
        i = 2;  # i赋值为2
        evaluate_hand();  # 评估手牌
        print("\n");  # 打印一个空行
        first = true;  # first赋值为true
        if (i == 6) {  # 如果i等于6
            if (fna(0) > 7) {  # 如果fna(0)大于7
                x = 11100;  // 设置变量 x 的值为 11100
                i = 7;      // 设置变量 i 的值为 7
                z = 23;     // 设置变量 z 的值为 23
            } else if (fna(0) > 7) {  // 如果 fna(0) 的返回值大于 7
                x = 11110;  // 设置变量 x 的值为 11110
                i = 7;      // 设置变量 i 的值为 7
                z = 23;     // 设置变量 z 的值为 23
            } else if (fna(0) < 2) {  // 如果 fna(0) 的返回值小于 2
                x = 11111;  // 设置变量 x 的值为 11111
                i = 7;      // 设置变量 i 的值为 7
                z = 23;     // 设置变量 z 的值为 23
            } else {  // 如果以上条件都不满足
                z = 1;      // 设置变量 z 的值为 1
                k = 0;      // 设置变量 k 的值为 0
                print("I CHECK.\n");  // 打印 "I CHECK."
                first = false;  // 设置变量 first 的值为 false
            }
        } else {  // 如果条件不满足
            if (u < 13) {  // 如果变量 u 的值小于 13
                if (fna(0) < 2) {  // 如果 fna(0) 的返回值小于 2
# 初始化变量 i 为 7
i = 7;
# 初始化变量 z 为 23
z = 23;
# 如果条件成立，则执行下面的代码块
if (u < 16) {
    # 将变量 z 的值设为 2
    z = 2;
    # 如果调用函数 fna(0) 的结果小于 1，则将变量 z 的值设为 35
    if (fna(0) < 1)
        z = 35;
# 如果条件不成立，则执行下面的代码块
} else {
    # 将变量 z 的值设为 35
    z = 35;
}
# 如果变量 first 的值为真（即非0），则执行下面的代码块
if (first) {
    # 将变量 v 的值设为 z 加上调用函数 fna(0) 的结果
    v = z + fna(0);
    # 初始化变量 g 为 0
    g = 0;
    # 如果调用函数 computer_low_in_money() 的结果为真（即非0）
                return;  # 返回空值，结束函数
            print("I'LL OPEN WITH $" + v + "\n");  # 打印带有变量 v 的字符串
            k = v;  # 将变量 v 赋值给变量 k
        }
        g = 0;  # 初始化变量 g 为 0
        do {
            print("\nWHAT IS YOUR BET");  # 打印提示信息
            t = parseFloat(await input());  # 将输入的值转换为浮点数并赋值给变量 t
            status = ask_for_bet();  # 调用 ask_for_bet 函数并将返回值赋给变量 status
        } while (status == 0) ;  # 当 status 等于 0 时循环执行上述代码块
        if (status == 2)
            return;  # 如果 status 等于 2，则返回空值，结束函数
        status = check_for_win(0);  # 调用 check_for_win 函数并将返回值赋给变量 status
        if (status == 1) {  # 如果 status 等于 1，则执行下面的代码块
            while (1) {  # 无限循环
                print("DO YOU WISH TO CONTINUE");  # 打印提示信息
                hs = await input();  # 等待输入并将输入值赋给变量 hs
                if (hs == "YES") {  # 如果 hs 等于 "YES"，则执行下面的代码块
                    status = 1;  # 将变量 status 赋值为 1
                    break;  # 跳出循环
                }
                if (hs == "NO") {  # 如果输入的字符串为"NO"
                    status = 2;  # 将状态设置为2
                    break;  # 退出循环
                }
                print("ANSWER YES OR NO, PLEASE.\n");  # 打印提示信息
            }
        }
        if (status == 2)  # 如果状态为2
            return;  # 返回
        if (status == 1) {  # 如果状态为1
            p = 0;  # 将p设置为0
            continue;  # 继续循环
        }
        print("\n");  # 打印换行
        print("NOW WE DRAW -- HOW MANY CARDS DO YOU WANT");  # 打印提示信息
        while (1) {  # 进入无限循环
            t = parseInt(await input());  # 将输入转换为整数并赋值给t
            if (t != 0) {  # 如果t不等于0
                z = 10;  # 将z设置为10
                if (t >= 4) {  # 如果抽牌次数大于等于4
                    print("YOU CAN'T DRAW MORE THAN THREE CARDS.\n");  # 打印提示信息
                    continue;  # 继续下一次循环
                }
                print("WHAT ARE THEIR NUMBERS:\n");  # 打印提示信息，询问抽到的牌的数字
                for (q = 1; q <= t; q++) {  # 循环t次，t为抽牌次数
                    u = parseInt(await input());  # 获取用户输入的牌的数字
                    z++;  # z加1
                    deal_card();  # 处理抽到的牌
                }
                print("YOUR NEW HAND:\n");  # 打印提示信息，展示新的手牌
                n = 1;  # 将n设为1
                show_cards();  # 展示手牌
            }
            break;  # 跳出循环
        }
        z = 10 + t;  # 将z设为10加上抽牌次数t
        for (u = 6; u <= 10; u++) {  # 循环6到10
            if (Math.floor(x / Math.pow(10, u - 6)) != 10 * Math.floor(x / Math.pow(10, u - 5)))  # 如果条件成立
                break;  # 跳出循环
            z++;  # 增加变量 z 的值
            deal_card();  # 调用 deal_card() 函数
        }
        print("\n");  # 打印换行符
        print("I AM TAKING " + (z - 10 - t) + " CARD");  # 打印"I AM TAKING "后面跟着 z - 10 - t 的值和 " CARD"
        if (z != 11 + t) {  # 如果 z 不等于 11 + t
            print("S");  # 打印"S"
        }
        print("\n");  # 打印换行符
        n = 6;  # 将变量 n 的值设为 6
        v = i;  # 将变量 v 的值设为 i 的值
        i = 1;  # 将变量 i 的值设为 1
        evaluate_hand();  # 调用 evaluate_hand() 函数
        b = u;  # 将变量 b 的值设为 u 的值
        m = d;  # 将变量 m 的值设为 d 的值
        if (v == 7) {  # 如果 v 等于 7
            z = 28;  # 将变量 z 的值设为 28
        } else if (i == 6) {  # 否则如果 i 等于 6
            z = 1;  # 将变量 z 的值设为 1
        } else {  # 否则
            # 如果 u 小于 13，则 z 等于 2
            if (u < 13) {
                z = 2;
                # 如果调用 fna(0) 返回值等于 6，则 z 等于 19
                if (fna(0) == 6)
                    z = 19;
            } 
            # 如果 u 大于等于 13 且小于 16，则 z 等于 19
            else if (u < 16) {
                z = 19;
                # 如果调用 fna(0) 返回值等于 8，则 z 等于 11
                if (fna(0) == 8)
                    z = 11;
            } 
            # 如果 u 大于等于 16，则 z 等于 2
            else {
                z = 2;
            }
        }
        # 初始化 k 和 g 为 0
        k = 0;
        g = 0;
        # 循环直到 status 不等于 0
        do {
            # 打印提示信息，要求输入赌注
            print("\nWHAT IS YOUR BET");
            # 将输入转换为浮点数并赋给 t
            t = parseFloat(await input());
            # 调用 ask_for_bet() 函数，将返回值赋给 status
            status = ask_for_bet();
        } while (status == 0) ;
        # 如果 status 等于 2，则执行以下代码
        if (status == 2)
            return;  # 返回语句，结束当前函数的执行并返回结果
        if (t == 0.5):  # 如果 t 的值等于 0.5
            if (v != 7 && i == 6):  # 如果 v 不等于 7 并且 i 等于 6
                print("I'LL CHECK\n");  # 打印"I'LL CHECK"并换行
            else:  # 否则
                v = z + fna(0);  # 将 z 和 fna(0) 的结果赋值给 v
                if (computer_low_in_money()):  # 如果计算机的钱不够
                    return;  # 返回语句，结束当前函数的执行并返回结果
                print("I'LL BET $" + v + "\n");  # 打印"I'LL BET $"加上 v 的值并换行
                k = v;  # 将 v 的值赋给 k
                do:  # do-while 循环
                    print("\nWHAT IS YOUR BET");  # 打印"\nWHAT IS YOUR BET"
                    t = parseFloat(await input());  # 将输入的值转换为浮点数并赋给 t
                    status = ask_for_bet();  # 调用 ask_for_bet 函数并将结果赋给 status
                while (status == 0);  # 当 status 等于 0 时继续循环
                if (status == 2):  # 如果 status 等于 2
                    return;  # 返回语句，结束当前函数的执行并返回结果
                status = check_for_win(0);  # 调用 check_for_win 函数并将结果赋给 status
                if (status == 1):  # 如果 status 等于 1
                    while (1):  # 无限循环
# 打印提示信息，询问用户是否希望继续
print("DO YOU WISH TO CONTINUE");
# 等待用户输入
hs = await input();
# 如果用户输入为"YES"，则将状态设置为1，并跳出循环
if (hs == "YES") {
    status = 1;
    break;
}
# 如果用户输入为"NO"，则将状态设置为2，并跳出循环
if (hs == "NO") {
    status = 2;
    break;
}
# 如果用户输入既不是"YES"也不是"NO"，则打印提示信息
print("ANSWER YES OR NO, PLEASE.\n");
# 如果状态为2，则直接返回
if (status == 2)
    return;
# 如果状态为1，则将p设置为0，并继续循环
if (status == 1) {
    p = 0;
    continue;
}
        } else {
            # 检查是否获胜
            status = check_for_win(0);
            # 如果获胜
            if (status == 1) {
                # 循环直到输入为YES或NO
                while (1) {
                    print("DO YOU WISH TO CONTINUE");
                    # 等待用户输入
                    hs = await input();
                    # 如果输入为YES
                    if (hs == "YES") {
                        status = 1;
                        break;
                    }
                    # 如果输入为NO
                    if (hs == "NO") {
                        status = 2;
                        break;
                    }
                    # 如果输入不是YES或NO，提示用户重新输入
                    print("ANSWER YES OR NO, PLEASE.\n");
                }
            }
            # 如果用户选择不继续游戏
            if (status == 2)
                return;
            # 如果用户选择继续游戏
            if (status == 1) {
                p = 0;  # 将变量 p 的值设为 0
                continue;  # 跳过当前循环的剩余代码，继续下一次循环
            }
        }
        print("\n");  # 打印一个空行
        print("NOW WE COMPARE HANDS:\n");  # 打印提示信息
        js = hs;  # 将变量 js 的值设为变量 hs 的值
        ks = is;  # 将变量 ks 的值设为变量 is 的值
        print("MY HAND:\n");  # 打印提示信息
        n = 6;  # 将变量 n 的值设为 6
        show_cards();  # 调用函数 show_cards()
        n = 1;  # 将变量 n 的值设为 1
        evaluate_hand();  # 调用函数 evaluate_hand()
        print("\n");  # 打印一个空行
        print("YOU HAVE ");  # 打印提示信息
        k = d;  # 将变量 k 的值设为变量 d 的值
        show_hand();  # 调用函数 show_hand()
        hs = js;  # 将变量 hs 的值设为变量 js 的值
        is = ks;  # 将变量 is 的值设为变量 ks 的值
        k = m;  # 将变量 k 的值设为变量 m 的值
        # 打印"AND I HAVE "字符串
        print("AND I HAVE ");
        # 调用show_hand函数展示手牌
        show_hand();
        # 初始化status变量为0
        status = 0;
        # 如果庄家的点数大于玩家的点数，将status设为1
        if (b > u) {
            status = 1;
        } 
        # 如果玩家的点数大于庄家的点数，将status设为2
        else if (u > b) {
            status = 2;
        } 
        # 如果玩家和庄家点数相等
        else {
            # 如果不是同花顺
            if (hs != "A FLUS") {
                # 如果玩家的点数小于庄家的点数，将status设为2
                if (fnb(m) < fnb(d))
                    status = 2;
                # 如果玩家的点数大于庄家的点数，将status设为1
                else if (fnb(m) > fnb(d))
                    status = 1;
            } 
            # 如果是同花顺
            else {
                # 如果玩家的点数大于庄家的点数，将status设为1
                if (fnb(m) > fnb(d))
                    status = 1;
                # 如果庄家的点数大于玩家的点数，将status设为2
                else if (fnb(d) > fnb(m))
                    status = 2;
            }
            # 如果status仍然为0
            if (status == 0) {
                # 打印出手牌已经展示的信息
                print("THE HAND IS DRAWN.\n");
                # 打印出当前奖池中的金额
                print("ALL $" + p + " REMAINS IN THE POT.\n");
                # 继续下一轮循环
                continue;
            }
        }
        # 检查是否有玩家获胜
        status = check_for_win(status);
        # 如果有玩家获胜
        if (status == 1) {
            # 循环直到玩家选择是否继续游戏
            while (1) {
                # 打印出询问是否继续游戏
                print("DO YOU WISH TO CONTINUE");
                # 等待用户输入
                hs = await input();
                # 如果用户选择继续游戏
                if (hs == "YES") {
                    status = 1;
                    break;
                }
                # 如果用户选择不继续游戏
                if (hs == "NO") {
                    status = 2;
                    break;
                }
                # 如果用户输入不符合要求，提示重新输入
                print("ANSWER YES OR NO, PLEASE.\n");
            }
        }
        if (status == 2)  # 如果状态为2，返回
            return;
        if (status == 1) {  # 如果状态为1
            p = 0;  # 将p置为0
            continue;  # 继续循环
        }
    }
}

main();  # 调用main函数
```