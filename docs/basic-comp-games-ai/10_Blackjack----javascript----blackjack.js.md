# `10_Blackjack\javascript\blackjack.js`

```
// 创建一个名为BLACKJACK的函数
//
// 由Oscar Toledo G. (nanochess)将BASIC转换为Javascript
//
// 创建一个名为print的函数，用于在页面上输出字符串
//
// 创建一个名为input的函数，用于获取用户输入
//
// 声明变量input_element和input_str
//
// 返回一个Promise对象，用于处理异步操作
//
// 创建一个input元素
//
// 在页面上输出问号，提示用户输入
//
// 设置input元素的类型为文本
//
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键按下事件监听器
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键（keyCode 为 13）
    if (event.keyCode == 13) {
        # 将输入字符串设置为输入元素的值
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
# 结束输入元素的添加
});
}

# 定义一个函数 tab，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串 str
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var da = [];  # 创建一个空数组
var pa = [];  # 创建一个空数组
var qa = [];  # 创建一个空数组
var ca = [];  # 创建一个空数组
var ta = [];  # 创建一个空数组
var sa = [];  # 创建一个空数组
var ba = [];  # 创建一个空数组
var za = [];  # 创建一个空数组
var ra = [];  # 创建一个空数组

var ds = "N A  2  3  4  5  6  7N 8  9 10  J  Q  K";  # 创建一个包含扑克牌面值的字符串
var is = "H,S,D,/,"  # 创建一个包含扑克牌花色的字符串

var q;  # 声明一个变量 q
var aa;  # 声明一个变量 aa
# 声明变量 ab
var ab;
# 声明变量 ac
var ac;
# 声明变量 h
var h;
# 声明变量 h1
var h1;

# 定义函数 af，接受参数 q，如果 q 大于等于 22，则返回 q 减去 11，否则返回 q
function af(q) {
    return q >= 22 ? q - 11 : q;
}

# 定义函数 reshuffle
function reshuffle()
{
    # 打印字符串 "RESHUFFLING"
    print("RESHUFFLING\n");
    # 循环，当 d 大于等于 1 时，执行循环体，每次循环结束后将 d 减一
    for (; d >= 1; d--)
        # 将数组 da 中索引为 d 的元素赋值给数组 ca 中索引为 c 的位置，然后将 c 减一
        ca[--c] = da[d];
    # 从 c1 的初始值 52 开始循环，每次循环结束后将 c1 减一
    for (c1 = 52; c1 >= c; c1--) {
        # 生成一个介于 c 和 c1 之间的随机整数，赋值给变量 c2
        c2 = Math.floor(Math.random() * (c1 - c + 1)) + c;
        # 将数组 ca 中索引为 c2 的元素赋值给变量 c3
        c3 = ca[c2];
        # 将数组 ca 中索引为 c2 的元素赋值为数组 ca 中索引为 c1 的元素
        ca[c2] = ca[c1];
        # 将数组 ca 中索引为 c1 的元素赋值为变量 c3
        ca[c1] = c3;
    }
```
}

// Subroutine to get a card.
function get_card()
{
    // 如果卡片数量超过51张，则重新洗牌
    if (c >= 51)
        reshuffle();
    // 返回当前卡片并递增计数器
    return ca[c++];
}

// Card printing subroutine
function card_print(x)
{
    // 打印指定位置的卡片信息
    print(ds.substr(3 * x - 3, 3) + "  ");
}

// Alternate card printing subroutine
function alt_card_print(x)
{
    // 打印指定位置的卡片信息（另一种格式）
    print(" " + ds.substr(3 * x - 2, 2) + "   ");
}

// Subroutine to add card 'which' to total 'q'
// 添加卡片'which'到总数'q'的子程序
function add_card(which)
{
    x1 = which;
    // 将卡片值赋给变量x1
    if (x1 > 10)
        x1 = 10;
    // 如果卡片值大于10，则将其赋值为10
    q1 = q + x1;
    // 将总数q和卡片值相加，赋值给变量q1
    if (q < 11) {
        // 如果总数q小于11
        if (which <= 1) {
            // 如果卡片值小于等于1
            q += 11;
            // 将总数q加上11
            return;
            // 返回
        }
        if (q1 >= 11)
            q = q1 + 11;
        // 如果q1大于等于11，则将q赋值为q1加上11
        else
            q = q1;
        // 否则将q赋值为q1
        return;
        // 返回
    }
    # 如果 q 小于等于 21 并且 q1 大于 21，则将 q 赋值为 q1 + 1
    if (q <= 21 && q1 > 21)
        q = q1 + 1;
    # 否则将 q 赋值为 q1
    else
        q = q1;
    # 如果 q 大于等于 33，则将 q 赋值为 -1
    if (q >= 33)
        q = -1;
}

// 评估手牌 'which' 的子程序。总数放入 qa[which]。总数的含义如下：
//  2-10...硬 2-10
// 11-21...软 11-21
// 22-32...硬 11-21
//  33+....爆牌
function evaluate_hand(which)
{
    q = 0;
    # 对于 q2 从 1 到 ra[which] 的循环
    for (q2 = 1; q2 <= ra[which]; q2++) {
        # 添加卡片 pa[i][q2] 的值
        add_card(pa[i][q2]);
    }
    qa[which] = q;  # 将变量 q 赋值给数组 qa 的第 which 个元素

// Subroutine to add a card to row i
function add_card_to_row(i, x) {
    ra[i]++;  # 将数组 ra 的第 i 个元素加一
    pa[i][ra[i]] = x;  # 将变量 x 赋值给二维数组 pa 的第 i 行的第 ra[i] 列
    q = qa[i];  # 将数组 qa 的第 i 个元素赋值给变量 q
    add_card(x);  # 调用 add_card 函数，传入参数 x
    qa[i] = q;  # 将变量 q 赋值给数组 qa 的第 i 个元素
    if (q < 0) {  # 如果 q 小于 0
        print("...BUSTED\n");  # 打印 "...BUSTED"
        discard_row(i);  # 调用 discard_row 函数，传入参数 i
    }
}

// Subroutine to discard row i
function discard_row(i) {
    while (ra[i]) {  # 当数组 ra 的第 i 个元素不为 0 时
        d++;  # 变量 d 加一
        da[d] = pa[i][ra[i]];  // 将数组 pa[i] 中索引为 ra[i] 的值赋给数组 da 的索引为 d 的位置
        ra[i]--;  // 将数组 ra 中索引为 i 的值减一
    }
}

// 打印手牌 i 的总数
function print_total(i) {
    print("\n");  // 打印换行符
    aa = qa[i];  // 将数组 qa 中索引为 i 的值赋给变量 aa
    total_aa();  // 调用函数 total_aa
    print("TOTAL IS " + aa + "\n");  // 打印 "TOTAL IS " 和变量 aa 的值，并换行
}

function total_aa()
{
    if (aa >= 22)  // 如果变量 aa 的值大于等于 22
        aa -= 11;  // 将变量 aa 的值减去 11
}

function total_ab()
{
    # 如果 ab 大于等于 22，则将 ab 减去 11
    if (ab >= 22)
        ab -= 11;
}

# 计算总 ac
function total_ac()
{
    # 如果 ac 大于等于 22，则将 ac 减去 11
    if (ac >= 22)
        ac -= 11;
}

# 处理输入字符串
function process_input(str)
{
    # 截取字符串的第一个字符
    str = str.substr(0, 1);
    # 循环遍历 h1 次，每次增加 2
    for (h = 1; h <= h1; h += 2) {
        # 如果字符串的第一个字符等于 is 字符串中的第 h-1 个字符，则跳出循环
        if (str == is.substr(h - 1, 1))
            break;
    }
    # 如果 h 小于等于 h1，则执行以下操作
    if (h <= h1) {
        # 将 h 值更新为 (h + 1) 除以 2
        h = (h + 1) / 2;
        return 0; // 返回值为0
    }
    print("TYPE " + is.substr(0, h1 - 1) + " OR " + is.substr(h1 - 1, 2) + " PLEASE"); // 打印字符串
    return 1; // 返回值为1
}

// Main program
async function main()
{
    print(tab(31) + "BLACK JACK\n"); // 打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n"); // 打印字符串
    print("\n"); // 打印换行
    print("\n"); // 打印换行
    print("\n"); // 打印换行
    // --pa[i][j] IS THE JTH CARD IN HAND I, qa[i] IS TOTAL OF HAND I
    // --C IS THE DECK BEING DEALT FROM, D IS THE DISCARD PILE,
    // --ta[i] IS THE TOTAL FOR PLAYER I, sa[i] IS THE TOTAL THIS HAND FOR
    // --PLAYER I, ba[i] IS TH BET FOR HAND I
    // --ra[i] IS THE LENGTH OF pa[I,*]
}
    // --Program starts here
    // 初始化
    for (i = 1; i <= 15; i++)
        pa[i] = [];
    for (i = 1; i <= 13; i++)
        for (j = 4 * i - 3; j <= 4 * i; j++)
            da[j] = i;
    d = 52;
    c = 53;
    print("DO YOU WANT INSTRUCTIONS");
    // 等待用户输入
    str = await input();
    // 如果用户输入的字符串不是以"N"开头，则打印游戏规则说明
    if (str.toUpperCase().substr(0, 1) != "N") {
        print("THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE\n");
        print("GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE\n");
        print("PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE\n");
        print("DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE\n");
        print("FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE\n");
        print("PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS\n");
        print("STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',\n");
        print("INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE\n");
```
```python
        // ... 省略部分代码
        # 打印初始响应，之后的响应应该是 'S' 或 'H'，除非牌已经分开，此时再次允许加倍下注。为了收集二十一点，初始响应应该是 'S'。
        print("INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR\n");
        print("'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING\n");
        print("DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR\n");
        print("BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.\n");
    }
    while (1) {
        # 打印玩家数量
        print("NUMBER OF PLAYERS");
        # 从输入中获取玩家数量
        n = parseInt(await input());
        print("\n");
        # 如果玩家数量小于1或大于7，则继续循环
        if (n < 1 || n > 7)
            continue;
        # 否则跳出循环
        else
            break;
    }
    # 初始化数组 ta
    for (i = 1; i <= 8; i++)
        ta[i] = 0;
    # 计算 d1 的值
    d1 = n + 1;
    while (1) {
        # 如果满足条件，则重新洗牌
        if (2 * d1 + c >= 52) {
            reshuffle();
        }  # 结束 if 语句块
        if (c == 2)  # 如果 c 的值等于 2
            c--;  # c 减一
        for (i = 1; i <= n; i++)  # 循环 i 从 1 到 n
            za[i] = 0;  # 将 za[i] 的值设为 0
        for (i = 1; i <= 15; i++)  # 循环 i 从 1 到 15
            ba[i] = 0;  # 将 ba[i] 的值设为 0
        for (i = 1; i <= 15; i++)  # 循环 i 从 1 到 15
            qa[i] = 0;  # 将 qa[i] 的值设为 0
        for (i = 1; i <= 7; i++)  # 循环 i 从 1 到 7
            sa[i] = 0;  # 将 sa[i] 的值设为 0
        for (i = 1; i <= 15; i++)  # 循环 i 从 1 到 15
            ra[i] = 0;  # 将 ra[i] 的值设为 0
        print("BETS:\n");  # 打印字符串 "BETS:\n"
        for (i = 1; i <= n; i++) {  # 循环 i 从 1 到 n
            do {  # 执行以下操作
                print("#" + i + " ");  # 打印字符串 "#"、i 的值、空格
                za[i] = parseFloat(await input());  # 将输入的值转换为浮点数并赋给 za[i]
            } while (za[i] <= 0 || za[i] > 500) ;  # 当 za[i] 小于等于 0 或者大于 500 时继续循环
        }
        for (i = 1; i <= n; i++)
            ba[i] = za[i];  # 将数组 za 的值复制到数组 ba 中

        print("PLAYER");  # 打印字符串 "PLAYER"

        for (i = 1; i <= n; i++) {
            print(" " + i + "    ");  # 打印空格、i 的值和多个空格
        }

        print("DEALER\n");  # 打印字符串 "DEALER" 并换行

        for (j = 1; j <= 2; j++) {
            print(tab(5));  # 打印 5 个空格
            for (i = 1; i <= d1; i++) {
                pa[i][j] = get_card();  # 为数组 pa 的元素赋值为从 get_card() 函数获取的卡片
                if (j == 1 || i <= n)
                    alt_card_print(pa[i][j]);  # 如果条件成立，打印 pa[i][j] 的备用卡片
            }
            print("\n");  # 换行
        }

        for (i = 1; i <= d1; i++)
            ra[i] = 2;  # 将数组 ra 的元素赋值为 2

        // --Test for insurance
        if (pa[d1][1] <= 1) {  # 如果条件成立，执行以下代码
            # 打印"ANY INSURANCE"
            print("ANY INSURANCE");
            # 等待用户输入字符串
            str = await input();
            # 如果输入字符串的第一个字符是"Y"
            if (str.substr(0, 1) == "Y") {
                # 打印"INSURANCE BETS"
                print("INSURANCE BETS\n");
                # 循环n次
                for (i = 1; i <= n; i++) {
                    # 循环直到用户输入的值大于等于0且小于等于ba[i]的一半
                    do {
                        # 打印"#" + i + " "
                        print("#" + i + " ");
                        # 将用户输入的值转换为浮点数并赋值给za[i]
                        za[i] = parseFloat(await input());
                    } while (za[i] < 0 || za[i] > ba[i] / 2) ;
                }
                # 根据条件计算sa[i]的值
                for (i = 1; i <= n; i++)
                    sa[i] = za[i] * ((pa[d1][2] >= 10 ? 3 : 0) - 1);
            }
        }
        # --Test for dealer blackjack
        # 初始化l1和l2的值为1
        l1 = 1;
        l2 = 1;
        # 如果pa[d1][1]等于1且pa[d1][2]大于9
        if (pa[d1][1] == 1 && pa[d1][2] > 9) {
            # 将l1和l2的值设为0
            l1 = 0;
            l2 = 0;
        }
        // 如果庄家的第一个牌是A并且第二个牌大于9，则将l1和l2都设为0
        if (pa[d1][2] == 1 && pa[d1][1] > 9) {
            l1 = 0;
            l2 = 0;
        }
        // 如果l1和l2都为0，则输出庄家的第二张牌并进行玩家手牌的评估
        if (l1 == 0 && l2 == 0) {
            print("\n");
            print("DEALER HAS A" + ds.substr(3 * pa[d1][2] - 3, 3) + " IN THE HOLE FOR BLACKJACK\n");
            for (i = 1; i <= d1; i++)
                evaluate_hand(i);
        } else {
            // --庄家没有黑杰克
            if (pa[d1][1] <= 1 || pa[d1][1] >= 10) {
                print("\n");
                print("NO DEALER BLACKJACK.\n");
            }
            // --现在进行玩家手牌的游戏
            for (i = 1; i <= n; i++) {
                print("PLAYER " + i + " ");
                h1 = 7;
                // 从输入中获取字符串，直到 process_input 返回 false
                do {
                    str = await input();
                } while (process_input(str)) ;
                // 如果 h 等于 1，玩家想要被击中
                if (h == 1) {
                    // 评估玩家手中的牌
                    evaluate_hand(i);
                    // 将 h1 设置为 3
                    h1 = 3;
                    // 获取一张牌
                    x = get_card();
                    // 打印接收到的牌
                    print("RECEIVED A");
                    // 打印牌的信息
                    card_print(x);
                    // 将牌添加到玩家的手中
                    add_card_to_row(i, x);
                    // 如果 q 大于 0，则打印玩家的总点数
                    if (q > 0)
                        print_total(i);
                } 
                // 如果 h 等于 2，玩家想要站立
                else if (h == 2) {
                    // 评估玩家手中的牌
                    evaluate_hand(i);
                    // 如果玩家手中的牌总点数为 21
                    if (qa[i] == 21) {
                        // 打印“BLACKJACK”
                        print("BLACKJACK\n");
                        // 累加玩家的赢得的筹码
                        sa[i] = sa[i] + 1.5 * ba[i];
                        // 将玩家的赌注设为 0
                        ba[i] = 0;
                        // 丢弃玩家手中的牌
                        discard_row(i);
                    } else {
                    // 打印玩家的总点数
                    print_total(i);
                    // 如果玩家选择 double down
                } else if (h == 3) {    // 玩家想要加倍下注
                    // 评估玩家手中的牌
                    evaluate_hand(i);
                    // 将 h1 设为 3，表示玩家已经 double down
                    h1 = 3;
                    // 将 h 设为 1，表示玩家需要进行下一步操作
                    h = 1;
                    // 进入循环，直到玩家决定停止或者爆牌
                    while (1) {
                        // 如果玩家选择要牌
                        if (h == 1) {   // 要牌
                            // 从牌堆中获取一张牌
                            x = get_card();
                            // 打印收到的牌
                            print("RECEIVED A");
                            // 打印收到的牌的具体信息
                            card_print(x);
                            // 将收到的牌添加到玩家手中
                            add_card_to_row(i, x);
                            // 如果玩家爆牌，则结束循环
                            if (q < 0)
                                break;
                            // 打印玩家选择要牌
                            print("HIT");
                        // 如果玩家选择停牌
                        } else if (h == 2) {    // 停牌
                            // 打印玩家的总点数
                            print_total(i);
                            // 结束循环
                            break;
                        }
                        // 执行一次
                            str = await input();  // 从输入流中读取字符串
                        } while (process_input(str)) ;  // 当处理输入的函数返回真时，继续循环
                        h1 = 3;  // 将变量 h1 的值设为 3
                    }
                } else if (h == 4) {    // 如果变量 h 的值为 4，表示玩家想要分牌
                    l1 = pa[i][1];  // 将变量 l1 的值设为玩家手中第一张牌的点数
                    if (l1 > 10)  // 如果 l1 大于 10
                        l1 = 10;  // 将 l1 的值设为 10
                    l2 = pa[i][2];  // 将变量 l2 的值设为玩家手中第二张牌的点数
                    if (l2 > 10)  // 如果 l2 大于 10
                        l2 = 10;  // 将 l2 的值设为 10
                    if (l1 != l2) {  // 如果 l1 不等于 l2
                        print("SPLITTING NOT ALLOWED.\n");  // 打印提示信息
                        i--;  // 将变量 i 减一
                        continue;  // 继续下一次循环
                    }
                    // --Play out split
                    i1 = i + d1;  // 将变量 i1 的值设为 i 加上 d1
                    ra[i1] = 2;  // 将数组 ra 中索引为 i1 的元素的值设为 2
                    pa[i1][1] = pa[i1][2];  // 将玩家手中第一张牌的点数设为玩家手中第二张牌的点数
                    ba[i + d1] = ba[i];  // 将 ba[i] 的值赋给 ba[i + d1]
                    x = get_card();  // 调用 get_card() 函数，将返回值赋给 x
                    print("FIRST HAND RECEIVES A");  // 打印输出字符串 "FIRST HAND RECEIVES A"
                    card_print(x);  // 调用 card_print() 函数，打印输出 x 对应的卡片信息
                    pa[i][2] = x;  // 将 x 的值赋给 pa[i][2]
                    evaluate_hand(i);  // 调用 evaluate_hand() 函数，对第 i 个手牌进行评估
                    print("\n");  // 打印输出换行符

                    x = get_card();  // 调用 get_card() 函数，将返回值赋给 x
                    print("SECOND HAND RECEIVES A");  // 打印输出字符串 "SECOND HAND RECEIVES A"
                    i = i1;  // 将 i1 的值赋给 i
                    card_print(x);  // 调用 card_print() 函数，打印输出 x 对应的卡片信息
                    pa[i][2] = x;  // 将 x 的值赋给 pa[i][2]
                    evaluate_hand(i);  // 调用 evaluate_hand() 函数，对第 i 个手牌进行评估
                    print("\n");  // 打印输出换行符

                    i = i1 - d1;  // 将 i1 - d1 的值赋给 i
                    if (pa[i][1] != 1) {  // 如果 pa[i][1] 的值不等于 1，则执行以下代码块
                        // --Now play the two hands
                        do {
                            print("HAND " + (i > d1 ? 2 : 1) + " ");  // 打印输出字符串 "HAND " 和 (i > d1 ? 2 : 1) 的值
h1 = 5;  // 初始化变量 h1 为 5
while (1) {  // 进入无限循环
    do {
        str = await input();  // 等待输入并将输入值赋给变量 str
    } while (process_input(str)) ;  // 如果 process_input 函数返回真，则继续循环
    h1 = 3;  // 将变量 h1 的值设为 3
    if (h == 1) {   // 如果 h 的值为 1，执行以下操作
        x = get_card();  // 调用 get_card 函数并将返回值赋给变量 x
        print("RECEIVED A");  // 打印字符串 "RECEIVED A"
        card_print(x);  // 调用 card_print 函数并传入参数 x
        add_card_to_row(i, x);  // 调用 add_card_to_row 函数并传入参数 i 和 x
        if (q < 0)  // 如果 q 小于 0
            break;  // 跳出循环
        print("HIT");  // 打印字符串 "HIT"
    } else if (h == 2) {    // 如果 h 的值为 2，执行以下操作
        print_total(i);  // 调用 print_total 函数并传入参数 i
        break;  // 跳出循环
    } else {    // 如果 h 的值既不是 1 也不是 2，执行以下操作
        x = get_card();  // 调用 get_card 函数并将返回值赋给变量 x
        ba[i] *= 2;  // 将数组 ba 中索引为 i 的元素乘以 2
# 打印"RECEIVED A"
print("RECEIVED A")
# 调用card_print函数，传入参数x
card_print(x)
# 调用add_card_to_row函数，传入参数i和x
add_card_to_row(i, x)
# 如果q大于0，则调用print_total函数，传入参数i
if (q > 0):
    print_total(i)
# 跳出循环
break
# 结束if语句

# 结束while循环
# 重新赋值i为i1-d1
i = i1 - d1

# --测试玩家是否要继续玩庄家的手牌
# 调用evaluate_hand函数，传入参数i
evaluate_hand(i)
# 遍历i从1到n
for (i = 1; i <= n; i++):
    # 如果ra[i]大于0或者ra[i + d1]大于0，则跳出循环
    if (ra[i] > 0 or ra[i + d1] > 0):
        break
# 结束for循环
            # 如果 i 大于 n，则打印“DEALER HAD A”，并获取 pa[d1][2] 对应的卡牌数据，打印出来
            if (i > n) {
                print("DEALER HAD A");
                x = pa[d1][2];
                card_print(x);
                print(" CONCEALED.\n");
            } else {
                # 如果 i 不大于 n，则打印“DEALER HAS A”加上 ds 中从第 3 * pa[d1][2] - 3 开始的 3 个字符，再加上“ CONCEALED ”
                print("DEALER HAS A" + ds.substr(3 * pa[d1][2] - 3, 3) + " CONCEALED ");
                i = d1;
                # 获取 qa[i] 对应的值，计算总和并打印
                aa = qa[i];
                total_aa();
                print("FOR A TOTAL OF " + aa + "\n");
                # 如果总和小于等于 16，则打印“DRAWS”，并执行以下循环
                if (aa <= 16) {
                    print("DRAWS");
                    do {
                        # 获取一张卡牌，打印出来
                        x = get_card();
                        alt_card_print(x);
                        # 将卡牌添加到行中
                        add_card_to_row(i, x);
                        aa = q;
                        # 重新计算总和并打印
                        total_aa();
                    } while (q > 0 && aa < 17) ;  # 当 q 大于 0 且 aa 小于 17 时执行循环
                    if (q < 0) {  # 如果 q 小于 0
                        qa[i] = q + 0.5;  # 将 qa[i] 赋值为 q + 0.5
                    } else {
                        qa[i] = q;  # 否则将 qa[i] 赋值为 q
                    }
                    if (q >= 0) {  # 如果 q 大于等于 0
                        aa = q;  # 将 aa 赋值为 q
                        total_aa();  # 调用 total_aa() 函数
                        print("---TOTAL IS " + aa + "\n");  # 打印总数
                    }
                }
                print("\n");  # 打印换行
            }
        }
        // --TALLY THE RESULT  # 计算结果
        str = "LOSES PUSHES WINS "  # 定义字符串
        print("\n");  # 打印换行
        for (i = 1; i <= n; i++) {  # 循环 n 次
            aa = qa[i]  # 将 aa 赋值为 qa[i]
# 调用total_aa函数
total_aa();
# 将qa[i + d1]的值赋给ab
ab = qa[i + d1];
# 调用total_ab函数
total_ab();
# 将qa[d1]的值赋给ac
ac = qa[d1];
# 调用total_ac函数
total_ac();
# 计算aa和ac的差值
signaaac = aa - ac;
# 如果signaaac不为0
if (signaaac) {
    # 如果signaaac小于0，则将其赋值为-1，否则赋值为1
    if (signaaac < 0)
        signaaac = -1;
    else
        signaaac = 1;
}
# 计算ab和ac的差值
signabac = ab - ac;
# 如果signabac不为0
if (signabac) {
    # 如果signabac小于0，则将其赋值为-1，否则赋值为1
    if (signabac < 0)
        signabac = -1;
    else
        signabac = 1;
}
# 计算sa[i]的新值
sa[i] = sa[i] + ba[i] * signaaac + ba[i + d1] * signabac;
            ba[i + d1] = 0;  // 将数组 ba 中索引为 i+d1 的元素赋值为 0
            print("PLAYER " + i + " ");  // 打印输出字符串 "PLAYER " + i + " "
            signsai = sa[i];  // 将变量 signsai 赋值为数组 sa 中索引为 i 的元素
            if (signsai) {  // 如果 signsai 不为 0
                if (signsai < 0)  // 如果 signsai 小于 0
                    signsai = -1;  // 将 signsai 赋值为 -1
                else
                    signsai = 1;  // 否则将 signsai 赋值为 1
            }
            print(str.substr(signsai * 6 + 6, 6) + " ");  // 打印输出从字符串 str 中截取的子字符串
            if (sa[i] == 0)  // 如果数组 sa 中索引为 i 的元素为 0
                print("      ");  // 打印输出空格
            else
                print(" " + Math.abs(sa[i]) + " ");  // 否则打印输出 sa[i] 的绝对值
            ta[i] = ta[i] + sa[i];  // 将数组 ta 中索引为 i 的元素加上 sa[i] 的值
            print("TOTAL= " + ta[i] + "\n");  // 打印输出字符串 "TOTAL= " + ta[i] + 换行符
            discard_row(i);  // 调用函数 discard_row 并传入参数 i
            ta[d1] = ta[d1] - sa[i];  // 将数组 ta 中索引为 d1 的元素减去 sa[i] 的值
            i += d1;  // 将变量 i 的值增加 d1
            discard_row(i);  // 调用函数 discard_row 并传入参数 i
            i -= d1;  # 减去玩家的第一个牌的点数
        }
        print("DEALER'S TOTAL= " + ta[d1] + "\n");  # 打印庄家的总点数
        print("\n");  # 打印空行
        discard_row(i);  # 丢弃玩家的牌
    }
}

main();  # 调用主函数
```