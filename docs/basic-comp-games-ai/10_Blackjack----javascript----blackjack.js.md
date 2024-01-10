# `basic-computer-games\10_Blackjack\javascript\blackjack.js`

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
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串并返回
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

// 初始化一些变量
var da = [];

var pa = [];
var qa = [];
var ca = [];
var ta = [];
var sa = [];
var ba = [];
var za = [];
var ra = [];

var ds = "N A  2  3  4  5  6  7N 8  9 10  J  Q  K";
var is = "H,S,D,/,"

var q;
var aa;
var ab;
var ac;
var h;
var h1;

// 定义一个函数，根据条件返回一个值
function af(q) {
    return q >= 22 ? q - 11 : q;
}

// 定义一个重新洗牌的函数
function reshuffle()
{
    // 打印信息
    print("RESHUFFLING\n");
    // 循环将牌重新放入牌堆
    for (; d >= 1; d--)
        ca[--c] = da[d];
}
    # 从52开始递减到c，每次循环都执行以下操作
    for (c1 = 52; c1 >= c; c1--) {
        # 生成一个介于c和c1之间的随机整数
        c2 = Math.floor(Math.random() * (c1 - c + 1)) + c;
        # 交换数组ca中索引为c2和c1的元素
        c3 = ca[c2];
        ca[c2] = ca[c1];
        ca[c1] = c3;
    }
// 结束当前的 JavaScript 函数
}

// 获取一张卡牌的子程序
function get_card()
{
    // 如果卡牌数量大于等于51，重新洗牌
    if (c >= 51)
        reshuffle();
    // 返回第 c 张卡牌，并将 c 加一
    return ca[c++];
}

// 打印卡牌的子程序
function card_print(x)
{
    // 打印从字符串 ds 中提取的卡牌信息
    print(ds.substr(3 * x - 3, 3) + "  ");
}

// 替代的卡牌打印子程序
function alt_card_print(x)
{
    // 打印从字符串 ds 中提取的替代卡牌信息
    print(" " + ds.substr(3 * x - 2, 2) + "   ");
}

// 将卡牌 'which' 添加到总数 'q' 的子程序
function add_card(which)
{
    x1 = which;
    // 如果卡牌大于10，将其视为10
    if (x1 > 10)
        x1 = 10;
    q1 = q + x1;
    // 如果总数小于11
    if (q < 11) {
        // 如果卡牌小于等于1，总数加11
        if (which <= 1) {
            q += 11;
            return;
        }
        // 如果 q1 大于等于11，总数为 q1+11，否则为 q1
        if (q1 >= 11)
            q = q1 + 11;
        else
            q = q1;
        return;
    }
    // 如果总数小于等于21且 q1 大于21，总数为 q1+1，否则为 q1
    if (q <= 21 && q1 > 21)
        q = q1 + 1;
    else
        q = q1;
    // 如果总数大于等于33，总数为-1
    if (q >= 33)
        q = -1;
}

// 评估手牌 'which' 的子程序。总数放入 qa[which] 中
function evaluate_hand(which)
{
    q = 0;
    for (q2 = 1; q2 <= ra[which]; q2++) {
        add_card(pa[i][q2]);
    }
    qa[which] = q;
}

// 将一张卡牌添加到第 i 行的子程序
function add_card_to_row(i, x) {
    ra[i]++;
    pa[i][ra[i]] = x;
    q = qa[i];
    add_card(x);
    qa[i] = q;
    // 如果总数小于0，打印“...BUSTED”，丢弃该行
    if (q < 0) {
        print("...BUSTED\n");
        discard_row(i);
    }
}

// 丢弃第 i 行的子程序
function discard_row(i) {
    while (ra[i]) {
        d++;
        da[d] = pa[i][ra[i]];
        ra[i]--;
    }
}

// 打印第 i 行的总数
function print_total(i) {
    print("\n");
    aa = qa[i];
    total_aa();
    print("TOTAL IS " + aa + "\n");
}

// 计算 aa 的总数
function total_aa()
{
    // 如果 aa 大于等于22，减去11
    if (aa >= 22)
        aa -= 11;
}

// 计算 ab 的总数
function total_ab()
{
    // 如果 ab 大于等于22，减去11
    if (ab >= 22)
        ab -= 11;
}

// 计算 ac 的总数
function total_ac()
{
    // 如果 ac 大于等于22，减去11
    if (ac >= 22)
        ac -= 11;
}

// 处理输入的子程序
function process_input(str)
{
    // 截取字符串的第一个字符
    str = str.substr(0, 1);
    // 循环遍历 is 字符串
    for (h = 1; h <= h1; h += 2) {
        if (str == is.substr(h - 1, 1))
            break;
    }
}
    # 如果h小于等于h1，则执行以下代码块
    if (h <= h1) {
        # 将h加1后除以2，然后将结果赋给h
        h = (h + 1) / 2;
        # 返回0
        return 0;
    }
    # 打印字符串"TYPE "和is字符串从索引0到h1-1的子串，以及字符串" OR "和is字符串从索引h1-1到h1+1的子串，以及字符串" PLEASE"
    print("TYPE " + is.substr(0, h1 - 1) + " OR " + is.substr(h1 - 1, 2) + " PLEASE");
    # 返回1
    return 1;
// 主程序
async function main()
{
    print(tab(31) + "BLACK JACK\n");  // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印信息
    print("\n");
    print("\n");
    print("\n");
    // --pa[i][j] IS THE JTH CARD IN HAND I, qa[i] IS TOTAL OF HAND I
    // --C IS THE DECK BEING DEALT FROM, D IS THE DISCARD PILE,
    // --ta[i] IS THE TOTAL FOR PLAYER I, sa[i] IS THE TOTAL THIS HAND FOR
    // --PLAYER I, ba[i] IS TH BET FOR HAND I
    // --ra[i] IS THE LENGTH OF pa[I,*]

    // --程序从这里开始
    // --初始化
    for (i = 1; i <= 15; i++)
        pa[i] = [];  // 初始化牌组
    for (i = 1; i <= 13; i++)
        for (j = 4 * i - 3; j <= 4 * i; j++)
            da[j] = i;  // 初始化牌堆
    d = 52;  // 初始化牌堆数量
    c = 53;  // 初始化牌堆数量
    print("DO YOU WANT INSTRUCTIONS");  // 打印提示信息
    str = await input();  // 获取用户输入
    if (str.toUpperCase().substr(0, 1) != "N") {  // 判断用户输入
        print("THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE\n");
        // 打印游戏规则
        print("GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE\n");
        // 打印游戏规则
        print("PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE\n");
        // 打印游戏规则
        print("DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE\n");
        // 打印游戏规则
        print("FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE\n");
        // 打印游戏规则
        print("PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS\n");
        // 打印游戏规则
        print("STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',\n");
        // 打印游戏规则
        print("INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE\n");
        // 打印游戏规则
        print("INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR\n");
        // 打印游戏规则
        print("'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING\n");
        // 打印游戏规则
        print("DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR\n");
        // 打印游戏规则
        print("BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.\n");
        // 打印游戏规则
    }
    while (1) {
        print("NUMBER OF PLAYERS");  // 打印提示信息
        n = parseInt(await input());  // 获取用户输入并转换为整数
        print("\n");
        if (n < 1 || n > 7)
            continue;  // 如果玩家数量不在1到7之间，则继续循环
        else
            break;  // 否则跳出循环
    }
    for (i = 1; i <= 8; i++)
        ta[i] = 0;  // 初始化玩家总分
    d1 = n + 1;  // 初始化变量
    }
}
# 调用名为main的函数
main();
```