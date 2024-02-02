# `basic-computer-games\83_Stock_Market\javascript\stockmarket.js`

```py
// STOCKMARKET
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串添加到输出元素中
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

// 定义缩进函数，返回指定数量空格组成的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义变量
var sa = [];
var pa = [];
var za = [];
var ca = [];
var i1;
var n1;
var e1;
var i2;
var n2;
var e2;
var x1;
var w3;
var t8;
var a;
var s4;

// 新股票值 - 子程序
function randomize_initial()
{
    // 根据前一天的值随机产生新的股票值
    // N1,N2 是随机天数，分别决定股票 I1 增加 10 点和股票 I2 减少 10 点
    // 如果 N1 天已经过去，选择一个 I1，设置 E1，确定新的 N1
    if (n1 <= 0) {
        i1 = Math.floor(4.99 * Math.random() + 1);  // 生成一个 1 到 5 之间的随机整数
        n1 = Math.floor(4.99 * Math.random() + 1);  // 生成一个 1 到 5 之间的随机整数
        e1 = 1;  // 设置 E1 为 1
    }
    // 如果 N2 天已经过去，选择一个 I2，设置 E2，确定新的 N2
    if (n2 <= 0) {
        i2 = Math.floor(4.99 * Math.random() + 1);  // 生成一个 1 到 5 之间的随机整数
        n2 = Math.floor(4.99 * Math.random() + 1);  // 生成一个 1 到 5 之间的随机整数
        e2 = 1;  // 设置 E2 为 1
    }
    // 从 N1 和 N2 中减去一天
    n1--;
    n2--;
    // 循环遍历所有股票
    for (i = 1; i <= 5; i++) {
        x1 = Math.random();  // 生成一个 0 到 1 之间的随机小数
        if (x1 < 0.25) {
            x1 = 0.25;  // 如果 x1 小于 0.25，将其设置为 0.25
        } else if (x1 < 0.5) {
            x1 = 0.5;  // 如果 x1 小于 0.5，将其设置为 0.5
        } else if (x1 < 0.75) {
            x1 = 0.75;  // 如果 x1 小于 0.75，将其设置为 0.75
        } else {
            x1 = 0.0;  // 否则将其设置为 0.0
        }
        // 大变化常数：W3（初始设置为零）
        w3 = 0;
        if (e1 >= 1 && Math.floor(i1 + 0.5) == Math.floor(i + 0.5)) {
            w3 = 10;  // 如果 E1 大于等于 1 并且 I1 四舍五入等于 i，将 w3 设置为 10
            e1 = 0;  // 重置 E1
        }
        if (e2 >= 1 && Math.floor(i2 + 0.5) == Math.floor(i + 0.5)) {
            w3 -= 10;  // 如果 E2 大于等于 1 并且 I2 四舍五入等于 i，将 w3 减去 10
            e2 = 0;  // 重置 E2
        }
        // C(I) 是股票价值的变化
        ca[i] = Math.floor(a * sa[i]) + x1 + Math.floor(3 - 6 * Math.random() + 0.5) + w3;  // 计算股票价值的变化
        ca[i] = Math.floor(100 * ca[i] + 0.5) / 100;  // 对 ca[i] 进行四舍五入保留两位小数
        sa[i] += ca[i];  // 更新股票价值
        if (sa[i] <= 0) {
            ca[i] = 0;  // 如果股票价值小于等于 0，将变化设置为 0
            sa[i] = 0;  // 将股票价值设置为 0
        } else {
            sa[i] = Math.floor(100 * sa[i] + 0.5) / 100;  // 对股票价值进行四舍五入保留两位小数
        }
    }
    // 在 T8 天之后随机改变趋势符号和斜率
    if (--t8 < 1) {
        // 随机改变趋势符号和斜率（A），以及趋势的持续时间（T8）
        t8 = Math.floor(4.99 * Math.random() + 1);  // 生成一个 1 到 5 之间的随机整数
        a = Math.floor((Math.random() / 10) * 100 + 0.5) / 100;  // 生成一个 0 到 0.1 之间的随机小数，并保留两位小数
        s4 = Math.random();  // 生成一个 0 到 1 之间的随机小数
        if (s4 > 0.5)
            a = -a;  // 如果 s4 大于 0.5，将 a 取反
    }
// 主程序
async function main()
{
    // 打印标题
    print(tab(30) + "STOCK MARKET\n");
    // 打印创意计算公司信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印股票市场模拟信息
    // 修订日期 8/18/70 (D. PESSEL, L. BRAUN, C. LOSIK)
    // 重要变量：A-MRKT TRND SLP; B5-BRKRGE FEE; C-TTL CSH ASSTS;
    // C5-TTL CSH ASSTS (TEMP); C(I)-CHNG IN STK VAL; D-TTL ASSTS;
    // E1,E2-LRG CHNG MISC; I-STCK #; I1,I2-STCKS W LRG CHNG;
    // N1,N2-LRG CHNG DAY CNTS; P5-TTL DAYS PRCHSS; P(I)-PRTFL CNTNTS;
    // Q9-NEW CYCL?; S4-SGN OF A; S5-TTL DYS SLS; S(I)-VALUE/SHR;
    // T-TTL STCK ASSTS; T5-TTL VAL OF TRNSCTNS;
    // W3-LRG CHNG; X1-SMLL CHNG(<$1); Z4,Z5,Z6-NYSE AVE.; Z(I)-TRNSCT
    // SLOPE OF MARKET TREND:A  (SAME FOR ALL STOCKS)
    x = 1;
    // 生成随机数并赋值给a
    a = Math.floor(Math.random() / 10 * 100 + 0.5) / 100;
    t5 = 0;
    x9 = 0;
    n1 = 0;
    n2 = 0;
    e1 = 0;
    e2 = 0;
    // 介绍部分
    print("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0)");
    // 获取用户输入并转换为整数
    z9 = parseInt(await input());
    print("\n");
    print("\n");
    # 如果 z9 大于等于 1，则执行以下操作
    if (z9 >= 1) {
        # 打印程序说明
        print("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN\n");
        print("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL\n");
        print("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT\n");
        print("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE\n");
        print("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES\n");
        print("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE\n");
        print("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION\n");
        print("MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK\n");
        print("TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE\n");
        print("NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED\n");
        print("ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS\n");
        print("TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU\n");
        print("HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.\n");
        print("(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST\n");
        print("10 DAYS)\n");
        print("-----GOOD LUCK!-----\n");
    }
    // GENERATION OF STOCK TABLE: INPUT REQUESTS
    // INITIAL STOCK VALUES
    # 初始化股票价格
    sa[1] = 100;
    sa[2] = 85;
    sa[3] = 150;
    sa[4] = 140;
    sa[5] = 110;
    // INITIAL T8 - # DAYS FOR FIRST TREND SLOPE (A)
    # 初始化 t8 为第一个趋势斜率的天数
    t8 = Math.floor(4.99 * Math.random() + 1);
    // RANDOMIZE SIGN OF FIRST TREND SLOPE (A)
    # 随机确定第一个趋势斜率的符号
    if (Math.random() <= 0.5)
        a -= a;
    # 随机初始化值
    randomize_initial();
    # 初始化投资组合内容
    for (i = 1; i <= 5; i++) {
        pa[i] = 0;
        za[i] = 0;
    }
    print("\n");
    print("\n");
    # 初始化现金资产
    c = 10000;
    z5 = 0;
    # 打印初始投资组合
    print("STOCK\t \t\t\tINITIALS\tPRICE/SHARE\n");
    print("INT. BALLISTIC MISSILES\t\t  IBM\t\t" + sa[1] + "\n");
    print("RED CROSS OF AMERICA\t\t  RCA\t\t" + sa[2] + "\n");
    # 打印"LICHTENSTEIN, BUMRAP & JOKE"的缩写、"LBJ"的缩写和sa列表中索引为3的元素的值
    print("LICHTENSTEIN, BUMRAP & JOKE\t  LBJ\t\t" + sa[3] + "\n");
    # 打印"AMERICAN BANKRUPT CO."的缩写、"ABC"的缩写和sa列表中索引为4的元素的值
    print("AMERICAN BANKRUPT CO.\t\t  ABC\t\t" + sa[4] + "\n");
    # 打印"CENSURED BOOKS STORE"的缩写、"CBS"的缩写和sa列表中索引为5的元素的值
    print("CENSURED BOOKS STORE\t\t  CBS\t\t" + sa[5] + "\n");
    # 结束函数
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```