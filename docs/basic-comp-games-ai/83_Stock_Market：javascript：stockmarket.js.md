# `83_Stock_Market\javascript\stockmarket.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入类型
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
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格字符追加到字符串变量str的末尾
    return str;  # 返回处理后的字符串变量str

var sa = [];  # 声明一个空数组sa
var pa = [];  # 声明一个空数组pa
var za = [];  # 声明一个空数组za
var ca = [];  # 声明一个空数组ca
var i1;  # 声明一个变量i1
var n1;  # 声明一个变量n1
var e1;  # 声明一个变量e1
var i2;  # 声明一个变量i2
var n2;  # 声明一个变量n2
var e2;  # 声明一个变量e2
var x1;  # 声明一个变量x1
var w3;  # 声明一个变量w3
var t8;  # 声明一个变量t8
var a;  # 声明一个变量a
var s4;  # 声明一个变量s4
// 新股票值 - 子程序
function randomize_initial()
{
    // 根据前一天的值随机产生新的股票值
    // N1，N2 是分别确定股票 I1 将增加 10 点和股票 I2 将减少 10 点的随机天数
    // 如果 N1 天已经过去，选择一个 I1，设置 E1，确定新的 N1
    if (n1 <= 0) {
        i1 = Math.floor(4.99 * Math.random() + 1);  // 产生一个1到5之间的随机整数
        n1 = Math.floor(4.99 * Math.random() + 1);  // 产生一个1到5之间的随机整数
        e1 = 1;
    }
    // 如果 N2 天已经过去，选择一个 I2，设置 E2，确定新的 N2
    if (n2 <= 0) {
        i2 = Math.floor(4.99 * Math.random() + 1);  // 产生一个1到5之间的随机整数
        n2 = Math.floor(4.99 * Math.random() + 1);  // 产生一个1到5之间的随机整数
        e2 = 1;
    }
    // 从 n1 和 n2 中减去一天
    n1--;
    n2--;
    // 遍历所有股票
    for (i = 1; i <= 5; i++) {
        x1 = Math.random();
        if (x1 < 0.25) {
            x1 = 0.25;
        } else if (x1 < 0.5) {
            x1 = 0.5;
        } else if (x1 < 0.75) {
            x1 = 0.75;
        } else {
            x1 = 0.0;
        }
        // 大变化常数：W3（初始设置为零）
        w3 = 0;
        if (e1 >= 1 && Math.floor(i1 + 0.5) == Math.floor(i + 0.5)) {
            // 给这支股票加上10个点；重置 e1
            w3 = 10;
            e1 = 0;  # 将变量 e1 设为 0
        }
        if (e2 >= 1 && Math.floor(i2 + 0.5) == Math.floor(i + 0.5)) {
            // 如果 e2 大于等于 1 并且 Math.floor(i2 + 0.5) 等于 Math.floor(i + 0.5)，则执行以下操作
            // 从这支股票中减去 10 点；重置 e2
            w3 -= 10;
            e2 = 0;
        }
        // C(I) 是股票价值的变化
        ca[i] = Math.floor(a * sa[i]) + x1 + Math.floor(3 - 6 * Math.random() + 0.5) + w3;  # 计算股票价值的变化
        ca[i] = Math.floor(100 * ca[i] + 0.5) / 100;  # 对 ca[i] 进行四舍五入保留两位小数
        sa[i] += ca[i];  # 更新股票价值
        if (sa[i] <= 0) {
            ca[i] = 0;
            sa[i] = 0;
        } else {
            sa[i] = Math.floor(100 * sa[i] + 0.5) / 100;  # 对 sa[i] 进行四舍五入保留两位小数
        }
    }
    // 在 T8 天之后随机改变趋势符号和斜率
    if (--t8 < 1) {
        // 随机改变趋势的符号和斜率（A），以及趋势的持续时间（T8）
        t8 = Math.floor(4.99 * Math.random() + 1); // 生成一个1到5之间的随机整数，表示趋势的持续时间
        a = Math.floor((Math.random() / 10) * 100 + 0.5) / 100; // 生成一个0到0.1之间的随机数，并保留两位小数，表示斜率
        s4 = Math.random(); // 生成一个0到1之间的随机数
        if (s4 > 0.5) // 如果随机数大于0.5
            a = -a; // 将斜率取反
    }
}

// 主程序
async function main()
{
    print(tab(30) + "STOCK MARKET\n"); // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n"); // 打印信息
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    // STOCK MARKET SIMULATION     -STOCK-
    // REVISED 8/18/70 (D. PESSEL, L. BRAUN, C. LOSIK)
    // 定义变量 x，初始化为 1
    x = 1;
    // 生成一个随机数，并将其乘以 10，然后取整，再除以 100，最后加上 0.5，得到一个小数点后两位的随机数
    a = Math.floor(Math.random() / 10 * 100 + 0.5) / 100;
    // 初始化变量 t5 为 0
    t5 = 0;
    // 初始化变量 x9 为 0
    x9 = 0;
    // 初始化变量 n1 为 0
    n1 = 0;
    // 初始化变量 n2 为 0
    n2 = 0;
    // 初始化变量 e1 为 0
    e1 = 0;
    // 初始化变量 e2 为 0
    e2 = 0;
    // 打印提示信息，询问用户是否需要说明（是-输入1，否-输入0）
    print("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0)");
    // 将用户输入的值转换为整数并赋给变量 z9
    z9 = parseInt(await input());
    // 打印换行符
    print("\n");
    # 打印空行
    print("\n");
    # 如果 z9 大于等于 1，则打印以下内容
    if (z9 >= 1) {
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
    # 生成股票表格：输入请求
    // GENERATION OF STOCK TABLE: INPUT REQUESTS
    // INITIAL STOCK VALUES
    // 初始化股票值
    sa[1] = 100;
    sa[2] = 85;
    sa[3] = 150;
    sa[4] = 140;
    sa[5] = 110;
    
    // INITIAL T8 - # DAYS FOR FIRST TREND SLOPE (A)
    // 初始化T8 - 第一个趋势斜率（A）的天数
    t8 = Math.floor(4.99 * Math.random() + 1);
    
    // RANDOMIZE SIGN OF FIRST TREND SLOPE (A)
    // 随机化第一个趋势斜率（A）的符号
    if (Math.random() <= 0.5)
        a -= a;
    
    // RANDOMIZE INITIAL VALUES
    // 随机化初始值
    randomize_initial();
    
    // INITIAL PORTFOLIO CONTENTS
    // 初始化投资组合内容
    for (i = 1; i <= 5; i++) {
        pa[i] = 0;
        za[i] = 0;
    }
    
    // 打印空行
    print("\n");
    // 打印空行
    print("\n");
    # 初始化现金资产为10000
    c = 10000
    # 初始化z5为0
    z5 = 0
    # 打印初始投资组合
    print("STOCK\t \t\t\tINITIALS\tPRICE/SHARE\n")
    print("INT. BALLISTIC MISSILES\t\t  IBM\t\t" + sa[1] + "\n")
    print("RED CROSS OF AMERICA\t\t  RCA\t\t" + sa[2] + "\n")
    print("LICHTENSTEIN, BUMRAP & JOKE\t  LBJ\t\t" + sa[3] + "\n")
    print("AMERICAN BANKRUPT CO.\t\t  ABC\t\t" + sa[4] + "\n")
    print("CENSURED BOOKS STORE\t\t  CBS\t\t" + sa[5] + "\n")
    while (1):
        print("\n")
        # 设置z4为z5的值
        z4 = z5
        # 重置z5为0
        z5 = 0
        # 重置t为0
        t = 0
        for (i = 1; i <= 5; i++):
            # 计算z5的值
            z5 += sa[i]
            # 计算t的值
            t += sa[i] * pa[i]
        z5 = Math.floor(100 * (z5 / 5) + 0.5) / 100;  // 对 z5 进行数学运算并取整
        z6 = Math.floor((z5 - z4) * 100 + 0.5) / 100;  // 对 z6 进行数学运算并取整
        // TOTAL ASSETS:D
        d = t + c;  // 计算总资产
        if (x9 <= 0) {  // 如果 x9 小于等于 0
            print("NEW YORK STOCK EXCHANGE AVERAGE: " + z5 + "\n");  // 打印纽约证券交易所平均值
        } else {
            print("NEW YORK STOCK EXCHANGE AVERAGE: " + z5 + " NET CHANGE " + z6 + "\n");  // 打印纽约证券交易所平均值和净变化
        }
        print("\n");  // 打印空行
        t = Math.floor(100 * t + 0.5) / 100;  // 对 t 进行数学运算并取整
        print("TOTAL STOCK ASSETS ARE   $" + t + "\n");  // 打印总股票资产
        c = Math.floor(100 * c + 0.5) / 100;  // 对 c 进行数学运算并取整
        print("TOTAL CASH ASSETS ARE    $" + c + "\n");  // 打印总现金资产
        d = Math.floor(100 * d + 0.5) / 100;  // 对 d 进行数学运算并取整
        print("TOTAL ASSETS ARE         $" + d + "\n");  // 打印总资产
        print("\n");  // 打印空行
        if (x9 != 0) {  // 如果 x9 不等于 0
            print("DO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)");  // 打印提示信息
            q9 = parseInt(await input());  // 获取用户输入并转换为整数
            if (q9 < 1) {  // 如果 q9 小于 1
                print("HOPE YOU HAD FUN!!\n");  // 打印消息“希望你玩得开心！！”
                return;  // 返回
            }
        }
        // INPUT TRANSACTIONS  // 输入交易
        while (1) {  // 循环
            print("WHAT IS YOUR TRANSACTION IN\n");  // 打印消息“你的交易是什么”
            print("IBM");  // 打印消息“IBM”
            za[1] = parseInt(await input());  // 将输入的值转换为整数并存储在数组 za 的第一个位置
            print("RCA");  // 打印消息“RCA”
            za[2] = parseInt(await input());  // 将输入的值转换为整数并存储在数组 za 的第二个位置
            print("LBJ");  // 打印消息“LBJ”
            za[3] = parseInt(await input());  // 将输入的值转换为整数并存储在数组 za 的第三个位置
            print("ABC");  // 打印消息“ABC”
            za[4] = parseInt(await input());  // 将输入的值转换为整数并存储在数组 za 的第四个位置
            print("CBS");  // 打印消息“CBS”
            za[5] = parseInt(await input());  // 将输入的值转换为整数并存储在数组 za 的第五个位置
            print("\n");  // 打印换行符
            // TOTAL DAY'S PURCHASES IN $:P5  // 当天的总购买金额为 $:P5
            p5 = 0;  // 初始化变量 p5 为 0
            // TOTAL DAY'S SALES IN $:S5
            s5 = 0;  // 初始化变量 s5 为 0
            for (i = 1; i <= 5; i++) {  // 循环 5 次，i 从 1 到 5
                za[i] = Math.floor(za[i] + 0.5);  // 对 za[i] 进行四舍五入取整
                if (za[i] > 0) {  // 如果 za[i] 大于 0
                    p5 += za[i] * sa[i];  // p5 加上 za[i] 乘以 sa[i] 的值
                } else {  // 如果 za[i] 不大于 0
                    s5 -= za[i] * sa[i];  // s5 减去 za[i] 乘以 sa[i] 的值
                    if (-za[i] > pa[i]) {  // 如果 -za[i] 大于 pa[i]
                        print("YOU HAVE OVERSOLD A STOCK; TRY AGAIN.\n");  // 打印错误信息
                        break;  // 跳出循环
                    }
                }
            }
            if (i <= 5)  // 如果 i 小于等于 5
                contine;  // 继续下一次循环
            // TOTAL VALUE OF TRANSACTIONS:T5
            t5 = p5 + s5;  // 计算 t5 的值为 p5 加上 s5
            // BROKERAGE FEE:B5
            // 计算并四舍五入得到b5的值
            b5 = Math.floor(0.01 * t5 * 100 + 0.5) / 100;
            // 计算新的现金资产
            c5 = c - p5 - b5 + s5;
            // 如果现金资产为负数，打印提示信息并继续循环
            if (c5 < 0) {
                print("YOU HAVE USED $" + (-c5) + " MORE THAN YOU HAVE.\n");
                continue;
            }
            // 如果现金资产为正数，跳出循环
            break;
        }
        // 更新现金资产
        c = c5;
        // 计算新的投资组合
        for (i = 1; i <= 5; i++) {
            pa[i] += za[i];
        }
        // 计算新的股票价值
        randomize_initial();
        // 打印投资组合
        // 在许多计算机上，铃声声音可能不同
        print("\n");
# 打印交易结束的提示信息
print("**********     END OF DAY'S TRADING     **********\n");
# 打印空行
print("\n");
# 打印空行
print("\n");
# 如果 x9 大于等于 1，则执行下面的代码
if (x9 >= 1) ;
# 打印股票信息表头
print("STOCK\tPRICE/SHARE\tHOLDINGS\tVALUE\tNET PRICE CHANGE\n");
# 打印 IBM 股票信息
print("IBM\t" + sa[1] + "\t\t" + pa[1] + "\t\t" + sa[1] * pa[1] + "\t" + ca[1] + "\n");
# 打印 RCA 股票信息
print("RCA\t" + sa[2] + "\t\t" + pa[2] + "\t\t" + sa[2] * pa[2] + "\t" + ca[2] + "\n");
# 打印 LBJ 股票信息
print("LBJ\t" + sa[3] + "\t\t" + pa[3] + "\t\t" + sa[3] * pa[3] + "\t" + ca[3] + "\n");
# 打印 ABC 股票信息
print("ABC\t" + sa[4] + "\t\t" + pa[4] + "\t\t" + sa[4] * pa[4] + "\t" + ca[4] + "\n");
# 打印 CBS 股票信息
print("CBS\t" + sa[5] + "\t\t" + pa[5] + "\t\t" + sa[5] * pa[5] + "\t" + ca[5] + "\n");
# 将 x9 设置为 1
x9 = 1;
# 打印空行
print("\n");
# 打印空行
print("\n");
# 调用 main 函数
}

main();
```