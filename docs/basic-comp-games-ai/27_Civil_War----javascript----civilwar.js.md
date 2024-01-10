# `basic-computer-games\27_Civil_War\javascript\civilwar.js`

```
// CIVIL WAR
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

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
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素的类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入元素的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 历史数据...可以通过在适当的信息后插入数据语句并调整读取来添加更多数据（战略等）
//                      0 - C$     1-M1  2-M2  3-C1 4-C2 5-D
// 历史数据数组，包含战争名称、参战人数、伤亡人数等信息
var historical_data = [,
                       ["BULL RUN",18000,18500,1967,2708,1],
                       ["SHILOH",40000.,44894.,10699,13047,3],
                       ["SEVEN DAYS",95000.,115000.,20614,15849,3],
                       ["SECOND BULL RUN",54000.,63000.,10000,14000,2],
                       ["ANTIETAM",40000.,50000.,10000,12000,3],
                       ["FREDERICKSBURG",75000.,120000.,5377,12653,1],
                       ["MURFREESBORO",38000.,45000.,11000,12000,1],
                       ["CHANCELLORSVILLE",32000,90000.,13000,17197,2],
                       ["VICKSBURG",50000.,70000.,12000,19000,1],
                       ["GETTYSBURG",72500.,85000.,20000,23000,3],
                       ["CHICKAMAUGA",66000.,60000.,18000,16000,2],
                       ["CHATTANOOGA",37000.,60000.,36700.,5800,2],
                       ["SPOTSYLVANIA",62000.,110000.,17723,18000,2],
                       ["ATLANTA",65000.,100000.,8500,3700,1]];
// 分别定义了6个空数组
var sa = [];
var da = [];
var fa = [];
var ha = [];
var ba = [];
var oa = [];

// 主程序
async function main()
{
    // 打印游戏标题和创作信息
    print(tab(26) + "CIVIL WAR\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 原始游戏设计者信息
    // 修改者信息和年份
    // 联盟信息
    sa[1] = 25;
    sa[2] = 25;
    sa[3] = 25;
    sa[4] = 25;
    // 生成一个随机数
    d = Math.random();
    print("\n");
    print("DO YOU WANT INSTRUCTIONS");
    // 循环直到输入为"YES"或"NO"
    while (1) {
        str = await input();
        if (str == "YES" || str == "NO")
            break;
        print("YES OR NO -- \n");
    }
    # 如果输入字符串为"YES"，则执行以下代码块
    if (str == "YES") {
        # 打印多个空行
        print("\n");
        print("\n");
        print("\n");
        print("\n");
        # 打印游戏介绍信息
        print("THIS IS A CIVIL WAR SIMULATION.\n");
        print("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.\n");
        print("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR\n");
        print("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE\n");
        print("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT\n");
        print("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!\n");
        print("\n");
        print("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS ");
        print("POSSIBLE.\n");
        print("\n");
        print("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:\n");
        print("        (1) ARTILLERY ATTACK\n");
        print("        (2) FORTIFICATION AGAINST FRONTAL ATTACK\n");
        print("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS\n");
        print("        (4) FALLING BACK\n");
        print(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:\n");
        print("        (1) ARTILLERY ATTACK\n");
        print("        (2) FRONTAL ATTACK\n");
        print("        (3) FLANKING MANEUVERS\n");
        print("        (4) ENCIRCLEMENT\n");
        print("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.\n");
    }
    # 打印多个空行
    print("\n");
    print("\n");
    print("\n");
    # 打印提示信息
    print("ARE THERE TWO GENERALS PRESENT ");
    # 无限循环，直到满足条件跳出循环
    while (1) {
        # 打印提示信息
        print("(ANSWER YES OR NO)");
        # 等待用户输入
        bs = await input();
        # 如果输入为"YES"，则将 d 赋值为 2，并跳出循环
        if (bs == "YES") {
            d = 2;
            break;
        } 
        # 如果输入为"NO"，则打印提示信息，将 d 赋值为 1，并跳出循环
        else if (bs == "NO") {
            print("\n");
            print("YOU ARE THE CONFEDERACY.   GOOD LUCK!\n");
            print("\n");
            d = 1;
            break;
        }
    }
    # 打印选择战斗的提示信息
    print("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON\n");
    print("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.\n");
    print("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION\n");
    # 打印提示信息，允许重新播放
    print("ALLOWING YOU TO REPLAY IT\n");
    # 打印空行
    print("\n");
    # 打印提示信息，说明负数的食物输入会导致程序使用上一场战斗的输入
    print("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO \n");
    # 打印提示信息，询问是否需要战斗描述
    print("USE THE ENTRIES FROM THE PREVIOUS BATTLE\n");
    # 打印空行
    print("\n");
    # 打印提示信息，询问是否需要战斗描述
    print("AFTER REQUESTING A BATTLE, DO YOU WISH ");
    # 循环直到输入为YES或者NO
    while (1) {
        # 打印提示信息，要求输入YES或者NO
        print("(ANSWER YES OR NO)");
        # 等待用户输入
        xs = await input();
        # 如果输入为YES或者NO，则跳出循环
        if (xs == "YES" || xs == "NO")
            break;
    }
    # 初始化变量
    l = 0;
    w = 0;
    r1 = 0;
    q1 = 0;
    m3 = 0;
    m4 = 0;
    p1 = 0;
    p2 = 0;
    t1 = 0;
    t2 = 0;
    # 循环两次
    for (i = 1; i <= 2; i++) {
        # 初始化数组元素为0
        da[i] = 0;
        fa[i] = 0;
        ha[i] = 0;
        ba[i] = 0;
        oa[i] = 0;
    }
    # 初始化变量
    r2 = 0;
    q2 = 0;
    c6 = 0;
    f = 0;
    w0 = 0;
    y = 0;
    y2 = 0;
    u = 0;
    u2 = 0;
    }
    # 打印多个空行
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    # 打印战争结果
    print("THE CONFEDERACY HAS WON " + w + " BATTLES AND LOST " + l + "\n");
    # 判断条件，根据不同条件打印不同结果
    if (y == 5 || (y2 != 5 && w <= l)) {
        print("THE UNION HAS WON THE WAR\n");
    } else {
        print("THE CONFEDERACY HAS WON THE WAR\n");
    }
    # 打印空行
    print("\n");
    # 判断条件，根据不同条件打印不同结果
    if (r1) {
        print("FOR THE " + (w + l + w0) + " BATTLES FOUGHT (EXCLUDING RERUNS)\n");
        print(" \t \t ");
        print("CONFEDERACY\t UNION\n");
        print("HISTORICAL LOSSES\t" + Math.floor(p1 + 0.5) + "\t" + Math.floor(p2 + 0.5) + "\n");
        print("SIMULATED LOSSES\t" + Math.floor(t1 + 0.5) + "\t" + Math.floor(t2 + 0.5) + "\n");
        print("\n");
        print("    % OF ORIGINAL\t" + Math.floor(100 * (t1 / p1) + 0.5) + "\t" + Math.floor(100 * (t2 / p2) + 0.5) + "\n");
        # 判断条件，根据不同条件打印不同结果
        if (bs != "YES") {
            print("\n");
            print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED \n");
            print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES\n");
            print(sa[1] + " " + sa[2] + " " + sa[3] + " " + sa[4] + "\n");
        }
    }
# 调用名为main的函数
main();
```