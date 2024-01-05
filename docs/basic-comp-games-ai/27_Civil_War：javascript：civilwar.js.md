# `d:/src/tocomm/basic-computer-games\27_Civil_War\javascript\civilwar.js`

```
# 定义函数print，用于在页面上输出字符串
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str));

# 定义函数input，用于获取用户输入
def input():
    var input_element;
    var input_str;

    # 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
                       # 创建一个input元素
                       input_element = document.createElement("INPUT");

                       # 在页面上输出提示符
                       print("? ");

                       # 设置input元素的类型为文本
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
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串

// Historical data...can add more (strat., etc) by inserting
// data statements after appro. info, and adjusting read
//                      0 - C$     1-M1  2-M2  3-C1 4-C2 5-D
var historical_data = [,
                       ["BULL RUN",18000,18500,1967,2708,1],  // 历史数据数组的第一个元素
                       ["SHILOH",40000.,44894.,10699,13047,3],  // 历史数据数组的第二个元素
                       ["SEVEN DAYS",95000.,115000.,20614,15849,3],  // 历史数据数组的第三个元素
                       ["SECOND BULL RUN",54000.,63000.,10000,14000,2],  // 历史数据数组的第四个元素
                       ["ANTIETAM",40000.,50000.,10000,12000,3],  // 历史数据数组的第五个元素
                       ["FREDERICKSBURG",75000.,120000.,5377,12653,1],  // 历史数据数组的第六个元素
                       ["MURFREESBORO",38000.,45000.,11000,12000,1],  // 历史数据数组的第七个元素
                       ["CHANCELLORSVILLE",32000,90000.,13000,17197,2],  // 历史数据数组的第八个元素
                       ["VICKSBURG",50000.,70000.,12000,19000,1],  // 历史数据数组的第九个元素
                       ["GETTYSBURG",72500.,85000.,20000,23000,3],  // 历史数据数组的第十个元素
                       ["CHICKAMAUGA",66000.,60000.,18000,16000,2],  // 历史数据数组的第十一个元素
                       ["CHATTANOOGA",37000.,60000.,36700.,5800,2],  // 历史数据数组的第十二个元素
var sa = [];  # 创建一个空数组sa
var da = [];  # 创建一个空数组da
var fa = [];  # 创建一个空数组fa
var ha = [];  # 创建一个空数组ha
var ba = [];  # 创建一个空数组ba
var oa = [];  # 创建一个空数组oa

// Main program  # 主程序
async function main()  # 异步函数main
{
    print(tab(26) + "CIVIL WAR\n");  # 打印26个空格和"CIVIL WAR"字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印15个空格和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"字符串
    print("\n");  # 打印一个换行符
    print("\n");  # 打印一个换行符
    print("\n");  # 打印一个换行符
    // Original game design: Cram, Goodie, Hibbard Lexington H.S.  # 原始游戏设计者信息
    // Modifications: G. Paul, R. Hess (Ties), 1973  # 修改者信息
    // Union info on likely confederate strategy  # 联盟对可能的联盟策略的信息
```
```python
    sa[1] = 25;  # 将数组 sa 的第一个元素赋值为 25
    sa[2] = 25;  # 将数组 sa 的第二个元素赋值为 25
    sa[3] = 25;  # 将数组 sa 的第三个元素赋值为 25
    sa[4] = 25;  # 将数组 sa 的第四个元素赋值为 25
    d = Math.random();  # 生成一个 0 到 1 之间的随机数并赋值给变量 d
    print("\n");  # 打印一个空行
    print("DO YOU WANT INSTRUCTIONS");  # 打印提示信息 "DO YOU WANT INSTRUCTIONS"
    while (1) {  # 进入一个无限循环
        str = await input();  # 等待用户输入并将输入的值赋给变量 str
        if (str == "YES" || str == "NO")  # 如果用户输入的值为 "YES" 或 "NO"
            break;  # 退出循环
        print("YES OR NO -- \n");  # 如果用户输入的值不是 "YES" 或 "NO"，则打印提示信息
    }
    if (str == "YES") {  # 如果用户输入的值为 "YES"
        print("\n");  # 打印一个空行
        print("\n");  # 打印一个空行
        print("\n");  # 打印一个空行
        print("\n");  # 打印一个空行
        print("THIS IS A CIVIL WAR SIMULATION.\n");  # 打印提示信息 "THIS IS A CIVIL WAR SIMULATION."
        print("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.\n");  # 打印提示信息 "TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS."
# 打印游戏提示信息
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
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    # 打印提示信息
    print("ARE THERE TWO GENERALS PRESENT ");
    # 进入循环，等待用户输入
    while (1) {
        # 打印提示信息，要求用户回答YES或NO
        print("(ANSWER YES OR NO)");
        # 等待用户输入
        bs = await input();
        # 如果用户输入YES，则设定d为2并跳出循环
        if (bs == "YES") {
            d = 2;
            break;
        } 
        # 如果用户输入NO，则设定d为1并跳出循环
        else if (bs == "NO") {
            print("\n");
            print("YOU ARE THE CONFEDERACY.   GOOD LUCK!\n");
            print("\n");
            d = 1;
            break;
        }
    }
    # 打印提示信息，要求用户选择战斗编号
    print("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON\n");
    # 打印提示信息，要求用户输入其他数字结束模拟
    print("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.\n");
    # 打印提示信息，提醒用户按下 '0' 键可以回到上一次的战斗情况
    print("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION\n");
    # 打印提示信息，提醒用户可以重播上一次的战斗
    print("ALLOWING YOU TO REPLAY IT\n");
    # 打印空行
    print("\n");
    # 打印提示信息，提醒用户负数的食物数会导致程序使用上一次战斗的数据
    print("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO \n");
    print("USE THE ENTRIES FROM THE PREVIOUS BATTLE\n");
    # 打印提示信息，询问用户是否希望在请求战斗后查看战斗描述
    print("AFTER REQUESTING A BATTLE, DO YOU WISH ");
    print("BATTLE DESCRIPTIONS ");
    # 进入循环，直到用户输入 YES 或 NO 为止
    while (1) {
        print("(ANSWER YES OR NO)");
        xs = await input();
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
    p1 = 0;  // 初始化变量 p1 为 0
    p2 = 0;  // 初始化变量 p2 为 0
    t1 = 0;  // 初始化变量 t1 为 0
    t2 = 0;  // 初始化变量 t2 为 0
    for (i = 1; i <= 2; i++) {  // 循环两次
        da[i] = 0;  // 初始化数组 da 的第 i 个元素为 0
        fa[i] = 0;  // 初始化数组 fa 的第 i 个元素为 0
        ha[i] = 0;  // 初始化数组 ha 的第 i 个元素为 0
        ba[i] = 0;  // 初始化数组 ba 的第 i 个元素为 0
        oa[i] = 0;  // 初始化数组 oa 的第 i 个元素为 0
    }
    r2 = 0;  // 初始化变量 r2 为 0
    q2 = 0;  // 初始化变量 q2 为 0
    c6 = 0;  // 初始化变量 c6 为 0
    f = 0;   // 初始化变量 f 为 0
    w0 = 0;  // 初始化变量 w0 为 0
    y = 0;   // 初始化变量 y 为 0
    y2 = 0;  // 初始化变量 y2 为 0
    u = 0;   // 初始化变量 u 为 0
    u2 = 0;  // 初始化变量 u2 为 0
    while (1):  # 进入无限循环
        print("\n")  # 打印换行
        print("\n")  # 打印换行
        print("\n")  # 打印换行
        print("WHICH BATTLE DO YOU WISH TO SIMULATE")  # 打印提示信息
        a = parseInt(await input())  # 从输入中获取整数并赋值给变量a
        if (a < 1 || a > 14):  # 如果a小于1或者大于14
            break  # 退出循环
        if (a != 0 || r == 0):  # 如果a不等于0或者r等于0
            cs = historical_data[a][0]  # 从historical_data中获取对应索引的值赋给cs
            m1 = historical_data[a][1]  # 从historical_data中获取对应索引的值赋给m1
            m2 = historical_data[a][2]  # 从historical_data中获取对应索引的值赋给m2
            c1 = historical_data[a][3]  # 从historical_data中获取对应索引的值赋给c1
            c2 = historical_data[a][4]  # 从historical_data中获取对应索引的值赋给c2
            m = historical_data[a][5]  # 从historical_data中获取对应索引的值赋给m
            u = 0  # 将变量u的值设为0
            # Inflation calc
            i1 = 10 + (l - w) * 2  # 计算i1的值
            i2 = 10 + (w - l) * 2  # 计算i2的值
            # Money available
            // 计算并设置 da[1] 的值
            da[1] = 100 * Math.floor((m1 * (100 - i1) / 2000) * (1 + (r1 - q1) / (r1 + 1)) + 0.5);
            // 计算并设置 da[2] 的值
            da[2] = 100 * Math.floor(m2 * (100 - i2) / 2000 + 0.5);
            // 如果 bs 等于 "YES"，重新计算并设置 da[2] 的值
            if (bs == "YES") {
                da[2] = 100 * Math.floor((m2 * (100 - i2) / 2000) * (1 + (r2 - q2) / (r2 + 1)) + 0.5);
            }
            // 计算并设置 m5 的值
            m5 = Math.floor(m1 * (1 + (p1 - t1) / (m3 + 1)));
            // 计算并设置 m6 的值
            m6 = Math.floor(m2 * (1 + (p2 - t2) / (m4 + 1)));
            // 计算并设置 f1 的值
            f1 = 5 * m1 / 6;
            // 打印空行
            print("\n");
            print("\n");
            print("\n");
            print("\n");
            print("\n");
            // 打印战斗信息
            print("THIS IS THE BATTLE OF " + cs + "\n");
            // 如果 xs 不等于 "NO"，执行以下代码
            if (xs != "NO") {
                switch (a) {
                    case 1:
                        // 打印特定的战斗信息
                        print("JULY 21, 1861.  GEN. BEAUREGARD, COMMANDING THE SOUTH, MET\n");
                        print("UNION FORCES WITH GEN. MCDOWELL IN A PREMATURE BATTLE AT\n");
# 打印关于战争历史的信息
print("BULL RUN. GEN. JACKSON HELPED PUSH BACK THE UNION ATTACK.\n");
# 跳出 switch 语句
break;
# 打印关于战争历史的信息
print("APRIL 6-7, 1862.  THE CONFEDERATE SURPRISE ATTACK AT\n");
print("SHILOH FAILED DUE TO POOR ORGANIZATION.\n");
break;
# 打印关于战争历史的信息
print("JUNE 25-JULY 1, 1862.  GENERAL LEE (CSA) UPHELD THE\n");
print("OFFENSIVE THROUGHOUT THE BATTLE AND FORCED GEN. MCCLELLAN\n");
print("AND THE UNION FORCES AWAY FROM RICHMOND.\n");
break;
# 打印关于战争历史的信息
print("AUG 29-30, 1862.  THE COMBINED CONFEDERATE FORCES UNDER LEE\n");
print("AND JACKSON DROVE THE UNION FORCES BACK INTO WASHINGTON.\n");
break;
# 打印关于战争历史的信息
print("SEPT 17, 1862.  THE SOUTH FAILED TO INCORPORATE MARYLAND\n");
print("INTO THE CONFEDERACY.\n");
break;
                    case 6:
                        # 打印1862年12月13日的事件
                        print("DEC 13, 1862.  THE CONFEDERACY UNDER LEE SUCCESSFULLY\n");
                        # 打印1862年12月13日的事件
                        print("REPULSED AN ATTACK BY THE UNION UNDER GEN. BURNSIDE.\n");
                        # 跳出switch语句
                        break;
                    case 7:
                        # 打印1862年12月31日的事件
                        print("DEC 31, 1862.  THE SOUTH UNDER GEN. BRAGG WON A CLOSE BATTLE.\n");
                        # 跳出switch语句
                        break;
                    case 8:
                        # 打印1863年5月1日至6日的事件
                        print("MAY 1-6, 1863.  THE SOUTH HAD A COSTLY VICTORY AND LOST\n");
                        # 打印1863年5月1日至6日的事件
                        print("ONE OF THEIR OUTSTANDING GENERALS, 'STONEWALL' JACKSON.\n");
                        # 跳出switch语句
                        break;
                    case 9:
                        # 打印1863年7月4日的事件
                        print("JULY 4, 1863.  VICKSBURG WAS A COSTLY DEFEAT FOR THE SOUTH\n");
                        # 打印1863年7月4日的事件
                        print("BECAUSE IT GAVE THE UNION ACCESS TO THE MISSISSIPPI.\n");
                        # 跳出switch语句
                        break;
                    case 10:
                        # 打印1863年7月1日至3日的事件
                        print("JULY 1-3, 1863.  A SOUTHERN MISTAKE BY GEN. LEE AT GETTYSBURG\n");
                        # 打印1863年7月1日至3日的事件
                        print("COST THEM ONE OF THE MOST CRUCIAL BATTLES OF THE WAR.\n");
                        # 跳出switch语句
                        break;
                    case 11:
                        # 打印1863年9月15日的事件
                        print("SEPT. 15, 1863. CONFUSION IN A FOREST NEAR CHICKAMAUGA LED\n");
                        print("TO A COSTLY SOUTHERN VICTORY.\n");  # 打印文本内容"TO A COSTLY SOUTHERN VICTORY.\n"
                        break;  # 跳出当前的switch语句
                    case 12:  # 当switch的表达式的值为12时执行以下代码
                        print("NOV. 25, 1863. AFTER THE SOUTH HAD SIEGED GEN. ROSENCRANS'\n");  # 打印文本内容"NOV. 25, 1863. AFTER THE SOUTH HAD SIEGED GEN. ROSENCRANS'\n"
                        print("ARMY FOR THREE MONTHS, GEN. GRANT BROKE THE SIEGE.\n");  # 打印文本内容"ARMY FOR THREE MONTHS, GEN. GRANT BROKE THE SIEGE.\n"
                        break;  # 跳出当前的switch语句
                    case 13:  # 当switch的表达式的值为13时执行以下代码
                        print("MAY 5, 1864.  GRANT'S PLAN TO KEEP LEE ISOLATED BEGAN TO\n");  # 打印文本内容"MAY 5, 1864.  GRANT'S PLAN TO KEEP LEE ISOLATED BEGAN TO\n"
                        print("FAIL HERE, AND CONTINUED AT COLD HARBOR AND PETERSBURG.\n");  # 打印文本内容"FAIL HERE, AND CONTINUED AT COLD HARBOR AND PETERSBURG.\n"
                        break;  # 跳出当前的switch语句
                    case 14:  # 当switch的表达式的值为14时执行以下代码
                        print("AUGUST, 1864.  SHERMAN AND THREE VETERAN ARMIES CONVERGED\n");  # 打印文本内容"AUGUST, 1864.  SHERMAN AND THREE VETERAN ARMIES CONVERGED\n"
                        print("ON ATLANTA AND DEALT THE DEATH BLOW TO THE CONFEDERACY.\n");  # 打印文本内容"ON ATLANTA AND DEALT THE DEATH BLOW TO THE CONFEDERACY.\n"
                        break;  # 跳出当前的switch语句
                }
            }
        } else {
            print(cs + " INSTANT REPLAY\n");  # 打印变量cs的值和" INSTANT REPLAY\n"
        }
        print("\n");  # 打印一个空行
        print(" \tCONFEDERACY\t UNION\n"),  # 打印 CONFEDERACY 和 UNION
        print("MEN\t  " + m5 + "\t\t " + m6 + "\n");  # 打印 MEN 和 m5、m6 的值
        print("MONEY\t $" + da[1] + "\t\t$" + da[2] + "\n");  # 打印 MONEY 和 da[1]、da[2] 的值
        print("INFLATION\t " + (i1 + 15) + "%\t " + i2 + "%\n");  # 打印 INFLATION 和 i1、i2 的值
        print("\n");  # 打印空行
        // ONLY IN PRINTOUT IS CONFED INFLATION = I1 + 15%
        // IF TWO GENERALS, INPUT CONFED, FIRST
        for (i = 1; i <= d; i++) {  # 循环，i 从 1 到 d
            if (bs == "YES" && i == 1)  # 如果 bs 为 "YES" 并且 i 等于 1
                print("CONFEDERATE GENERAL---");  # 打印 CONFEDERATE GENERAL---
            print("HOW MUCH DO YOU WISH TO SPEND FOR\n");  # 打印 HOW MUCH DO YOU WISH TO SPEND FOR
            while (1) {  # 进入无限循环
                print(" - FOOD......");  # 打印 - FOOD......
                f = parseInt(await input());  # 将输入的值转换为整数并赋给 f
                if (f < 0) {  # 如果 f 小于 0
                    if (r1 == 0) {  # 如果 r1 等于 0
                        print("NO PREVIOUS ENTRIES\n");  # 打印 NO PREVIOUS ENTRIES
                        continue;  # 继续下一次循环
                    }
                    print("ASSUME YOU WANT TO KEEP SAME ALLOCATIONS\n");  # 打印 ASSUME YOU WANT TO KEEP SAME ALLOCATIONS
                    print("\n");  # 打印空行
                    break;  # 跳出循环
                }
                fa[i] = f;  # 将输入的值赋给fa[i]
                while (1):  # 进入循环
                    print(" - SALARIES..");  # 打印提示信息
                    ha[i] = parseInt(await input());  # 将输入的值转换为整数并赋给ha[i]
                    if (ha[i] >= 0):  # 如果ha[i]大于等于0
                        break;  # 跳出循环
                    print("NEGATIVE VALUES NOT ALLOWED.\n");  # 打印提示信息
                }
                while (1):  # 进入循环
                    print(" - AMMUNITION");  # 打印提示信息
                    ba[i] = parseInt(await input());  # 将输入的值转换为整数并赋给ba[i]
                    if (ba[i] >= 0):  # 如果ba[i]大于等于0
                        break;  # 跳出循环
                    print("NEGATIVE VALUES NOT ALLOWED.\n");  # 打印提示信息
                }
                print("\n");  # 打印空行
                if (fa[i] + ha[i] + ba[i] > da[i]):  # 如果fa[i] + ha[i] + ba[i]大于da[i]
                    print("THINK AGAIN! YOU HAVE ONLY $" + da[i] + "\n");  // 如果条件成立，打印提示信息
                } else {
                    break;  // 结束循环
                }
            }
            if (bs != "YES" || i == 2)  // 如果条件不成立，或者 i 等于 2，结束循环
                break;
            print("UNION GENERAL---");  // 打印提示信息
        }
        for (z = 1; z <= d; z++) {  // 循环，z 从 1 到 d
            if (bs == "YES") {  // 如果条件成立
                if (z == 1)
                    print("CONFEDERATE ");  // 打印提示信息
                else
                    print("      UNION ");  // 打印提示信息
            }
            // Find morale  // 查找士气
            o = ((2 * Math.pow(fa[z], 2) + Math.pow(ha[z], 2)) / Math.pow(f1, 2) + 1);  // 计算士气值
            if (o >= 10) {  // 如果条件成立
                print("MORALE IS HIGH\n");  // 打印提示信息
            } else if (o >= 5) {  # 如果 o 大于等于 5
                print("MORALE IS FAIR\n");  # 打印“士气一般”
            } else {  # 否则
                print("MORALE IS POOR\n");  # 打印“士气低落”
            }
            if (bs != "YES")  # 如果 bs 不等于 "YES"
                break;  # 跳出循环
            oa[z] = o;  # 将 o 赋值给 oa[z]
        }
        o2 = oa[2];  # 将 oa[2] 赋值给 o2
        o = oa[1];  # 将 oa[1] 赋值给 o
        print("CONFEDERATE GENERAL---");  # 打印“联邦将军---”
        // Actual off/def battle situation  # 实际的攻守战况
        if (m == 3) {  # 如果 m 等于 3
            print("YOU ARE ON THE OFFENSIVE\n");  # 打印“你处于进攻状态”
        } else if (m == 1) {  # 否则如果 m 等于 1
            print("YOU ARE ON THE DEFENSIVE\n");  # 打印“你处于防守状态”
        } else {  # 否则
            print("BOTH SIDES ARE ON THE OFFENSIVE \n");  # 打印“双方都处于进攻状态”
        }
        print("\n");  # 打印一个空行
        // Choose strategies  # 选择策略
        if (bs != "YES") {  # 如果 bs 不等于 "YES"
            print("YOUR STRATEGY ");  # 打印 "YOUR STRATEGY "
            while (1) {  # 进入无限循环
                y = parseInt(await input());  # 将输入的内容转换为整数并赋值给 y
                if (Math.abs(y - 3) < 3)  # 如果 y 与 3 的差的绝对值小于 3
                    break;  # 退出循环
                print("STRATEGY " + y + " NOT ALLOWED.\n");  # 打印 "STRATEGY " + y + " NOT ALLOWED.\n"
            }
            if (y == 5) {  # 如果 y 等于 5
                print("THE CONFEDERACY HAS SURRENDERED.\n");  # 打印 "THE CONFEDERACY HAS SURRENDERED.\n"
                break;  # 退出循环
            }
            // Union strategy is computer choesn  # 联盟策略是计算机选择的
            print("UNION STRATEGY IS ");  # 打印 "UNION STRATEGY IS "
            if (a == 0) {  # 如果 a 等于 0
                while (1) {  # 进入无限循环
                    y2 = parseInt(await input());  # 将输入的内容转换为整数并赋值给 y2
                    if (y2 > 0 && y2 < 5)  # 如果 y2 大于 0 并且小于 5
# 循环开始
for (i = 1; i <= 2; i++) {
    # 如果 i 等于 1，则打印 "CONFEDERATE STRATEGY "
    if (i == 1)
        print("CONFEDERATE STRATEGY ");
                while (1) {
                    // 从输入中获取一个整数并转换为数字
                    y = parseInt(await input());
                    // 如果 y 和 3 的差的绝对值小于 3，则跳出循环
                    if (Math.abs(y - 3) < 3)
                        break;
                    // 打印不允许的策略信息
                    print("STRATEGY " + y + " NOT ALLOWED.\n");
                }
                // 如果 i 等于 2
                if (i == 2) {
                    // 将 y2 设置为 y 的值
                    y2 = y;
                    // 将 y 设置为 y1 的值
                    y = y1;
                    // 如果 y2 不等于 5，则跳出循环
                    if (y2 != 5)
                        break;
                } else {
                    // 将 y1 设置为 y 的值
                    y1 = y;
                }
                // 打印联合策略信息
                print("UNION STRATEGY ");
            }
            // 模拟损失 - 北方
            // 计算 c6 的值
            c6 = (2 * c2 / 5) * (1 + 1 / (2 * (Math.abs(y2 - y) + 1)));
            // 根据公式计算 c6 的值
            c6 = c6 * (1.28 + (5 * m2 / 6) / (ba[2] + 1));
            // 对 c6 进行四舍五入取整
            c6 = Math.floor(c6 * (1 + 1 / o2) + 0.5);
            // 如果损失大于当前的人数，重新调整损失
            e2 = 100 / o2;
            if (Math.floor(c6 + e2) >= m6) {
                c6 = Math.floor(13 * m6 / 20);
                e2 = 7 * c6 / 13;
                u2 = 1;
            }
        }
        // 计算模拟损失
        print("\n");
        print("\n");
        print("\n");
        print("\t\tCONFEDERACY\tUNION\n");
        c5 = (2 * c1 / 5) * (1 + 1 / (2 * (Math.abs(y2 - y) + 1)));
        c5 = Math.floor(c5 * (1 + 1 / o) * (1.28 + f1 / (ba[1] + 1)) + 0.5);
        e = 100 / o;
        if (c5 + 100 / o >= m1 * (1 + (p1 - t1) / (m3 + 1))) {
            c5 = Math.floor(13 * m1 / 20 * (1 + (p1 - t1) / (m3 + 1)));
            e = 7 * c5 / 13;
            u = 1;
        }
        if (d == 1) {  // 如果 d 等于 1
            c6 = Math.floor(17 * c2 * c1 / (c5 * 20));  // 计算 c6 的值
            e2 = 5 * o;  // 计算 e2 的值
        }
        print("CASUALTIES\t" + c5 + "\t\t" + c6 + "\n");  // 打印伤亡人数
        print("DESERTIONS\t" + Math.floor(e) + "\t\t" + Math.floor(e2) + "\n");  // 打印叛逃人数
        print("\n");  // 打印空行
        if (bs == "YES") {  // 如果 bs 等于 "YES"
            print("COMPARED TO THE ACTUAL CASUALTIES AT " + cs + "\n");  // 打印与实际伤亡人数的比较
            print("CONFEDERATE: " + Math.floor(100 * (c5 / c1) + 0.5) + "% OF THE ORIGINAL\n");  // 打印南方伤亡人数占比
            print("UNION:       " + Math.floor(100 * (c6 / c2) + 0.5) + "% OF THE ORIGINAL\n");  // 打印北方伤亡人数占比
        }
        print("\n");  // 打印空行
        // 1 Who one
        if (u == 1 && u2 == 1 || (u != 1 && u2 != 1 && c5 + e == c6 + e2)) {  // 如果满足条件
            print("BATTLE OUTCOME UNRESOLVED\n");  // 打印战斗结果未解决
            w0++;  // w0 加一
        } else if (u == 1 || (u != 1 && u2 != 1 && c5 + e > c6 + e2)) {  // 如果满足条件
            print("THE UNION WINS " + cs + "\n");  // 打印北方获胜
            if (a != 0)
                l++;  // 如果a不等于0，则l加1
        } else  {
            print("THE CONFEDERACY WINS " + cs + "\n");  // 如果不满足上一个条件，则打印"The Confederacy Wins "和cs的值
            if (a != 0)
                w++;  // 如果a不等于0，则w加1
        }
        // Lines 2530 to 2590 from original are unreachable.  // 原始代码中2530到2590行是无法到达的
        if (a != 0) {
            t1 += c5 + e;  // 如果a不等于0，则t1加上c5和e的值
            t2 += c6 + e2;  // 如果a不等于0，则t2加上c6和e2的值
            p1 += c1;  // 如果a不等于0，则p1加上c1的值
            p2 += c2;  // 如果a不等于0，则p2加上c2的值
            q1 += fa[1] + ha[1] + ba[1];  // 如果a不等于0，则q1加上fa[1]、ha[1]和ba[1]的值
            q2 += fa[2] + ha[2] + ba[2];  // 如果a不等于0，则q2加上fa[2]、ha[2]和ba[2]的值
            r1 += m1 * (100 - i1) / 20;  // 如果a不等于0，则r1加上m1乘以(100 - i1)除以20的值
            r2 += m2 * (100 - i2) / 20;  // 如果a不等于0，则r2加上m2乘以(100 - i2)除以20的值
            m3 += m1;  // 如果a不等于0，则m3加上m1的值
            m4 += m2;  // 如果a不等于0，则m4加上m2的值
            // Learn present strategy, start forgetting old ones  // 学习当前策略，开始忘记旧的策略
            // 设定南方策略获得3*s分，其他方向失去s分
            // 除非某个策略下降到5%以下，否则概率得分
            s = 3;
            s0 = 0;
            for (i = 1; i <= 4; i++) {
                if (sa[i] <= 5)
                    continue; // 如果某个策略下降到5%以下，则跳过本次循环
                sa[i] -= 5; // 减去5%的概率
                s0 += s; // 计算总得分
            }
            sa[y] += s0; // 将总得分加到指定方向上
        }
        u = 0; // 重置u为0
        u2 = 0; // 重置u2为0
        print("---------------"); // 打印分隔线
        continue; // 继续下一次循环
    }
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 打印输出联邦赢得的战斗次数和失败的战斗次数
    print("THE CONFEDERACY HAS WON " + w + " BATTLES AND LOST " + l + "\n");
    # 如果联邦赢得的战斗次数等于5次或者（联邦赢得的战斗次数不等于5次并且赢得的战斗次数小于等于失败的战斗次数）
    if (y == 5 || (y2 != 5 && w <= l)) {
        # 打印输出联邦赢得了战争
        print("THE UNION HAS WON THE WAR\n");
    } else {
        # 打印输出联邦输掉了战争
        print("THE CONFEDERACY HAS WON THE WAR\n");
    }
    # 打印输出联邦和联盟的历史损失和模拟损失
    print("\n");
    if (r1) {
        print("FOR THE " + (w + l + w0) + " BATTLES FOUGHT (EXCLUDING RERUNS)\n");
        print(" \t \t ");
        print("CONFEDERACY\t UNION\n");
        print("HISTORICAL LOSSES\t" + Math.floor(p1 + 0.5) + "\t" + Math.floor(p2 + 0.5) + "\n");
        print("SIMULATED LOSSES\t" + Math.floor(t1 + 0.5) + "\t" + Math.floor(t2 + 0.5) + "\n");
        print("\n");
        print("    % OF ORIGINAL\t" + Math.floor(100 * (t1 / p1) + 0.5) + "\t" + Math.floor(100 * (t2 / p2) + 0.5) + "\n");
        # 如果bs不等于"YES"，则打印输出一个空行
        if (bs != "YES") {
            print("\n");
# 打印联盟情报表明南方使用了以下百分比的策略1、2、3、4
print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED \n");
print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES\n");
# 打印策略1、2、3、4的百分比
print(sa[1] + " " + sa[2] + " " + sa[3] + " " + sa[4] + "\n");
# 结束if语句
}
# 结束for循环
}
# 调用主函数
main();
```