# `39_Golf\javascript\golf.js`

```
# GOLF
#
# Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
#

# 定义一个打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义一个输入函数，返回一个 Promise 对象
def input():
    var input_element
    var input_str

    return new Promise(function (resolve) {
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT")

                       # 打印提示符
                       print("? ")
                       # 设置输入元素的类型为文本
                       input_element.setAttribute("type", "text")
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
var la = []; // 创建一个空数组 la
var f; // 声明一个变量 f，未赋值
var s1; // 声明一个变量 s1，未赋值
var g2; // 声明一个变量 g2，未赋值
var g3; // 声明一个变量 g3，未赋值
var x; // 声明一个变量 x，未赋值

var hole_data = [ // 创建一个名为 hole_data 的数组，包含一系列数字
    361,4,4,2,389,4,3,3,206,3,4,2,500,5,7,2,
    408,4,2,4,359,4,6,4,424,4,4,2,388,4,4,4,
    196,3,7,2,400,4,7,2,560,5,7,2,132,3,2,2,
    357,4,4,4,294,4,2,4,475,5,2,3,375,4,4,2,
    180,3,6,2,550,5,6,6,
];

function show_obstacle() // 声明一个名为 show_obstacle 的函数
# 根据 la[x] 的值进行不同的操作
switch (la[x]) {
    # 如果 la[x] 的值为 1，则打印 "FAIRWAY.\n" 并跳出 switch 语句
    case 1:
        print("FAIRWAY.\n");
        break;
    # 如果 la[x] 的值为 2，则打印 "ROUGH.\n" 并跳出 switch 语句
    case 2:
        print("ROUGH.\n");
        break;
    # 如果 la[x] 的值为 3，则打印 "TREES.\n" 并跳出 switch 语句
    case 3:
        print("TREES.\n");
        break;
    # 如果 la[x] 的值为 4，则打印 "ADJACENT FAIRWAY.\n" 并跳出 switch 语句
    case 4:
        print("ADJACENT FAIRWAY.\n");
        break;
    # 如果 la[x] 的值为 5，则打印 "TRAP.\n" 并跳出 switch 语句
    case 5:
        print("TRAP.\n");
        break;
    # 如果 la[x] 的值为 6，则打印 "WATER.\n" 并跳出 switch 语句
    case 6:
        print("WATER.\n");
        break;
    }
}
```
这是一个函数的结束标志。

```
function show_score()
{
    g2 += s1;
    print("TOTAL PAR FOR " + (f - 1) + " HOLES IS " + g3 + "  YOUR TOTAL IS " + g2 + "\n");
}
```
这是一个名为show_score的函数，它用于计算和显示得分。

```
// Main program
async function main()
{
    print(tab(34) + "GOLF\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("WELCOME TO THE CREATIVE COMPUTING COUNTRY CLUB,\n");
    print("AN EIGHTEEN HOLE CHAMPIONSHIP LAYOUT LOCATED A SHORT\n");
    print("DISTANCE FROM SCENIC DOWNTOWN MORRISTOWN.  THE\n");
```
这是一个名为main的异步函数，它是程序的主要部分，用于打印欢迎消息和介绍信息。
    # 打印提示信息，告知玩家游戏进行时会有注释员解说
    print("COMMENTATOR WILL EXPLAIN THE GAME AS YOU PLAY.\n");
    # 打印提示信息，祝玩家游戏愉快并期待在19洞见面
    print("ENJOY YOUR GAME; SEE YOU AT THE 19TH HOLE...\n");
    # 打印空行
    print("\n");
    # 打印空行
    print("\n");
    # 初始化下一洞的变量
    next_hole = 0;
    # 初始化第一洞的成绩
    g1 = 18;
    # 初始化第二洞的成绩
    g2 = 0;
    # 初始化第三洞的成绩
    g3 = 0;
    # 初始化总成绩
    a = 0;
    # 初始化标准杆
    n = 0.8;
    # 初始化第二洞的杆数
    s2 = 0;
    # 初始化标志位
    f = 1;
    # 进入循环，等待玩家输入
    while (1) {
        # 打印提示信息，询问玩家的差点
        print("WHAT IS YOUR HANDICAP");
        # 获取玩家输入的差点并转换为整数
        h = parseInt(await input());
        # 打印空行
        print("\n");
        # 如果差点小于0或大于30，则打印提示信息
        if (h < 0 || h > 30) {
            print("PGA HANDICAPS RANGE FROM 0 TO 30.\n");
        } else {
            # 差点在合理范围内则跳出循环
            break;
    }
    do {
        # 打印困难的高尔夫问题列表
        print("DIFFICULTIES AT GOLF INCLUDE:\n");
        # 打印高尔夫困难问题的编号和描述
        print("0=HOOK, 1=SLICE, 2=POOR DISTANCE, 4=TRAP SHOTS, 5=PUTTING\n");
        # 打印提示信息，要求用户输入最糟糕的高尔夫问题编号
        print("WHICH ONE (ONLY ONE) IS YOUR WORST");
        # 从用户输入中获取最糟糕的高尔夫问题编号
        t = parseInt(await input());
        # 打印换行符
        print("\n");
    } while (t > 5) ;
    # 初始化变量 s1
    s1 = 0;
    # 初始化变量 first_routine
    first_routine = true;
    # 进入无限循环
    while (1) {
        # 如果是第一次执行循环
        if (first_routine) {
            # 初始化数组 la 的第一个元素为 0
            la[0] = 0;
            # 初始化变量 j 为 0
            j = 0;
            # 初始化变量 q 为 0
            q = 0;
            # 变量 s2 自增 1
            s2++;
            # 初始化变量 k 为 0
            k = 0;
            # 如果 f 不等于 1
            if (f != 1) {
                # 打印上一洞的得分
                print("YOUR SCORE ON HOLE " + (f - 1) + " WAS " + s1 + "\n");
                show_score();  // 调用函数显示当前比赛得分
                if (g1 == f - 1)    // 判断是否完成了所有的比赛洞
                    return;         // 如果完成了，退出游戏
                if (s1 > p + 2) {   // 如果当前洞的杆数大于标准杆加2
                    print("KEEP YOUR HEAD DOWN.\n");  // 打印信息提示保持低头
                } else if (s1 == p) {  // 如果当前洞的杆数等于标准杆
                    print("A PAR.  NICE GOING.\n");  // 打印信息提示平标准杆，做得很好
                } else if (s1 == p - 1) {  // 如果当前洞的杆数比标准杆少1杆
                    print("A BIRDIE.\n");  // 打印信息提示小鸟球
                } else if (s1 == p - 2) {  // 如果当前洞的杆数比标准杆少2杆
                    if (p != 3)  // 如果标准杆不等于3
                        print("A GREAT BIG EAGLE.\n");  // 打印信息提示大鹰
                    else
                        print("A HOLE IN ONE.\n");  // 否则打印信息提示一杆进洞
                }
            }
            if (f == 19) {  // 如果当前洞数等于19
                print("\n");  // 打印换行
                show_score();  // 调用函数显示当前比赛得分
                if (g1 == f - 1)  // 判断是否完成了所有的比赛洞
                    return;  # 如果条件满足，直接返回，结束函数执行
            }
            s1 = 0;  # 将变量 s1 的值设为 0
            print("\n");  # 打印换行符
            if (s1 != 0 && la[0] < 1)  # 如果 s1 不等于 0 并且 la[0] 小于 1
                la[0] = 1;  # 将 la[0] 的值设为 1
        }
        if (s1 == 0) {  # 如果 s1 的值等于 0
            d = hole_data[next_hole++];  # 从 hole_data 中获取下一个值并赋给变量 d
            p = hole_data[next_hole++];  # 从 hole_data 中获取下一个值并赋给变量 p
            la[1] = hole_data[next_hole++];  # 从 hole_data 中获取下一个值并赋给 la[1]
            la[2] = hole_data[next_hole++];  # 从 hole_data 中获取下一个值并赋给 la[2]
            print("\n");  # 打印换行符
            print("YOU ARE AT THE TEE OFF HOLE " + f + " DISTANCE " + d + " YARDS, PAR " + p + "\n");  # 打印提示信息
            g3 += p;  # 变量 g3 的值加上 p
            print("ON YOUR RIGHT IS ");  # 打印提示信息
            x = 1;  # 将变量 x 的值设为 1
            show_obstacle();  # 调用 show_obstacle 函数
            print("ON YOUR LEFT IS ");  # 打印提示信息
            x = 2  # 将变量 x 的值设为 2
            show_obstacle();  // 调用函数show_obstacle()来展示障碍物信息
        } else {
            x = 0;  // 将变量x赋值为0
            if (la[0] > 5) {  // 如果数组la的第一个元素大于5
                if (la[0] > 6) {  // 如果数组la的第一个元素大于6
                    print("YOUR SHOT WENT OUT OF BOUNDS.\n");  // 打印信息"YOUR SHOT WENT OUT OF BOUNDS."
                } else {
                    print("YOUR SHOT WENT INTO THE WATER.\n");  // 打印信息"YOUR SHOT WENT INTO THE WATER."
                }
                s1++;  // 变量s1加1
                print("PENALTY STROKE ASSESSED.  HIT FROM PREVIOUS LOCATION.\n");  // 打印信息"PENALTY STROKE ASSESSED.  HIT FROM PREVIOUS LOCATION."
                j++;  // 变量j加1
                la[0] = 1;  // 将数组la的第一个元素赋值为1
                d = b;  // 将变量d赋值为b
            } else {
                print("SHOT WENT " + d1 + " YARDS.  IT'S " + d2 + " YARDS FROM THE CUP.\n");  // 打印信息"SHOT WENT " + d1 + " YARDS.  IT'S " + d2 + " YARDS FROM THE CUP."
                print("BALL IS " + Math.floor(o) + " YARDS OFF LINE... IN ");  // 打印信息"BALL IS " + Math.floor(o) + " YARDS OFF LINE... IN "
                show_obstacle();  // 调用函数show_obstacle()来展示障碍物信息
            }
        }
        while (1):  # 进入无限循环
            if (a != 1):  # 如果a不等于1
                print("SELECTION OF CLUBS\n")  # 打印"SELECTION OF CLUBS"
                print("YARDAGE DESIRED                       SUGGESTED CLUBS\n")  # 打印"YARDAGE DESIRED                       SUGGESTED CLUBS"
                print("200 TO 280 YARDS                           1 TO 4\n")  # 打印"200 TO 280 YARDS                           1 TO 4"
                print("100 TO 200 YARDS                          19 TO 13\n")  # 打印"100 TO 200 YARDS                          19 TO 13"
                print("  0 TO 100 YARDS                          29 TO 23\n")  # 打印"  0 TO 100 YARDS                          29 TO 23"
                a = 1  # 将a设置为1
            print("WHAT CLUB DO YOU CHOOSE")  # 打印"WHAT CLUB DO YOU CHOOSE"
            c = parseInt(await input())  # 从输入中获取值并将其转换为整数，赋值给c
            print("\n")  # 打印换行
            if (c >= 1 and c <= 29 and (c < 5 or c >= 12)):  # 如果c大于等于1且小于等于29且（c小于5或者大于等于12）
                if (c > 4):  # 如果c大于4
                    c -= 6  # c减去6
                if (la[0] <= 5 or c == 14 or c == 23):  # 如果la的第一个元素小于等于5或者c等于14或者c等于23
                    s1 += 1  # s1加1
                    w = 1  # 将w设置为1
                    if (c <= 13):  # 如果c小于等于13
# 如果 f 除以 3 的余数为 0，并且 s2 + q + (10 * (f - 1) / 18) 小于 (f - 1) * (72 + ((h + 1) / 0.85)) / 18
if (f % 3 == 0 && s2 + q + (10 * (f - 1) / 18) < (f - 1) * (72 + ((h + 1) / 0.85)) / 18) {
    # 增加 q 的值
    q++;
    # 如果 s1 除以 2 的余数不为 0 并且 d 大于等于 95
    if (s1 % 2 != 0 && d >= 95) {
        # 打印信息并更新 d 的值
        print("BALL HIT TREE - BOUNCED INTO ROUGH " + (d - 75) + " YARDS FROM HOLE.\n");
        d -= 75;
        # 继续下一次循环
        continue;
    }
    # 打印信息
    print("YOU DUBBED IT.\n");
    # 更新 d1 和 second_routine 的值
    d1 = 35;
    second_routine = 1;
    # 退出循环
    break;
# 如果 c 小于 4 并且 la 列表的第一个元素等于 2
} else if (c < 4 && la[0] == 2) {
    # 打印信息
    print("YOU DUBBED IT.\n");
    # 更新 d1 和 second_routine 的值
    d1 = 35;
    second_routine = 1;
    # 退出循环
    break;
# 否则
} else {
    # 更新 second_routine 的值
    second_routine = 0;
    # 退出循环
    break;
}
                    } else {
                        # 打印提示信息，要求用户输入一个百分比来衡量距离
                        print("NOW GAUGE YOUR DISTANCE BY A PERCENTAGE (1 TO 100)\n");
                        print("OF A FULL SWING");
                        # 读取用户输入的百分比并转换为整数
                        w = parseInt(await input());
                        w /= 100;  # 将输入的百分比转换为小数
                        print("\n");
                        if (w <= 1) {  # 如果输入的百分比小于等于1
                            if (la[0] == 5) {  # 如果条件满足
                                if (t == 3) {  # 如果条件满足
                                    if (Math.random() <= n) {  # 如果随机数小于等于n
                                        n *= 0.2;  # 更新n的值
                                        print("SHOT DUBBED, STILL IN TRAP.\n");  # 打印提示信息
                                        continue;  # 继续下一次循环
                                    }
                                    n = 0.8;  # 更新n的值
                                }
                                d2 = 1 + (3 * Math.floor((80 / (40 - h)) * Math.random()));  # 更新d2的值
                                second_routine = 2;  # 更新second_routine的值
                                break;  # 跳出循环
                            }
                            if (c != 14)  # 如果c不等于14
                                c -= 10;  # 则c减去10
                            second_routine = 0;  # 将second_routine设置为0
                            break;  # 跳出循环
                        }
                        s1--;  # s1减1
                        // Fall through to THAT CLUB IS NOT IN THE BAG.  # 跳转到THAT CLUB IS NOT IN THE BAG.
                    }
                }
            }
            print("THAT CLUB IS NOT IN THE BAG.\n");  # 打印"THAT CLUB IS NOT IN THE BAG."
            print("\n");  # 打印换行
        }
        if (second_routine == 0) {  # 如果second_routine等于0
            if (s1 > 7 && d < 200) {  # 如果s1大于7并且d小于200
                d2 = 1 + (3 * Math.floor((80 / (40 - h)) * Math.random()));  # 计算d2的值
                second_routine = 2;  # 将second_routine设置为2
            } else {
                d1 = Math.floor(((30 - h) * 2.5 + 187 - ((30 - h) * 0.25 + 15) * c / 2) + 25 * Math.random());  # 计算d1的值
                d1 = Math.floor(d1 * w);  # 计算d1的值乘以w
                # 如果 t 等于 2，则将 d1 乘以 0.85 并向下取整
                if (t == 2)
                    d1 = Math.floor(d1 * 0.85);
            }
        }
        # 如果 second_routine 小于等于 1
        if (second_routine <= 1) {
            # 计算 o 的值
            o = (Math.random() / 0.8) * (2 * h + 16) * Math.abs(Math.tan(d1 * 0.0035));
            # 计算 d2 的值
            d2 = Math.floor(Math.sqrt(Math.pow(o, 2) + Math.pow(Math.abs(d - d1), 2)));
            # 如果 d - d1 小于 0
            if (d - d1 < 0) {
                # 如果 d2 大于等于 20，则打印提示信息
                if (d2 >= 20)
                    print("TOO MUCH CLUB, YOU'RE PAST THE HOLE.\n");
            }
            # 将 d 的值赋给 b
            b = d;
            # 将 d2 的值赋给 d
            d = d2;
            # 如果 d2 大于 27
            if (d2 > 27) {
                # 如果 o 小于 30 或 j 大于 0
                if (o < 30 || j > 0) {
                    # 将 la 数组的第一个元素设为 1
                    la[0] = 1;
                } else {
                    # 如果 t 小于等于 0
                    if (t <= 0) {
                        # 计算 s9 的值
                        s9 = (s2 + 1) / 15;
                        # 如果 s9 的向下取整等于 s9
                        if (Math.floor(s9) == s9) {
                            print("YOU SLICED- ");  # 打印“YOU SLICED- ”
                            la[0] = la[1];  # 将列表la的第一个元素赋值为la的第二个元素
                        } else {
                            print("YOU HOOKED- ");  # 打印“YOU HOOKED- ”
                            la[0] = la[2];  # 将列表la的第一个元素赋值为la的第三个元素
                        }
                    } else {
                        s9 = (s2 + 1) / 15;  # 计算s9的值
                        if (Math.floor(s9) == s9):  # 如果s9的整数部分等于s9本身
                            print("YOU HOOKED- ");  # 打印“YOU HOOKED- ”
                            la[0] = la[2];  # 将列表la的第一个元素赋值为la的第三个元素
                        else:
                            print("YOU SLICED- ");  # 打印“YOU SLICED- ”
                            la[0] = la[1];  # 将列表la的第一个元素赋值为la的第二个元素
                        }
                    }
                    if (o > 45):  # 如果o大于45
                        print("BADLY.\n");  # 打印“BADLY.”
                }
                first_routine = false;  # 将first_routine赋值为false
            } else if (d2 > 20) {  # 如果 d2 大于 20
                la[0] = 5;  # 将 la 数组的第一个元素设置为 5
                first_routine = false;  # 将 first_routine 变量设置为 false
            } else if (d2 > 0.5) {  # 如果 d2 大于 0.5
                la[0] = 8;  # 将 la 数组的第一个元素设置为 8
                d2 = Math.floor(d2 * 3);  # 将 d2 乘以 3 并向下取整
                second_routine = 2;  # 将 second_routine 设置为 2
            } else {  # 否则
                la[0] = 9;  # 将 la 数组的第一个元素设置为 9
                print("YOU HOLED IT.\n");  # 打印消息 "YOU HOLED IT.\n"
                print("\n");  # 打印空行
                f++;  # f 变量加一
                first_routine = true;  # 将 first_routine 变量设置为 true
            }
        }
        if (second_routine == 2) {  # 如果 second_routine 等于 2
            while (1) {  # 进入无限循环
                print("ON GREEN, " + d2 + " FEET FROM THE PIN.\n");  # 打印消息 "ON GREEN, " + d2 + " FEET FROM THE PIN.\n"
                print("CHOOSE YOUR PUTT POTENCY (1 TO 13):");  # 打印消息 "CHOOSE YOUR PUTT POTENCY (1 TO 13):"
                i = parseInt(await input());  # 将用户输入转换为整数并赋值给 i
                s1++;  // 增加 s1 的值
                if (s1 + 1 - p <= (h * 0.072) + 2 && k <= 2) {  // 如果满足条件
                    k++;  // 增加 k 的值
                    if (t == 4)  // 如果 t 的值等于 4
                        d2 -= i * (4 + 1 * Math.random()) + 1;  // 对 d2 进行计算
                    else
                        d2 -= i * (4 + 2 * Math.random()) + 1.5;  // 对 d2 进行计算
                    if (d2 < -2) {  // 如果 d2 小于 -2
                        print("PASSED BY CUP.\n");  // 打印信息
                        d2 = Math.floor(-d2);  // 对 d2 进行取整
                        continue;  // 继续下一次循环
                    }
                    if (d2 > 2) {  // 如果 d2 大于 2
                        print("PUTT SHORT.\n");  // 打印信息
                        d2 = Math.floor(d2);  // 对 d2 进行取整
                        continue;  // 继续下一次循环
                    }
                }
                print("YOU HOLED IT.\n");  // 打印信息
                print("\n");  // 打印空行
                f++;  # 增加变量 f 的值
                break;  # 跳出当前循环
            }
            first_routine = true;  # 将变量 first_routine 的值设为 true
        }
    }
}

main();  # 调用名为 main 的函数
```