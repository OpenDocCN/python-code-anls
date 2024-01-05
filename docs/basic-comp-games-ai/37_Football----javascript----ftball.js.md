# `37_Football\javascript\ftball.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入
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
# 函数定义结束
});
# 函数定义结束
}

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

}

var os = [];  # 创建一个空数组 os
var sa = [];  # 创建一个空数组 sa
var ls = [, "KICK","RECEIVE"," YARD ","RUN BACK FOR ","BALL ON ",
          "YARD LINE"," SIMPLE RUN"," TRICKY RUN"," SHORT PASS",
          " LONG PASS","PUNT"," QUICK KICK "," PLACE KICK"," LOSS ",
          " NO GAIN","GAIN "," TOUCHDOWN "," TOUCHBACK ","SAFETY***",
          "JUNK"];  # 创建一个包含字符串的数组 ls
var p;  # 声明变量 p
var x;  # 声明变量 x
var x1;  # 声明变量 x1

function fnf(x)  # 定义函数 fnf，接受参数 x
{
    return 1 - 2 * p;  # 返回表达式的计算结果
}
# 计算函数 fng 的返回值，根据输入参数 z 计算得到
function fng(z)
{
    return p * (x1 - x) + (1 - p) * (x - x1);
}

# 显示分数信息
function show_score()
{
    # 打印分数信息
    print("\n");
    print("SCORE:  " + sa[0] + " TO " + sa[1] + "\n");
    print("\n");
    print("\n");
}

# 显示位置信息
function show_position()
{
    # 如果 x 小于等于 50，则打印对应信息
    if (x <= 50) {
        print(ls[5] + os[0] + " " + x + " " + ls[6] + "\n");
    } else {
        # 否则打印另一组对应信息
        print(ls[5] + os[1] + " " + (100 - x) + " " + ls[6] + "\n");
    }
}
}

# 定义一个名为 offensive_td 的函数，用于处理进攻得分
function offensive_td()
{
    # 打印 ls 列表中索引为 17 的元素加上 "***" 的字符串
    print(ls[17] + "***\n");
    # 如果随机数小于等于 0.8
    if (Math.random() <= 0.8) {
        # 将 sa 列表中索引为 p 的元素加上 7
        sa[p] = sa[p] + 7;
        # 打印 "KICK IS GOOD."
        print("KICK IS GOOD.\n");
    } else {
        # 打印 "KICK IS OFF TO THE SIDE"
        print("KICK IS OFF TO THE SIDE\n");
        # 将 sa 列表中索引为 p 的元素加上 6
        sa[p] = sa[p] + 6;
    }
    # 调用 show_score 函数显示比分
    show_score();
    # 打印 os 列表中索引为 p 的元素加上 " KICKS OFF" 的字符串
    print(os[p] + " KICKS OFF\n");
    # 将 p 取反
    p = 1 - p;
}

# 主程序
async function main()
{
    # 打印字符串"FTBALL"，使用tab函数在前面添加33个空格
    print(tab(33) + "FTBALL\n");
    # 打印字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，使用tab函数在前面添加15个空格
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    # 打印两个空行
    print("\n");
    print("\n");
    # 打印字符串"THIS IS DARTMOUTH CHAMPIONSHIP FOOTBALL."
    print("THIS IS DARTMOUTH CHAMPIONSHIP FOOTBALL.\n");
    # 打印一个空行
    print("\n");
    # 打印字符串"YOU WILL QUARTERBACK DARTMOUTH. CALL PLAYS AS FOLLOWS:"
    print("YOU WILL QUARTERBACK DARTMOUTH. CALL PLAYS AS FOLLOWS:\n");
    # 打印一系列可选的比赛策略
    print("1= SIMPLE RUN; 2= TRICKY RUN; 3= SHORT PASS;\n");
    print("4= LONG PASS; 5= PUNT; 6= QUICK KICK; 7= PLACE KICK.\n");
    # 打印字符串"CHOOSE YOUR OPPONENT"
    print("CHOOSE YOUR OPPONENT")
    # 从用户输入中获取对手的选择，并存储在os[1]中
    os[1] = await input();
    # 将字符串"DARMOUTH"存储在os[0]中
    os[0] = "DARMOUTH";
    # 初始化sa数组的两个元素为0
    sa[0] = 0;
    sa[1] = 0;
    # 生成一个随机数p，范围在0到1之间
    p = Math.floor(Math.random() * 2);
    # 打印对手赢得抛硬币的信息
    print(os[p] + " WON THE TOSS\n");
    # 如果p不等于0，则打印对手选择接球
    if (p != 0) {
        print(os[1] + " ELECTS TO RECEIVE.\n");
        print("\n");  # 打印空行
    } else {  # 否则
        print("DO YOU ELECT TO KICK OR RECEIVE");  # 打印提示信息
        while (1) {  # 进入循环
            str = await input();  # 等待用户输入
            print("\n");  # 打印空行
            if (str == ls[1] || str == ls[2])  # 如果用户输入等于ls列表中的第一个或第二个元素
                break;  # 退出循环
            print("INCORRECT ANSWER.  PLEASE TYPE 'KICK' OR 'RECEIVE'");  # 打印提示信息
        }
        e = (str == ls[1]) ? 1 : 2;  # 如果用户输入等于ls列表中的第一个元素，则e为1，否则为2
        if (e == 1)  # 如果e为1
            p = 1;  # 则p为1
    }
    t = 0;  # t赋值为0
    start = 1;  # start赋值为1
    while (1) {  # 进入循环
        if (start <= 1) {  # 如果start小于等于1
            x = 40 + (1 - p) * 20;  # 计算x的值
        }
        if (start <= 2) {  # 如果比赛开始的阶段小于等于2
            y = Math.floor(200 * Math.pow((Math.random() - 0.5), 3) + 55);  # 计算一个随机数并赋值给y
            print(" " + y + " " + ls[3] + " KICKOFF\n");  # 打印输出一段文本
            x = x - fnf(1) * y;  # 根据计算结果更新x的值
            if (Math.abs(x - 50) >= 50) {  # 如果x与50的差的绝对值大于等于50
                print("TOUCHBACK FOR " + os[p] + ".\n");  # 打印输出一段文本
                x = 20 + p * 60;  # 更新x的值
                start = 4;  # 更新比赛开始的阶段
            } else {
                start = 3;  # 更新比赛开始的阶段
            }
        }
        if (start <= 3) {  # 如果比赛开始的阶段小于等于3
            y = Math.floor(50 * Math.pow(Math.random(), 2)) + (1 - p) * Math.floor(50 * Math.pow(Math.random(), 4));  # 计算一个随机数并赋值给y
            x = x + fnf(1) * y;  # 根据计算结果更新x的值
            if (Math.abs(x - 50) < 50) {  # 如果x与50的差的绝对值小于50
                print(" " + y + " " + ls[3] + " RUNBACK\n");  # 打印输出一段文本
            } else {
                print(ls[4]);  # 打印输出一段文本
                offensive_td();  # 调用函数
                start = 1;  // 设置变量 start 的值为 1
                continue;  // 继续执行下一次循环
            }
        }
        if (start <= 4) {  // 如果 start 的值小于等于 4
            // First down  // 第一次进攻
            show_position();  // 显示位置信息
        }
        if (start <= 5) {  // 如果 start 的值小于等于 5
            x1 = x;  // 将变量 x 的值赋给 x1
            d = 1;  // 设置变量 d 的值为 1
            print("\n");  // 打印换行符
            print("FIRST DOWN " + os[p] + "***\n");  // 打印 "FIRST DOWN " 后接变量 os[p] 的值再接 "***" 和换行符
            print("\n");  // 打印两个换行符
            print("\n");  // 再次打印两个换行符
        }
        // New play  // 新的进攻
        t++;  // 变量 t 的值加一
        if (t == 30) {  // 如果 t 的值等于 30
            if (Math.random() <= 1.3) {  // 如果随机数小于等于 1.3
                # 打印"GAME DELAYED.  DOG ON FIELD."字符串
                print("GAME DELAYED.  DOG ON FIELD.\n");
                # 打印换行符
                print("\n");
            }
        }
        # 如果t大于等于50并且随机数小于等于0.2，则跳出循环
        if (t >= 50 && Math.random() <= 0.2)
            break;
        # 如果p不等于1
        if (p != 1) {
            # 对手的出牌
            if (d <= 1) {
                # 如果随机数大于1/3，则z等于1，否则z等于3
                z = Math.random() > 1 / 3 ? 1 : 3;
            } else if (d != 4) {
                if (10 + x - x1 < 5 || x < 5) {
                    # 如果10 + x - x1小于5或者x小于5，则如果随机数大于1/3，则z等于1，否则z等于3
                    z = Math.random() > 1 / 3 ? 1 : 3;
                } else if (x <= 10) {
                    # 取0到1之间的随机整数，赋值给a
                    a = Math.floor(2 * Math.random());
                    # z等于2加上a
                    z = 2 + a;
                } else if (x <= x1 || d < 3 || x < 45) {
                    # 取0到1之间的随机整数，赋值给a
                    a = Math.floor(2 * Math.random());
                    # z等于2加上a乘以2
                    z = 2 + a * 2;
                } else {
                    if (Math.random() > 1 / 4)  # 如果随机数大于1/4
                        z = 4;  # 则将z赋值为4
                    else
                        z = 6;  # 否则将z赋值为6
                }
            } else {
                if (x <= 30) {  # 如果x小于等于30
                    z = 5;  # 则将z赋值为5
                } else if (10 + x - x1 < 3 || x < 3) {  # 否则如果10 + x - x1小于3或者x小于3
                    z = Math.random() > 1 / 3 ? 1 : 3;  # 则根据随机数大于1/3的概率将z赋值为1或3
                } else {
                    z = 7;  # 否则将z赋值为7
                }
            }
        } else {
            print("NEXT PLAY");  # 打印"NEXT PLAY"
            while (1) {  # 进入循环，条件为1（永远为真）
                z = parseInt(await input());  # 将输入的值转换为整数并赋值给z
                if (Math.abs(z - 4) <= 3)  # 如果z与4的差的绝对值小于等于3
                    break;  # 则跳出循环
                print("ILLEGAL PLAY NUMBER, RETYPE");  # 打印错误消息，要求重新输入
            }  # 结束 if 语句块
        }  # 结束 for 循环
        f = 0;  # 初始化变量 f 为 0
        print(ls[z + 6] + ".  ");  # 打印 ls[z + 6] 的值并加上句号和空格
        r = Math.random() * (0.98 + fnf(1) * 0.02);  # 生成一个随机数并赋值给变量 r
        r1 = Math.random();  # 生成一个随机数并赋值给变量 r1
        switch (z) {  # 开始 switch 语句块，根据 z 的值进行不同的处理
            case 1: // Simple run  # 当 z 的值为 1 时执行以下代码，表示简单跑
            case 2: // Tricky run  # 当 z 的值为 2 时执行以下代码，表示复杂跑
                if (z == 1) {  # 如果 z 的值为 1
                    y = Math.floor(24 * Math.pow(r - 0.5, 3) + 3);  # 计算 y 的值
                    if (Math.random() >= 0.05) {  # 如果生成的随机数大于等于 0.05
                        routine = 1;  # 将 routine 的值设为 1
                        break;  # 跳出 switch 语句块
                    }  # 结束 if 语句块
                } else {  # 如果 z 的值不为 1
                    y = Math.floor(20 * r - 5);  # 计算 y 的值
                    if (Math.random() > 0.1) {  # 如果生成的随机数大于 0.1
                        routine = 1;  # 将 routine 的值设为 1
                break;  // 结束当前循环或者 switch 语句的执行
            }
        }
        f = -1;  // 将变量 f 赋值为 -1
        x3 = x;  // 将变量 x3 赋值为 x 的值
        x = x + fnf(1) * y;  // 将变量 x 的值加上 fnf(1) 乘以 y 的值
        if (Math.abs(x - 50) < 50) {  // 如果 x 与 50 的差的绝对值小于 50
            print("***  FUMBLE AFTER ");  // 打印 "***  FUMBLE AFTER "
            routine = 2;  // 将变量 routine 赋值为 2
            break;  // 结束当前循环或者 switch 语句的执行
        } else {
            print("***  FUMBLE.\n");  // 打印 "***  FUMBLE.\n"
            routine = 4;  // 将变量 routine 赋值为 4
            break;  // 结束当前循环或者 switch 语句的执行
        }
    case 3:  // 如果 z 的值为 3，执行以下代码
    case 4:  // 如果 z 的值为 4，执行以下代码
        if (z == 3) {  // 如果 z 的值为 3
            y = Math.floor(60 * Math.pow(r1 - 0.5, 3) + 10);  // 计算 y 的值
        } else {  // 如果 z 的值不为 3
# 如果 z 等于 3 并且 r 小于 0.05，或者 z 等于 4 并且 r 小于 0.1
if (z == 3 && r < 0.05 || z == 4 && r < 0.1) {
    # 如果 d 不等于 4
    if (d != 4) {
        # 打印 "INTERCEPTED."
        print("INTERCEPTED.\n");
        # 将 f 设为 -1
        f = -1;
        # 将 x 增加 fnf(1) 乘以 y
        x = x + fnf(1) * y;
        # 如果 x 与 50 的差的绝对值大于等于 50
        if (Math.abs(x - 50) >= 50) {
            # 将 routine 设为 4
            routine = 4;
            # 跳出循环
            break;
        }
        # 将 routine 设为 3
        routine = 3;
        # 跳出循环
        break;
    } else {
        # 将 y 设为 0
        y = 0;
        # 如果 Math.random() 小于 0.3
        if (Math.random() < 0.3) {
            # 打印 "BATTED DOWN.  "
            print("BATTED DOWN.  ");
        } else {
            # 打印 "INCOMPLETE.  "
            print("INCOMPLETE.  ");
        }
    }
}
# 设置变量 routine 为 1
routine = 1;
# 跳出循环
break;
# 如果 z 等于 4 并且 r 小于 0.3
} else if (z == 4 && r < 0.3) {
    # 打印 "PASSER TACKLED."
    print("PASSER TACKLED.  ");
    # 设置变量 y 为 -15 * r1 + 3 的向下取整
    y = -Math.floor(15 * r1 + 3);
    # 设置变量 routine 为 1
    routine = 1;
    # 跳出循环
    break;
# 如果 z 等于 3 并且 r 小于 0.15
} else if (z == 3 && r < 0.15) {
    # 打印 "PASSER TACLKED."
    print("PASSER TACLKED.  ");
    # 设置变量 y 为 -10 * r1 的向下取整
    y = -Math.floor(10 * r1);
    # 设置变量 routine 为 1
    routine = 1;
    # 跳出循环
    break;
# 如果 z 等于 3 并且 r 小于 0.55 或者 z 等于 4 并且 r 小于 0.75
} else if (z == 3 && r < 0.55 || z == 4 && r < 0.75) {
    # 设置变量 y 为 0
    y = 0;
    # 如果随机数小于 0.3
    if (Math.random() < 0.3) {
        # 打印 "BATTED DOWN."
        print("BATTED DOWN.  ");
    } else {
        # 否则打印 "INCOMPLETE."
        print("INCOMPLETE.  ");
    }
                    routine = 1;  // 设置变量routine的值为1
                    break;  // 跳出switch语句
                } else {  // 如果条件不成立
                    print("COMPLETE.  ");  // 打印输出COMPLETE.
                    routine = 1;  // 设置变量routine的值为1
                    break;  // 跳出switch语句
                }
            case 5:  // 如果switch的值为5
            case 6:  // 如果switch的值为6
                y = Math.floor(100 * Math.pow((r - 0.5), 3) + 35);  // 计算y的值
                if (d != 4)  // 如果d不等于4
                    y = Math.floor(y * 1.3);  // 计算y的值
                print(" " + y + " " + ls[3] + " PUNT\n");  // 打印输出y、ls[3]和PUNT
                if (Math.abs(x + y * fnf(1) - 50) < 50 && d >= 4) {  // 如果条件成立
                    y1 = Math.floor(Math.pow(r1, 2) * 20);  // 计算y1的值
                    print(" " + y1 + " " + ls[3] + " RUN BACK\n");  // 打印输出y1、ls[3]和RUN BACK
                    y = y - y1;  // 计算y的值
                }
                f = -1;  // 设置变量f的值为-1
                x = x + fnf(1) * y;  // 计算x的值
                if (Math.abs(x - 50) >= 50) {  # 如果球门中心和球员位置的横向距离大于等于50
                    routine = 4;  # 设置动作为4
                    break;  # 跳出循环
                }
                routine = 3;  # 设置动作为3
                break;  # 跳出循环
            case 7: // Place kick  # 如果动作为7，表示进行定位踢球
                y = Math.floor(100 * Math.pow((r - 0.5), 3) + 35);  # 计算踢球的力度
                if (r1 <= 0.15) {  # 如果随机数r1小于等于0.15
                    print("KICK IS BLOCKED  ***\n");  # 打印“踢球被挡住”
                    x = x - 5 * fnf(1);  # 球员位置向后移动5个单位
                    p = 1 - p;  # 改变p的值
                    start = 4;  # 设置开始动作为4
                    continue;  # 继续下一次循环
                }
                x = x + fnf(1) * y;  # 根据力度和方向计算球员位置
                if (Math.abs(x - 50) >= 60) {  # 如果球门中心和球员位置的横向距离大于等于60
                    if (r1 <= 0.5) {  # 如果随机数r1小于等于0.5
                        print("KICK IS OFF TO THE SIDE.\n");  # 打印“踢偏了”
                        print(ls[18] + "\n");  # 打印ls列表中第18个元素
                        p = 1 - p;  # 切换球队控球权
                        x = 20 + p * 60;  # 更新球场上的位置
                        start = 4;  # 设置下一次进攻的起始位置
                        continue;  # 继续执行循环
                    } else {
                        print("FIELD GOAL ***\n");  # 打印信息，表示进攻方踢进了三分球
                        sa[p] = sa[p] + 3;  # 更新得分
                        show_score();  # 显示当前比分
                        print(os[p] + " KICKS OFF\n");  # 打印信息，表示失去控球权的球队将进行开球
                        p = 1 - p;  # 切换球队控球权
                        start = 1;  # 设置下一次进攻的起始位置
                        continue;  # 继续执行循环
                    }
                } else {
                    print("KICK IS SHORT.\n");  # 打印信息，表示踢球距离不够远
                    if (Math.abs(x - 50) >= 50) {  # 判断是否为触底球
                        // Touchback  # 注释，表示触底球
                        print(ls[18] + "\n");  # 打印信息，表示触底球
                        p = 1 - p;  # 切换球队控球权
                        x = 20 + p * 60;  # 更新球场上的位置
                        start = 4;  // 设置变量 start 的值为 4
                        continue;  // 跳过当前循环的剩余代码，继续下一次循环
                    }
                    p = 1 - p;  // 计算 p 的新值
                    start = 3;  // 设置变量 start 的值为 3
                    continue;  // 跳过当前循环的剩余代码，继续下一次循环
                }

        }
        // Gain or loss
        if (routine <= 1) {  // 如果 routine 的值小于等于 1
            x3 = x;  // 将 x 的值赋给 x3
            x = x + fnf(1) * y;  // 计算新的 x 的值
            if (Math.abs(x - 50) >= 50) {  // 如果 x 与 50 的差的绝对值大于等于 50
                routine = 4;  // 设置 routine 的值为 4
            }
        }
        if (routine <= 2) {  // 如果 routine 的值小于等于 2
            if (y != 0) {  // 如果 y 不等于 0
                print(" " + Math.abs(y) + " " + ls[3]);  // 打印信息
                # 如果 y 小于 0，则将 yt 设置为 -1
                if (y < 0)
                    yt = -1;
                # 如果 y 大于 0，则将 yt 设置为 1
                else if (y > 0)
                    yt = 1;
                # 否则将 yt 设置为 0
                else
                    yt = 0;
                # 打印 ls 列表中索引为 15 + yt 的元素
                print(ls[15 + yt]);
                # 如果 x3 与 50 的差的绝对值小于等于 40 并且随机数小于 0.1
                if (Math.abs(x3 - 50) <= 40 && Math.random() < 0.1) {
                    # 犯规
                    p3 = Math.floor(2 * Math.random());
                    # 打印 os 列表中索引为 p3 的元素和一段提示信息
                    print(os[p3] + " OFFSIDES -- PENALTY OF 5 YARDS.\n");
                    # 打印空行
                    print("\n");
                    # 打印空行
                    print("\n");
                    # 如果 p3 不等于 0
                    if (p3 != 0) {
                        # 打印提示信息
                        print("DO YOU ACCEPT THE PENALTY");
                        # 循环直到输入为 "YES" 或 "NO"
                        while (1) {
                            str = await input();
                            if (str == "YES" || str == "NO")
                                break;
                            # 打印提示信息
                            print("TYPE 'YES' OR 'NO'");
                        }
                        如果 (str == "YES") {
                            f = 0;  // 重置变量 f 为 0
                            d = d - 1;  // 变量 d 减 1
                            if (p != p3)  // 如果 p 不等于 p3
                                x = x3 + fnf(1) * 5;  // x 等于 x3 加上 fnf(1) 乘以 5
                            else
                                x = x3 - fnf(1) * 5;  // 否则 x 等于 x3 减去 fnf(1) 乘以 5
                        }
                    } else {
                        // 对手在点球时的策略
                        if ((p != 1 && (y <= 0 || f < 0 || fng(1) < 3 * d - 2))
                            || (p == 1 && ((y > 5 && f >= 0) || d < 4 || fng(1) >= 10))) {
                            print("PENALTY REFUSED.\n");  // 打印 "点球被拒绝"
                        } else {
                            print("PENALTY ACCEPTED.\n");  // 打印 "点球被接受"
                            f = 0;  // 重置变量 f 为 0
                            d = d - 1;  // 变量 d 减 1
                            if (p != p3)  // 如果 p 不等于 p3
                                x = x3 + fnf(1) * 5;  // x 等于 x3 加上 fnf(1) 乘以 5
# 如果routine小于等于3，则执行以下操作
if (routine <= 3) {
    # 显示位置
    show_position();
    # 如果f不等于0，则执行以下操作
    if (f != 0) {
        # p等于1减去p
        p = 1 - p;
        # start等于5
        start = 5;
        # 继续执行下一轮循环
        continue;
    # 如果fng(1)大于等于10，则执行以下操作
    } else if (fng(1) >= 10) {
        # start等于5
        start = 5;
        # 继续执行下一轮循环
        continue;
    # 如果d等于4，则执行以下操作
    } else if (d == 4) {
        # p等于1减去p
        p = 1 - p;
        # start等于5
        start = 5;
                continue;  # 继续执行下一次循环
            } else:
                d += 1  # 下降码数加一
                print("DOWN: " + str(d) + "     ")  # 打印当前下降次数
                if ((x1 - 50) * fnf(1) >= 40):  # 如果（球场位置 - 50）* fnf(1) 大于等于40
                    print("GOAL TO GO\n")  # 打印进攻目标达成
                else:
                    print("YARDS TO GO: " + str(10 - fng(1)) + "\n")  # 打印剩余码数
                print("\n")  # 打印空行
                print("\n")  # 打印空行
                start = 6  # 开始位置设为6
                continue  # 继续执行下一次循环
        if (routine <= 4):  # 如果routine小于等于4
            # Ball in end-zone
            e = 1 if (x >= 100) else 0  # 如果x大于等于100，e为1，否则为0
            case = 1 + e - f * 2 + p * 4  # 计算case的值
            switch case:  # 根据case的值进行分支
                case 1:  # 如果case的值为1
                case 5:
                    // Safety
                    // 将另一方的得分增加2分
                    sa[1 - p] = sa[1 - p] + 2;
                    // 打印信息
                    print(ls[19] + "\n");
                    // 显示比分
                    show_score();
                    // 打印信息
                    print(os[p] + " KICKS OFF FROM ITS 20 YARD LINE.\n");
                    // 计算新的位置
                    x = 20 + p * 60;
                    // 切换球权
                    p = 1 - p;
                    // 设置开始状态
                    start = 2;
                    // 继续执行下一条语句
                    continue;
                case 3:
                case 6:
                    // Defensive TD
                    // 打印信息
                    print(ls[17] + "FOR " + os[1 - p] + "***\n");
                    // 切换球权
                    p = 1 - p;
                    // 继续执行下一条语句
                    // Fall-thru
                case 2:
                case 8:
                    // Offensive TD
                    // 打印信息
                    print(ls[17] + "***\n");
                    if (Math.random() <= 0.8) {  // 如果随机数小于等于0.8
                        sa[p] = sa[p] + 7;  // 球队p得到7分
                        print("KICK IS GOOD.\n");  // 打印“KICK IS GOOD.”
                    } else {
                        print("KICK IS OFF TO THE SIDE\n");  // 打印“KICK IS OFF TO THE SIDE”
                        sa[p] = sa[p] + 6;  // 球队p得到6分
                    }
                    show_score();  // 显示比分
                    print(os[p] + " KICKS OFF\n");  // 打印球队p的名称和“KICKS OFF”
                    p = 1 - p;  // 切换球队
                    start = 1;  // 设置start为1
                    continue;  // 继续执行下一轮循环
                case 4:
                case 7:
                    // Touchback  // 触地得分
                    print(ls[18] + "\n");  // 打印ls[18]
                    p = 1 - p;  // 切换球队
                    x = 20 + p * 60;  // 设置x的值
                    start = 4;  // 设置start为4
                    continue;  // 继续执行下一轮循环
    }
```
这是一个缺少注释的代码片段，看起来是一个代码块的结束，但缺少上下文无法确定其作用。
```