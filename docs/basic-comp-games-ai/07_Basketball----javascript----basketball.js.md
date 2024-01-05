# `07_Basketball\javascript\basketball.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到输出元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从输出元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为tab的函数，接受一个参数space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当space大于0时，执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格字符添加到字符串变量str的末尾
    return str;  # 返回处理后的字符串

}

var s = [0, 0];  # 创建一个包含两个元素的数组变量s，初始值为0
var z;  # 声明一个未赋值的变量z
var d;  # 声明一个未赋值的变量d
var p;  # 声明一个未赋值的变量p
var your_turn;  # 声明一个未赋值的变量your_turn
var game_restart;  # 声明一个未赋值的变量game_restart

function two_minutes()  # 定义一个名为two_minutes的函数
{
    print("\n");  # 打印一个换行符
    print("   *** TWO MINUTES LEFT IN THE GAME ***\n");  # 打印提示信息
    print("\n");  # 打印一个换行符
}

function show_scores()  # 定义一个名为show_scores的函数
{
    print("SCORE: " + s[1] + " TO " + s[0] + "\n");
    # 打印比分信息，s[1]代表玩家得分，s[0]代表计算机得分
}

function score_computer()
{
    s[0] = s[0] + 2;
    show_scores();
    # 计算机得分加2，并显示更新后的比分
}

function score_player()
{
    s[1] = s[1] + 2;
    show_scores();
    # 玩家得分加2，并显示更新后的比分
}

function half_time()
{
    print("\n");
    print("   ***** END OF FIRST HALF *****\n");
    print("SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
    # 打印上半场结束时的比分信息，s[1]代表玩家得分，s[0]代表计算机得分，os代表对手的名称
}
    print("\n");  # 打印空行
    print("\n");  # 打印空行
}

function foul()  # 定义名为 foul 的函数
{
    if (Math.random() <= 0.49) {  # 如果随机数小于等于0.49
        print("SHOOTER MAKES BOTH SHOTS.\n");  # 打印射手两次投篮都命中
        s[1 - p] = s[1 - p] + 2;  # 更新得分
        show_scores();  # 显示得分
    } else if (Math.random() <= 0.75) {  # 否则如果随机数小于等于0.75
        print("SHOOTER MAKES ONE SHOT AND MISSES ONE.\n");  # 打印射手一次投篮命中一次，一次未命中
        s[1 - p] = s[1 - p] + 1;  # 更新得分
        show_scores();  # 显示得分
    } else {  # 否则
        print("BOTH SHOTS MISSED.\n");  # 打印两次投篮都未命中
        show_scores();  # 显示得分
    }
}
# 定义名为 player_play 的函数
def player_play():
    # 如果 z 的值为 1 或 2
    if (z == 1 or z == 2):
        # t 值加一
        t += 1
        # 如果 t 的值等于 50
        if (t == 50):
            # 调用 half_time 函数
            half_time()
            # 将 game_restart 的值设为 1
            game_restart = 1
            # 返回
            return
        # 如果 t 的值等于 92
        if (t == 92):
            # 调用 two_minutes 函数
            two_minutes()
        # 打印 "JUMP SHOT\n"
        print("JUMP SHOT\n")
        # 如果随机数小于等于 0.341 乘以 d 除以 8
        if (Math.random() <= 0.341 * d / 8):
            # 打印 "SHOT IS GOOD.\n"
            print("SHOT IS GOOD.\n")
            # 调用 score_player 函数
            score_player()
            # 返回
            return
        # 如果随机数小于等于 0.682 乘以 d 除以 8
        if (Math.random() <= 0.682 * d / 8):
            # 打印 "SHOT IS OFF TARGET.\n"
            print("SHOT IS OFF TARGET.\n")
            # 如果 d 除以 6 乘以随机数大于等于 0.45
            if (d / 6 * Math.random() >= 0.45):
                # 打印信息，指示球被对手抢断，并返回
                print("REBOUND TO " + os + "\n");
                return;
            }
            # 打印信息，指示达特茅斯控制篮板
            print("DARTMOUTH CONTROLS THE REBOUND.\n");
            # 如果随机数大于0.4
            if (Math.random() > 0.4) {
                # 如果骰子数为6
                if (d == 6) {
                    # 如果随机数大于0.6
                    if (Math.random() > 0.6) {
                        # 打印信息，指示球被对手抢断并进行上篮得分
                        print("PASS STOLEN BY " + os + " EASY LAYUP.\n");
                        # 对手得分
                        score_computer();
                        return;
                    }
                }
                # 打印信息，指示球传回你手中
                print("BALL PASSED BACK TO YOU. ");
                # 标记轮到你进攻
                your_turn = 1;
                return;
            }
        } else if (Math.random() <= 0.782 * d / 8) {
            # 打印信息，指示投篮被阻挡
            print("SHOT IS BLOCKED.  BALL CONTROLLED BY ");
            # 如果随机数小于等于0.5
            if (Math.random() <= 0.5) {
                # 打印信息，指示球被达特茅斯控制
                print("DARTMOUTH.\n");
                your_turn = 1;  # 设置变量your_turn为1，表示轮到你的回合
                return;  # 返回
            }
            print(os + ".\n");  # 打印os加上换行符
            return;  # 返回
        } else if (Math.random() <= 0.843 * d / 8) {  # 如果随机数小于等于0.843乘以d除以8
            print("SHOOTER IS FOULED.  TWO SHOTS.\n");  # 打印“投手犯规。两次投篮。”
            foul();  # 调用foul函数
            return;  # 返回
            // In original code but lines 1180-1195 aren't used (maybe replicate from computer's play)
            //        } else if (Math.random() <= 0.9 * d / 8) {
            //            print("PLAYER FOULED, TWO SHOTS.\n");
            //            foul();
            //            return;
        } else {
            print("CHARGING FOUL.  DARTMOUTH LOSES BALL.\n");  # 打印“冲撞犯规。达特茅斯失球。”
            return;  # 返回
        }
    }
    while (1) {  # 进入无限循环
        # 如果 t 的值加一等于 50，则执行 half_time 函数，设置 game_restart 为 1，然后返回
        if (++t == 50) {
            half_time();
            game_restart = 1;
            return;
        }
        # 如果 t 的值等于 92，则执行 two_minutes 函数
        if (t == 92)
            two_minutes();
        # 如果 z 的值等于 0，则将 your_turn 设置为 2，然后返回
        if (z == 0) {
            your_turn = 2;
            return;
        }
        # 如果 z 的值小于等于 3，则打印 "LAY UP.\n"
        if (z <= 3)
            print("LAY UP.\n");
        # 否则打印 "SET SHOT.\n"
        else
            print("SET SHOT.\n");
        # 如果 7 除以 d 乘以 Math.random() 的值小于等于 0.4，则打印 "SHOT IS GOOD.  TWO POINTS.\n"，执行 score_player 函数，然后返回
        if (7 / d * Math.random() <= 0.4) {
            print("SHOT IS GOOD.  TWO POINTS.\n");
            score_player();
            return;
        }
        # 如果随机数满足条件，表示投篮偏离篮筐
        if (7 / d * Math.random() <= 0.7) {
            print("SHOT IS OFF THE RIM.\n");
            # 如果随机数满足条件，表示你控制篮板
            if (Math.random() <= 2.0 / 3.0) {
                print(os + " CONTROLS THE REBOUND.\n");
                return;
            }
            # 如果你没有控制篮板，达特茅斯队控制篮板
            print("DARMOUTH CONTROLS THE REBOUND.\n");
            # 如果随机数满足条件，表示球传回你手中
            if (Math.random() <= 0.4)
                continue;
            print("BALL PASSED BACK TO YOU.\n");
            your_turn = 1;
            return;
        }
        # 如果随机数满足条件，表示投篮者犯规，两罚
        if (7 /d * Math.random() <= 0.875) {
            print("SHOOTER FOULED.  TWO SHOTS.\n");
            foul();
            return;
        }
        # 如果随机数满足条件，表示投篮被阻挡，对方球权
        if (7 /d * Math.random() <= 0.925) {
            print("SHOT BLOCKED. " + os + "'S BALL.\n");
            return;  # 结束当前函数的执行
        }
        print("CHARGING FOUL.  DARTHMOUTH LOSES THE BALL.\n");  # 打印信息，指出达特茅斯队犯规失球
        return;  # 结束当前函数的执行
    }
}

function computer_play()
{
    rebound = 0;  # 初始化篮板球数为0
    while (1) {  # 进入无限循环
        p = 1;  # 将p设为1
        if (++t == 50) {  # 如果t自增后等于50
            half_time();  # 调用half_time函数
            game_restart = 1;  # 将game_restart设为1
            return;  # 结束当前函数的执行
        }
        print("\n");  # 打印换行符
        z1 = 10 / 4 * Math.random() + 1;  # 生成一个随机数并赋值给z1
        if (z1 <= 2) {  # 如果z1小于等于2
            # 打印"JUMP SHOT.\n"
            print("JUMP SHOT.\n");
            # 如果 8 / d * Math.random() 小于等于 0.35，则打印"SHOT IS GOOD.\n"，调用 score_computer() 函数，然后返回
            if (8 / d * Math.random() <= 0.35) {
                print("SHOT IS GOOD.\n");
                score_computer();
                return;
            }
            # 如果 8 / d * Math.random() 小于等于 0.75，则打印"SHOT IS OFF RIM.\n"
            if (8 / d * Math.random() <= 0.75) {
                print("SHOT IS OFF RIM.\n");
                # 如果 d / 6 * Math.random() 小于等于 0.5，则打印"DARMOUTH CONTROLS THE REBOUND.\n"，然后返回
                if (d / 6 * Math.random() <= 0.5) {
                    print("DARMOUTH CONTROLS THE REBOUND.\n");
                    return;
                }
                # 打印 os + " CONTROLS THE REBOUND.\n"
                print(os + " CONTROLS THE REBOUND.\n");
                # 如果 d 等于 6，则继续执行下面的代码
                if (d == 6) {
                    # 如果 Math.random() 小于等于 0.75，则打印"BALL STOLEN.  EASY LAP UP FOR DARTMOUTH.\n"，调用 score_player() 函数，然后继续循环
                    if (Math.random() <= 0.75) {
                        print("BALL STOLEN.  EASY LAP UP FOR DARTMOUTH.\n");
                        score_player();
                        continue;
                    }
                    # 如果 Math.random() 大于 0.6，则执行下面的代码
                    if (Math.random() > 0.6) {
# 打印信息，说明球被对方抢断并容易上篮得分
print("PASS STOLEN BY " + os + " EASY LAYUP.\n")
# 调用计算机得分函数
score_computer()
# 返回函数，结束程序
return

# 打印信息，说明球被传回给你
print("BALL PASSED BACK TO YOU. ")
# 返回函数，结束程序
return

# 如果随机数小于等于0.5，打印信息，说明球被传回给对方的后卫
if (Math.random() <= 0.5):
    print("PASS BACK TO " + os + " GUARD.\n")
    # 继续循环
    continue

# 如果随机数小于等于0.90，打印信息，说明球员犯规，需要罚两次球
elif (8 / d * Math.random() <= 0.90):
    print("PLAYER FOULED.  TWO SHOTS.\n")
    # 调用犯规函数
    foul()
    # 返回函数，结束程序
    return

# 否则，打印信息，说明进攻犯规，达特茅斯队得到球权
else:
    print("OFFENSIVE FOUL.  DARTMOUTH'S BALL.\n")
    # 返回函数，结束程序
    return
        while (1) {  # 进入一个无限循环
            if (z1 > 3) {  # 如果 z1 大于 3
                print("SET SHOT.\n");  # 打印 "SET SHOT."
            } else {  # 否则
                print("LAY UP.\n");  # 打印 "LAY UP."
            }
            if (7 / d * Math.random() <= 0.413) {  # 如果 7 除以 d 乘以一个随机数小于等于 0.413
                print("SHOT IS GOOD.\n");  # 打印 "SHOT IS GOOD."
                score_computer();  # 调用 score_computer() 函数
                return;  # 结束函数执行
            }
            print("SHOT IS MISSED.\n");  # 打印 "SHOT IS MISSED."
            // Spaguetti jump, better to replicate code  # 注释：意味着这段代码可能混乱，最好重写
            if (d / 6 * Math.random() <= 0.5) {  # 如果 d 除以 6 乘以一个随机数小于等于 0.5
                print("DARMOUTH CONTROLS THE REBOUND.\n");  # 打印 "DARMOUTH CONTROLS THE REBOUND."
                return;  # 结束函数执行
            }
            print(os + " CONTROLS THE REBOUND.\n");  # 打印 os 和 " CONTROLS THE REBOUND."
            if (d == 6) {  # 如果 d 等于 6
                if (Math.random() <= 0.75) {  # 如果随机数小于等于 0.75
                    # 打印信息，表示球被抢断，达特茅斯轻松上篮得分
                    print("BALL STOLEN.  EASY LAP UP FOR DARTMOUTH.\n");
                    # 球员得分
                    score_player();
                    # 跳出循环
                    break;
                }
                # 如果随机数大于0.6
                if (Math.random() > 0.6) {
                    # 打印信息，表示球被对手抢断，对手轻松上篮得分
                    print("PASS STOLEN BY " + os + " EASY LAYUP.\n");
                    # 对手得分
                    score_computer();
                    # 返回
                    return;
                }
                # 打印信息，表示球传回给你
                print("BALL PASSED BACK TO YOU. ");
                # 返回
                return;
            }
            # 如果随机数小于等于0.5
            if (Math.random() <= 0.5) {
                # 打印信息，表示球传回给对手的后卫
                print("PASS BACK TO " + os + " GUARD.\n");
                # 跳出循环
                break;
            }
        }
    }
}
// 主程序
async function main()
{
    // 打印标题
    print(tab(31) + "BASKETBALL\n");
    // 打印创意计算的地址
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏介绍
    print("THIS IS DARTMOUTH COLLEGE BASKETBALL.  YOU WILL BE DARTMOUTH\n");
    print(" CAPTAIN AND PLAYMAKER.  CALL SHOTS AS FOLLOWS:  1. LONG\n");
    print(" (30 FT.) JUMP SHOT; 2. SHORT (15 FT.) JUMP SHOT; 3. LAY\n");
    print(" UP; 4. SET SHOT.\n");
    print("BOTH TEAMS WILL USE THE SAME DEFENSE.  CALL DEFENSE AS\n");
    print("FOLLOWS:  6. PRESS; 6.5 MAN-TO MAN; 7. ZONE; 7.5 NONE.\n");
    print("TO CHANGE DEFENSE, JUST TYPE 0 AS YOUR NEXT SHOT.\n");
    print("YOUR STARTING DEFENSE WILL BE");
    t = 0;  // 初始化t变量为0
    p = 0;  // 初始化p变量为0
    d = parseFloat(await input());  // 从用户输入获取浮点数并赋值给d变量
    if (d < 6) {  // 如果d小于6
        your_turn = 2;  # 设置变量your_turn的值为2
    } else {
        print("\n");  # 打印一个空行
        print("CHOOSE YOUR OPPONENT");  # 打印提示信息"CHOOSE YOUR OPPONENT"
        os = await input();  # 从用户输入中获取对手的选择，并赋值给变量os
        game_restart = 1;  # 设置变量game_restart的值为1
    }
    while (1) {  # 进入无限循环
        if (game_restart) {  # 如果game_restart为真
            game_restart = 0;  # 将game_restart的值设为0
            print("CENTER JUMP\n");  # 打印提示信息"CENTER JUMP"并换行
            if (Math.random() > 3.0 / 5.0) {  # 如果随机数大于3/5
                print("DARMOUTH CONTROLS THE TAP.\n");  # 打印提示信息"DARMOUTH CONTROLS THE TAP."并换行
            } else {
                print(os + " CONTROLS THE TAP.\n");  # 打印对手的选择加上"CONTROLS THE TAP."并换行
                computer_play();  # 调用computer_play函数
            }
        }
        if (your_turn == 2) {  # 如果your_turn的值为2
            print("YOUR NEW DEFENSIVE ALLIGNMENT IS");  # 打印提示信息"YOUR NEW DEFENSIVE ALLIGNMENT IS"
        # 从输入中获取浮点数并赋值给变量d
        d = parseFloat(await input());
        # 打印换行符
        print("\n");
        # 进入无限循环，直到条件满足才会跳出循环
        while (1) {
            # 打印"YOUR SHOT"
            print("YOUR SHOT");
            # 从输入中获取整数并赋值给变量z
            z = parseInt(await input());
            # 初始化变量p为0
            p = 0;
            # 如果z不是整数或者小于0或者大于4，则打印错误信息并要求重新输入
            if (z != Math.floor(z) || z < 0 || z > 4)
                print("INCORRECT ANSWER.  RETYPE IT. ");
            else
                # 如果z满足条件，则跳出循环
                break;
        }
        # 如果随机数小于0.5或者t小于100，则执行以下代码块
        if (Math.random() < 0.5 || t < 100) {
            # 将game_restart和your_turn都赋值为0
            game_restart = 0;
            your_turn = 0;
            # 调用player_play函数
            player_play();
            # 如果game_restart为0且your_turn为0，则调用computer_play函数
            if (game_restart == 0 && your_turn == 0)
                computer_play();
        } else {
            # 打印换行符
            print("\n");
            if (s[1] == s[0]) {  # 如果第二个队伍的得分等于第一个队伍的得分
                print("\n");  # 打印空行
                print("   ***** END OF SECOND HALF *****\n");  # 打印比赛下半场结束的提示
                print("\n");  # 打印空行
                print("SCORE AT END OF REGULATION TIME:\n");  # 打印常规时间结束时的比分
                print("        DARTMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");  # 打印两队的得分
                print("\n");  # 打印空行
                print("BEGIN TWO MINUTE OVERTIME PERIOD\n");  # 打印开始两分钟的加时赛
                t = 93;  # 设置时间为93
                print("CENTER JUMP\n");  # 打印中心跳球
                if (Math.random() > 3.0 / 5.0)  # 如果随机数大于3/5
                    print("DARMOUTH CONTROLS THE TAP.\n");  # 打印达特茅斯控球
                else
                    print(os + " CONTROLS THE TAP.\n");  # 否则打印对手控球
            } else {  # 否则
                print("   ***** END OF GAME *****\n");  # 打印比赛结束的提示
                print("FINAL SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");  # 打印最终比分
                break;  # 跳出循环
            }
        }
    }
}
```
这部分代码是一个函数的结束和一个主函数的调用。在Python中，函数的定义以关键字def开始，后面是函数名和参数列表，然后是函数体。在示例中，函数read_zip(fname)的定义从def开始，一直到最后的return语句结束。接着是一个空行，然后是一个右大括号}，表示函数定义的结束。紧接着是一个空行，然后是一个主函数的调用main()，这里假设有一个名为main的函数用来执行程序的主要逻辑。
```