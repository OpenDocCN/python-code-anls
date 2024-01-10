# `basic-computer-games\07_Basketball\javascript\basketball.js`

```
// BASKETBALL
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义一个打印函数，将字符串输出到指定的元素上
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

                       // 输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素上
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量空格组成的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一些全局变量
var s = [0, 0];
var z;
var d;
var p;
var your_turn;
var game_restart;

// 定义一个两分钟剩余的函数
function two_minutes()
{
    print("\n");
    print("   *** TWO MINUTES LEFT IN THE GAME ***\n");
    print("\n");
}

// 定义一个展示比分的函数
function show_scores()
{
    print("SCORE: " + s[1] + " TO " + s[0] + "\n");
}

// 定义一个给电脑加分的函数
function score_computer()
{
    s[0] = s[0] + 2;
    show_scores();
}

// 定义一个给玩家加分的函数
function score_player()
{
    s[1] = s[1] + 2;
    show_scores();
}

// 定义一个中场休息的函数
function half_time()
{
    print("\n");
    # 打印第一半比赛结束的提示信息
    print("   ***** END OF FIRST HALF *****\n");
    # 打印得分信息，包括达特茅斯队和对手队伍的得分
    print("SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
    # 打印两个空行
    print("\n");
    print("\n");
# 定义一个名为foul的函数，用于处理犯规情况
function foul()
{
    # 如果随机数小于等于0.49，表示投篮命中两次
    if (Math.random() <= 0.49) {
        # 打印投篮命中两次的信息
        print("SHOOTER MAKES BOTH SHOTS.\n");
        # 更新得分数组中对方球员的得分
        s[1 - p] = s[1 - p] + 2;
        # 显示更新后的得分
        show_scores();
    } 
    # 如果随机数大于0.49且小于等于0.75，表示投篮命中一次，未命中一次
    else if (Math.random() <= 0.75) {
        # 打印投篮命中一次，未命中一次的信息
        print("SHOOTER MAKES ONE SHOT AND MISSES ONE.\n");
        # 更新得分数组中对方球员的得分
        s[1 - p] = s[1 - p] + 1;
        # 显示更新后的得分
        show_scores();
    } 
    # 如果随机数大于0.75，表示两次投篮都未命中
    else {
        # 打印两次投篮都未命中的信息
        print("BOTH SHOTS MISSED.\n");
        # 显示更新后的得分
        show_scores();
    }
}

# 定义一个名为player_play的函数，用于处理球员的比赛表现
function player_play()
{
    # 如果 z 的值为 1 或 2，则执行以下操作
    if (z == 1 || z == 2) {
        # t 值加一
        t++;
        # 如果 t 的值为 50，则执行 half_time() 函数，设置 game_restart 为 1，然后返回
        if (t == 50) {
            half_time();
            game_restart = 1;
            return;
        }
        # 如果 t 的值为 92，则执行 two_minutes() 函数
        if (t == 92)
            two_minutes();
        # 打印 "JUMP SHOT"
        print("JUMP SHOT\n");
        # 如果随机数小于等于 0.341 * d / 8，则打印 "SHOT IS GOOD."，执行 score_player() 函数，然后返回
        if (Math.random() <= 0.341 * d / 8) {
            print("SHOT IS GOOD.\n");
            score_player();
            return;
        }
        # 如果随机数小于等于 0.682 * d / 8，则打印 "SHOT IS OFF TARGET."
        if (Math.random() <= 0.682 * d / 8) {
            print("SHOT IS OFF TARGET.\n");
            # 如果 d / 6 乘以随机数大于等于 0.45，则打印 "REBOUND TO " + os，然后返回
            if (d / 6 * Math.random() >= 0.45) {
                print("REBOUND TO " + os + "\n");
                return;
            }
            # 否则打印 "DARTMOUTH CONTROLS THE REBOUND."
            print("DARTMOUTH CONTROLS THE REBOUND.\n");
            # 如果随机数大于 0.4，则执行以下操作
            if (Math.random() > 0.4) {
                # 如果 d 的值为 6 且随机数大于 0.6，则打印 "PASS STOLEN BY " + os + " EASY LAYUP."，执行 score_computer() 函数，然后返回
                if (d == 6) {
                    if (Math.random() > 0.6) {
                        print("PASS STOLEN BY " + os + " EASY LAYUP.\n");
                        score_computer();
                        return;
                    }
                }
                # 否则打印 "BALL PASSED BACK TO YOU."，设置 your_turn 为 1，然后返回
                print("BALL PASSED BACK TO YOU. ");
                your_turn = 1;
                return;
            }
        } else if (Math.random() <= 0.782 * d / 8) {
            # 如果随机数小于等于 0.782 * d / 8，则打印 "SHOT IS BLOCKED.  BALL CONTROLLED BY "
            print("SHOT IS BLOCKED.  BALL CONTROLLED BY ");
            # 如果随机数小于等于 0.5，则打印 "DARTMOUTH."，设置 your_turn 为 1，然后返回
            if (Math.random() <= 0.5) {
                print("DARTMOUTH.\n");
                your_turn = 1;
                return;
            }
            # 否则打印 os + "."，然后返回
            print(os + ".\n");
            return;
        } else if (Math.random() <= 0.843 * d / 8) {
            # 如果随机数小于等于 0.843 * d / 8，则打印 "SHOOTER IS FOULED.  TWO SHOTS."
            print("SHOOTER IS FOULED.  TWO SHOTS.\n");
            # 执行 foul() 函数，然后返回
            foul();
            return;
            # 在原始代码中，但是行 1180-1195 没有被使用（可能是从计算机的玩法复制过来）
            #        } else if (Math.random() <= 0.9 * d / 8) {
            #            print("PLAYER FOULED, TWO SHOTS.\n");
            #            foul();
            #            return;
        } else {
            # 否则打印 "CHARGING FOUL.  DARTMOUTH LOSES BALL."，然后返回
            print("CHARGING FOUL.  DARTMOUTH LOSES BALL.\n");
            return;
        }
    }
    # 进入循环，直到条件不满足
    while (1) {
        # 如果 t 自增后等于 50，执行 half_time 函数，设置 game_restart 为 1，然后返回
        if (++t == 50) {
            half_time();
            game_restart = 1;
            return;
        }
        # 如果 t 等于 92，执行 two_minutes 函数
        if (t == 92)
            two_minutes();
        # 如果 z 等于 0，设置 your_turn 为 2，然后返回
        if (z == 0) {
            your_turn = 2;
            return;
        }
        # 如果 z 小于等于 3，打印 "LAY UP."
        if (z <= 3)
            print("LAY UP.\n");
        # 否则，打印 "SET SHOT."
        else
            print("SET SHOT.\n");
        # 如果 7 除以 d 乘以一个随机数小于等于 0.4，打印 "SHOT IS GOOD.  TWO POINTS."，执行 score_player 函数，然后返回
        if (7 / d * Math.random() <= 0.4) {
            print("SHOT IS GOOD.  TWO POINTS.\n");
            score_player();
            return;
        }
        # 如果 7 除以 d 乘以一个随机数小于等于 0.7，打印 "SHOT IS OFF THE RIM."，然后根据条件打印不同信息，最后根据条件返回
        if (7 / d * Math.random() <= 0.7) {
            print("SHOT IS OFF THE RIM.\n");
            if (Math.random() <= 2.0 / 3.0) {
                print(os + " CONTROLS THE REBOUND.\n");
                return;
            }
            print("DARMOUTH CONTROLS THE REBOUND.\n");
            if (Math.random() <= 0.4)
                continue;
            print("BALL PASSED BACK TO YOU.\n");
            your_turn = 1;
            return;
        }
        # 如果 7 除以 d 乘以一个随机数小于等于 0.875，打印 "SHOOTER FOULED.  TWO SHOTS."，执行 foul 函数，然后返回
        if (7 /d * Math.random() <= 0.875) {
            print("SHOOTER FOULED.  TWO SHOTS.\n");
            foul();
            return;
        }
        # 如果 7 除以 d 乘以一个随机数小于等于 0.925，打印 "SHOT BLOCKED. " + os + "'S BALL."，然后返回
        if (7 /d * Math.random() <= 0.925) {
            print("SHOT BLOCKED. " + os + "'S BALL.\n");
            return;
        }
        # 否则，打印 "CHARGING FOUL.  DARTHMOUTH LOSES THE BALL."，然后返回
        print("CHARGING FOUL.  DARTHMOUTH LOSES THE BALL.\n");
        return;
    }
}
// 定义一个名为 computer_play 的函数
function computer_play()
{
    // 初始化变量 rebound 为 0
    rebound = 0;
    // 结束函数定义
}
// 结束函数定义

// 主程序
async function main()
{
    // 打印标题
    print(tab(31) + "BASKETBALL\n");
    // 打印创意计算的信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏规则
    print("THIS IS DARTMOUTH COLLEGE BASKETBALL.  YOU WILL BE DARTMOUTH\n");
    print(" CAPTAIN AND PLAYMAKER.  CALL SHOTS AS FOLLOWS:  1. LONG\n");
    print(" (30 FT.) JUMP SHOT; 2. SHORT (15 FT.) JUMP SHOT; 3. LAY\n");
    print(" UP; 4. SET SHOT.\n");
    print("BOTH TEAMS WILL USE THE SAME DEFENSE.  CALL DEFENSE AS\n");
    print("FOLLOWS:  6. PRESS; 6.5 MAN-TO MAN; 7. ZONE; 7.5 NONE.\n");
    print("TO CHANGE DEFENSE, JUST TYPE 0 AS YOUR NEXT SHOT.\n");
    print("YOUR STARTING DEFENSE WILL BE");
    // 初始化变量 t 和 p 为 0
    t = 0;
    p = 0;
    // 将输入的字符串转换为浮点数并赋值给变量 d
    d = parseFloat(await input());
    // 如果 d 小于 6
    if (d < 6) {
        // 设置变量 your_turn 为 2
        your_turn = 2;
    } else {
        // 打印提示信息
        print("\n");
        print("CHOOSE YOUR OPPONENT");
        // 将输入的字符串赋值给变量 os
        os = await input();
        // 设置变量 game_restart 为 1
        game_restart = 1;
    }
}
    # 进入游戏循环
    while (1) {
        # 如果需要重新开始游戏
        if (game_restart) {
            # 重置重新开始游戏的标志位
            game_restart = 0;
            # 打印提示信息
            print("CENTER JUMP\n");
            # 根据随机数决定哪方控制开球
            if (Math.random() > 3.0 / 5.0) {
                print("DARMOUTH CONTROLS THE TAP.\n");
            } else {
                print(os + " CONTROLS THE TAP.\n");
                # 让计算机进行操作
                computer_play();
            }
        }
        # 如果轮到你进行第二次进攻
        if (your_turn == 2) {
            # 提示新的防守布局
            print("YOUR NEW DEFENSIVE ALLIGNMENT IS");
            # 获取输入的浮点数
            d = parseFloat(await input());
        }
        # 打印换行符
        print("\n");
        # 进入投篮循环
        while (1) {
            # 提示输入投篮结果
            print("YOUR SHOT");
            # 获取输入的整数
            z = parseInt(await input());
            # 初始化变量 p
            p = 0;
            # 如果输入不是整数或者不在指定范围内，则提示重新输入
            if (z != Math.floor(z) || z < 0 || z > 4)
                print("INCORRECT ANSWER.  RETYPE IT. ");
            else
                break;
        }
        # 根据随机数和条件判断是否重新开始游戏
        if (Math.random() < 0.5 || t < 100) {
            game_restart = 0;
            your_turn = 0;
            # 玩家进行操作
            player_play();
            # 如果游戏没有重新开始且轮到你进行操作，则让计算机进行操作
            if (game_restart == 0 && your_turn == 0)
                computer_play();
        } else {
            # 打印换行符
            print("\n");
            # 如果比分相等
            if (s[1] == s[0]) {
                # 打印提示信息
                print("\n");
                print("   ***** END OF SECOND HALF *****\n");
                print("\n");
                print("SCORE AT END OF REGULATION TIME:\n");
                print("        DARTMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
                print("\n");
                print("BEGIN TWO MINUTE OVERTIME PERIOD\n");
                # 重置时间
                t = 93;
                print("CENTER JUMP\n");
                # 根据随机数决定哪方控制开球
                if (Math.random() > 3.0 / 5.0)
                    print("DARMOUTH CONTROLS THE TAP.\n");
                else
                    print(os + " CONTROLS THE TAP.\n");
            } else {
                # 打印提示信息
                print("   ***** END OF GAME *****\n");
                print("FINAL SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
                # 结束游戏循环
                break;
            }
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```