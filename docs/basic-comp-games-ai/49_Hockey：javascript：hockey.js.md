# `d:/src/tocomm/basic-computer-games\49_Hockey\javascript\hockey.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上显示提示符 "? "
// 设置 INPUT 元素的类型为文本输入
                       // 设置输入框的长度为50
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到 id 为 "output" 的元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       // 初始化输入字符串为 undefined
                       input_str = undefined;
                       // 添加键盘按下事件监听器
                       input_element.addEventListener("keydown", function (event) {
                           // 如果按下的是回车键
                           if (event.keyCode == 13) {
                               // 将输入框中的值赋给 input_str
                               input_str = input_element.value;
                               // 移除输入框
                               document.getElementById("output").removeChild(input_element);
                               // 打印输入的字符串
                               print(input_str);
                               // 打印换行符
                               print("\n");
                               // 返回输入的字符串
                               resolve(input_str);
                           }
                       });
                   });
}

function tab(space)
{
    var str = "";
    // 循环 space 次，每次将空格添加到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

var as = [];  // 声明一个空数组as
var bs = [];  // 声明一个空数组bs
var ha = [];  // 声明一个空数组ha
var ta = [];  // 声明一个空数组ta
var t1 = [];  // 声明一个空数组t1
var t2 = [];  // 声明一个空数组t2
var t3 = [];  // 声明一个空数组t3

// Main program
async function main()
{
    print(tab(33) + "HOCKEY\n");  // 在控制台打印带有33个空格的字符串和"HOCKEY"字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在控制台打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"字符串
    print("\n");  // 在控制台打印一个空行
    print("\n");  // 在控制台打印一个空行
    print("\n");  // 在控制台打印一个空行
    // 初始化数组 ha，将其所有元素都设为 0
    for (c = 0; c <= 20; c++)
        ha[c] = 0;
    // 初始化数组 ta, t1, t2, t3，将它们所有元素都设为 0
    for (c = 1; c <= 5; c++) {
        ta[c] = 0;
        t1[c] = 0;
        t2[c] = 0;
        t3[c] = 0;
    }
    // 初始化变量 x 为 1
    x = 1;
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    // 进入无限循环
    while (1) {
        // 打印提示信息
        print("WOULD YOU LIKE THE INSTRUCTIONS");
        // 等待用户输入
        str = await input();
        // 打印一个空行
        print("\n");
        // 如果用户输入为 "YES" 或 "NO"，则跳出循环
        if (str == "YES" || str == "NO")
            break;
        // 如果用户输入不是 "YES" 或 "NO"，则打印提示信息并继续循环
        print("ANSWER YES OR NO!!\n");
    # 如果字符串等于"YES"，则执行以下操作
    if (str == "YES"):
        # 打印空行
        print("\n")
        # 打印提示信息
        print("THIS IS A SIMULATED HOCKEY GAME.\n")
        # 打印问题和回答的表头
        print("QUESTION     RESPONSE\n")
        # 打印提示信息
        print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD\n")
        # 打印提示信息
        print("             LIKE TO MAKE, FROM 0 TO 3.\n")
        # 打印提示信息
        print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT\n")
        # 打印提示信息
        print("             YOU WANT TO MAKE.  ENTER:\n")
        # 打印提示信息
        print("             1 FOR A SLAPSHOT\n")
        # 打印提示信息
        print("             2 FOR A WRISTSHOT\n")
        # 打印提示信息
        print("             3 FOR A BACKHAND\n")
        # 打印提示信息
        print("             4 FOR A SNAP SHOT\n")
        # 打印提示信息
        print("AREA         TYPE IN THE NUMBER CORRESPONDING TO\n")
        # 打印提示信息
        print("             THE AREA YOU ARE AIMING AT.  ENTER:\n")
        # 打印提示信息
        print("             1 FOR UPPER LEFT HAND CORNER\n")
        # 打印提示信息
        print("             2 FOR UPPER RIGHT HAND CORNER\n")
        # 打印提示信息
        print("             3 FOR LOWER LEFT HAND CORNER\n")
        # 打印提示信息
        print("             4 FOR LOWER RIGHT HAND CORNER\n")
        # 打印空行
        print("\n")
        # 打印游戏开始时的提示信息
        print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES\n");
        print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: \n");
        print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,\n");
        print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL\n");
        print("HAVE EXPLANATORY INSTRUCTIONS.\n");
    }
    # 打印提示信息，要求输入两支球队的名称
    print("ENTER THE TWO TEAMS");
    # 获取用户输入的字符串
    str = await input();
    # 查找逗号的位置
    c = str.indexOf(",");
    # 将逗号前的部分赋值给as[7]
    as[7] = str.substr(0, c);
    # 将逗号后的部分赋值给bs[7]
    bs[7] = str.substr(c + 1);
    # 打印换行
    print("\n");
    # 循环，要求输入比赛的分钟数，直到输入大于等于1的数字为止
    do {
        print("ENTER THE NUMBER OF MINUTES IN A GAME");
        # 将用户输入的字符串转换为整数
        t6 = parseInt(await input());
        print("\n");
    } while (t6 < 1) ;
    # 打印换行
    print("\n");
    # 打印提示信息，询问是否"as[7]"教练要输入他的球队
    print("WOULD THE " + as[7] + " COACH ENTER HIS TEAM\n");
    # 打印换行
    print("\n");
    # 循环6次，提示用户输入球员信息并存储到as列表中
    for (i = 1; i <= 6; i++) {
        print("PLAYER " + i + " ");
        as[i] = await input();
    }
    # 打印换行
    print("\n");
    # 打印提示信息
    print("WOULD THE " + bs[7] + " COACH DO THE SAME\n");
    # 打印换行
    print("\n");
    # 循环6次，提示用户输入球员信息并存储到bs列表中
    for (t = 1; t <= 6; t++) {
        print("PLAYER " + t + " ");
        bs[t] = await input();
    }
    # 打印换行
    print("\n");
    # 提示用户输入裁判信息并存储到rs变量中
    print("INPUT THE REFEREE FOR THIS GAME");
    rs = await input();
    # 打印换行
    print("\n");
    # 打印球队阵容信息
    print(tab(10) + as[7] + " STARTING LINEUP\n");
    # 循环6次，打印球员信息
    for (t = 1; t <= 6; t++) {
        print(as[t] + "\n");
    }
    # 打印换行
    print("\n");
    # 打印固定长度的空格和 bs 列表中索引为 7 的元素，再加上 " STARTING LINEUP" 字符串
    print(tab(10) + bs[7] + " STARTING LINEUP\n");
    # 循环打印 bs 列表中索引为 1 到 6 的元素
    for (t = 1; t <= 6; t++) {
        print(bs[t] + "\n");
    }
    # 打印换行符
    print("\n");
    # 打印 "WE'RE READY FOR TONIGHTS OPENING FACE-OFF." 字符串
    print("WE'RE READY FOR TONIGHTS OPENING FACE-OFF.\n");
    # 打印 rs 变量的值，再加上 " WILL DROP THE PUCK BETWEEN "、as 列表中索引为 2 的元素和 bs 列表中索引为 2 的元素
    print(rs + " WILL DROP THE PUCK BETWEEN " + as[2] + " AND " + bs[2] + "\n");
    # 初始化 s2 和 s3 变量的值为 0
    s2 = 0;
    s3 = 0;
    # 循环，l 从 1 到 t6
    for (l = 1; l <= t6; l++) {
        # 生成一个随机数 c，值为 1 或 2
        c = Math.floor(2 * Math.random()) + 1;
        # 如果 c 的值为 1，则打印 as 列表中索引为 7 的元素和 " HAS CONTROL OF THE PUCK" 字符串；否则打印 bs 列表中索引为 7 的元素和 " HAS CONTROL." 字符串
        if (c == 1)
            print(as[7] + " HAS CONTROL OF THE PUCK\n");
        else
            print(bs[7] + " HAS CONTROL.\n");
        # 进入循环
        do {
            # 打印 "PASS" 字符串
            print("PASS");
            # 将用户输入的值转换为整数并赋给 p 变量
            p = parseInt(await input());
            # 循环，n 从 1 到 3
            for (n = 1; n <= 3; n++)
                ha[n] = 0;  // 将数组 ha 的第 n 个元素赋值为 0
        } while (p < 0 || p > 3) ;  // 当 p 小于 0 或者大于 3 时，执行循环
        do {
            for (j = 1; j <= p + 2; j++)  // 循环执行 p+2 次
                ha[j] = Math.floor(5 * Math.random()) + 1;  // 将 ha 数组的第 j 个元素赋值为 1 到 5 之间的随机整数
        } while (ha[j - 1] == ha[j - 2] || (p + 2 >= 3 && (ha[j - 1] == ha[j - 3] || ha[j - 2] == ha[j - 3]))) ;  // 当 ha[j-1] 等于 ha[j-2] 或者 (p+2 大于等于 3 并且 (ha[j-1] 等于 ha[j-3] 或者 ha[j-2] 等于 ha[j-3])) 时，执行循环
        if (p == 0) {  // 如果 p 等于 0
            while (1) {  // 无限循环
                print("SHOT");  // 打印 "SHOT"
                s = parseInt(await input());  // 将输入的值转换为整数并赋值给 s
                if (s >= 1 && s <= 4)  // 如果 s 大于等于 1 并且小于等于 4
                    break;  // 退出循环
            }
            if (c == 1) {  // 如果 c 等于 1
                print(as[ha[j - 1]]);  // 打印 as 数组中 ha[j-1] 对应的值
                g = ha[j - 1];  // 将 ha[j-1] 的值赋给 g
                g1 = 0;  // 将 g1 的值赋为 0
                g2 = 0;  // 将 g2 的值赋为 0
            } else {
                print(bs[ha[j - 1]]);  // 打印 bs 数组中 ha[j-1] 对应的值
                g2 = 0;  // 初始化变量 g2 为 0
                g2 = 0;  // 再次将变量 g2 设置为 0，上一行代码可能是多余的
                g = ha[j - 1];  // 从数组 ha 中取出索引为 j-1 的元素赋值给变量 g
            }
            switch (s) {  // 开始一个 switch 语句，根据变量 s 的值进行不同的操作
                case 1:  // 当变量 s 的值为 1 时
                    print(" LET'S A BOOMER GO FROM THE RED LINE!!\n");  // 打印字符串 " LET'S A BOOMER GO FROM THE RED LINE!!\n"
                    z = 10;  // 将变量 z 的值设置为 10
                    break;  // 结束 case 1
                case 2:  // 当变量 s 的值为 2 时
                    print(" FLIPS A WRISTSHOT DOWN THE ICE\n");  // 打印字符串 " FLIPS A WRISTSHOT DOWN THE ICE\n"
                    // 可能是原始代码中缺少了第 430 行
                case 3:  // 当变量 s 的值为 3 时
                    print(" BACKHANDS ONE IN ON THE GOALTENDER\n");  // 打印字符串 " BACKHANDS ONE IN ON THE GOALTENDER\n"
                    z = 25;  // 将变量 z 的值设置为 25
                    break;  // 结束 case 3
                case 4:  // 当变量 s 的值为 4 时
                    print(" SNAPS A LONG FLIP SHOT\n");  // 打印字符串 " SNAPS A LONG FLIP SHOT\n"
                    z = 17;  // 将变量 z 的值设置为 17
                    break;  // 结束 case 4
            }
        } else {
            if (c == 1) {  # 如果c等于1
                switch (p) {  # 根据p的值进行不同的操作
                    case 1:  # 如果p等于1
                        print(as[ha[j - 2]] + " LEADS " + as[ha[j - 1]] + " WITH A PERFECT PASS.\n");  # 打印特定格式的字符串
                        print(as[ha[j - 1]] + " CUTTING IN!!!\n");  # 打印特定格式的字符串
                        g = ha[j - 1];  # 将g的值设置为ha[j - 1]
                        g1 = ha[j - 2];  # 将g1的值设置为ha[j - 2]
                        g2 = 0;  # 将g2的值设置为0
                        z1 = 3;  # 将z1的值设置为3
                        break;  # 跳出switch语句
                    case 2:  # 如果p等于2
                        print(as[ha[j - 2]] + " GIVES TO A STREAKING " + as[ha[j - 1]] + "\n");  # 打印特定格式的字符串
                        print(as[ha[j - 3]] + " COMES DOWN ON " + bs[5] + " AND " + bs[4] + "\n");  # 打印特定格式的字符串
                        g = ha[j - 3];  # 将g的值设置为ha[j - 3]
                        g1 = ha[j - 1];  # 将g1的值设置为ha[j - 1]
                        g2 = ha[j - 2];  # 将g2的值设置为ha[j - 2]
                        z1 = 2;  # 将z1的值设置为2
                        break;  # 跳出switch语句
                    case 3:  # 如果情况为3
                        print("OH MY GOD!! A ' 4 ON 2 ' SITUATION\n");  # 打印字符串
                        print(as[ha[j - 3]] + " LEADS " + as[ha[j - 2]] + "\n");  # 打印字符串和变量
                        print(as[ha[j - 2]] + " IS WHEELING THROUGH CENTER.\n");  # 打印字符串和变量
                        print(as[ha[j - 2]] + " GIVES AND GOEST WITH " + as[ha[j - 1]] + "\n");  # 打印字符串和变量
                        print("PRETTY PASSING!\n");  # 打印字符串
                        print(as[ha[j - 1]] + " DROPS IT TO " + as[ha[j - 4]] + "\n");  # 打印字符串和变量
                        g = ha[j - 4];  # 将变量赋值给g
                        g1 = ha[j - 1];  # 将变量赋值给g1
                        g2 = ha[j - 2];  # 将变量赋值给g2
                        z1 = 1;  # 将1赋值给z1
                        break;  # 跳出switch语句
                }
            } else {
                switch (p) {  # 开始一个新的switch语句
                    case 1:  # 如果情况为1
                        print(bs[ha[j - 1]] + " HITS " + bs[ha[j - 2]] + " FLYING DOWN THE LEFT SIDE\n");  # 打印字符串和变量
                        g = ha[j - 2];  # 将变量赋值给g
                        g1 = ha[j - 1];  # 将变量赋值给g1
                        g2 = 0;  # 将0赋值给g2
                        z1 = 3;  # 将变量 z1 赋值为 3
                        break;  # 跳出 switch 语句
                    case 2:  # 如果 switch 表达式的值等于 2
                        print("IT'S A ' 3 ON 2 '!\n");  # 打印字符串 "IT'S A ' 3 ON 2 '!"
                        print("ONLY " + as[4] + " AND " + as[5] + " ARE BACK.\n");  # 打印特定位置的数组元素
                        print(bs[ha[j - 2]] + " GIVES OFF TO " + bs[ha[j - 1]] + "\n");  # 打印特定位置的数组元素
                        print(bs[ha[j - 1]] + " DROPS TO " + bs[ha[j - 3]] + "\n");  # 打印特定位置的数组元素
                        g = ha[j - 3];  # 将 g 赋值为特定位置的数组元素
                        g1 = ha[j - 1];  # 将 g1 赋值为特定位置的数组元素
                        g2 = ha[j - 2];  # 将 g2 赋值为特定位置的数组元素
                        z1 = 2;  # 将变量 z1 赋值为 2
                        break;  # 跳出 switch 语句
                    case 3:  # 如果 switch 表达式的值等于 3
                        print(" A '3 ON 2 ' WITH A ' TRAILER '!\n");  # 打印字符串 " A '3 ON 2 ' WITH A ' TRAILER '!"
                        print(bs[ha[j - 4]] + " GIVES TO " + bs[ha[j - 2]] + " WHO SHUFFLES IT OFF TO\n");  # 打印特定位置的数组元素
                        print(bs[ha[j - 1]] + " WHO FIRES A WING TO WING PASS TO \n");  # 打印特定位置的数组元素
                        print(bs[ha[j - 3]] + " AS HE CUTS IN ALONE!!\n");  # 打印特定位置的数组元素
                        g = ha[j - 3];  # 将 g 赋值为特定位置的数组元素
                        g1 = ha[j - 1];  # 将 g1 赋值为特定位置的数组元素
                        g2 = ha[j - 2];  # 将 g2 赋值为特定位置的数组元素
                        z1 = 1;  // 初始化变量z1为1
                        break;   // 跳出当前循环
                }
            }
            do {
                print("SHOT");  // 打印"SHOT"
                s = parseInt(await input());  // 从输入中获取一个整数并赋值给变量s
            } while (s < 1 || s > 4) ;  // 当s小于1或大于4时重复执行循环
            if (c == 1)
                print(as[g]);  // 如果c等于1，则打印as[g]
            else
                print(bs[g]);  // 否则打印bs[g]
            switch (s) {  // 根据s的值进行不同的操作
                case 1:
                    print(" LET'S A BIG SLAP SHOT GO!!\n");  // 如果s等于1，则打印" LET'S A BIG SLAP SHOT GO!!\n"
                    z = 4;  // 变量z赋值为4
                    z += z1;  // z加上z1的值
                    break;  // 跳出当前switch语句
                case 2:
                    print(" RIPS A WRIST SHOT OFF\n");  // 如果s等于2，则打印" RIPS A WRIST SHOT OFF\n"
                    z = 2;  // 初始化变量 z 为 2
                    z += z1;  // 将 z1 的值加到 z 上
                    break;  // 跳出 switch 语句
                case 3:  // 如果 c 的值为 3
                    print(" GETS A BACKHAND OFF\n");  // 打印信息
                    z = 3;  // 将 z 的值设为 3
                    z += z1;  // 将 z1 的值加到 z 上
                    break;  // 跳出 switch 语句
                case 4:  // 如果 c 的值为 4
                    print(" SNAPS OFF A SNAP SHOT\n");  // 打印信息
                    z = 2;  // 将 z 的值设为 2
                    z += z1;  // 将 z1 的值加到 z 上
                    break;  // 跳出 switch 语句
            }
        }
        do {
            print("AREA");  // 打印信息
            a = parseInt(await input());  // 从输入中获取一个整数并赋值给变量 a
        } while (a < 1 || a > 4) ;  // 当 a 的值小于 1 或大于 4 时重复执行 do 里面的代码
        if (c == 1)  // 如果 c 的值为 1
            s2++;  // 增加变量 s2 的值
        else
            s3++;  // 增加变量 s3 的值
        a1 = Math.floor(4 * Math.random()) + 1;  // 生成一个 1 到 4 之间的随机整数赋值给变量 a1
        if (a == a1) {  // 如果变量 a 的值等于 a1 的值
            while (1) {  // 进入一个无限循环
                ha[20] = Math.floor(100 * Math.random()) + 1;  // 生成一个 1 到 100 之间的随机整数赋值给数组 ha 的第 21 个元素
                if (ha[20] % z != 0)  // 如果数组 ha 的第 21 个元素除以 z 的余数不等于 0
                    break;  // 退出循环
                a2 = Math.floor(100 * Math.random()) + 1;  // 生成一个 1 到 100 之间的随机整数赋值给变量 a2
                if (a2 % 4 == 0) {  // 如果变量 a2 能被 4 整除
                    if (c == 1)  // 如果变量 c 的值等于 1
                        print("SAVE " + bs[6] + " --  REBOUND\n");  // 打印字符串 "SAVE " 加上数组 bs 的第 7 个元素的值，再加上字符串 " --  REBOUND"
                    else
                        print("SAVE " + as[6] + " --  FOLLOW up\n");  // 打印字符串 "SAVE " 加上数组 as 的第 7 个元素的值，再加上字符串 " --  FOLLOW up"
                    continue;  // 继续下一次循环
                } else {
                    a1 = a + 1;  // 将变量 a 的值加 1 赋值给变量 a1，使得 a 不等于 a1
                }
            }
            if (ha[20] % z != 0):  # 如果 ha[20] 除以 z 的余数不等于 0
                if (c == 1):  # 如果 c 等于 1
                    print("GOAL " + as[7] + "\n")  # 打印 "GOAL " 和 as[7] 的值，并换行
                    ha[9]++  # ha[9] 自增 1
                else:  # 否则
                    print("SCORE " + bs[7] + "\n")  # 打印 "SCORE " 和 bs[7] 的值，并换行
                    ha[8]++  # ha[8] 自增 1
                # Bells in origninal
                print("\n")  # 打印一个空行
                print("SCORE: ")  # 打印 "SCORE: "
                if (ha[8] <= ha[9]):  # 如果 ha[8] 小于等于 ha[9]
                    print(as[7] + ": " + ha[9] + "\t" + bs[7] + ": " + ha[8] + "\n")  # 打印 as[7]、ha[9]、bs[7]、ha[8] 的值，并换行
                else:  # 否则
                    print(bs[7] + ": " + ha[8] + "\t" + as[7] + ": " + ha[9] + "\n")  # 打印 bs[7]、ha[8]、as[7]、ha[9] 的值，并换行
                if (c == 1):  # 如果 c 等于 1
                    print("GOAL SCORED BY: " + as[g] + "\n")  # 打印 "GOAL SCORED BY: " 和 as[g] 的值，并换行
                    if (g1 != 0):  # 如果 g1 不等于 0
                        if (g2 != 0):  # 如果 g2 不等于 0
                    // 打印球员助攻信息
                    print(" ASSISTED BY: " + as[g1] + " AND " + as[g2] + "\n");
                } else {
                    // 打印单个球员助攻信息
                    print(" ASSISTED BY: " + as[g1] + "\n");
                }
            } else {
                // 打印未助攻信息
                print(" UNASSISTED.\n");
            }
            // 更新球队总进球数
            ta[g]++;
            // 更新球员1的进球数
            t1[g1]++;
            // 更新球员2的进球数
            t1[g2]++;
            // 1540
        } else {
            // 打印进球球员信息
            print("GOAL SCORED BY: " + bs[g] + "\n");
            if (g1 != 0) {
                if (g2 != 0) {
                    // 打印球员助攻信息
                    print(" ASSISTED BY: " + bs[g1] + " AND " + bs[g2] + "\n");
                } else {
                    // 打印单个球员助攻信息
                    print(" ASSISTED BY: " + bs[g1] + "\n");
                }
            } else {
                    print(" UNASSISTED.\n");  // 打印 UNASSISTED.\n
                }
                t2[g]++;  // t2 数组中索引为 g 的元素加一
                t3[g1]++;  // t3 数组中索引为 g1 的元素加一
                t3[g2]++;  // t3 数组中索引为 g2 的元素加一
                // 1540  // 注释：这里是一个占位注释，没有实际作用
            }
        }
    }
    if (a != a1) {  // 如果 a 不等于 a1
        s1 = Math.floor(6 * Math.random()) + 1;  // 生成一个 1 到 6 之间的随机整数
        if (c == 1) {  // 如果 c 等于 1
            switch (s1) {  // 根据 s1 的值进行判断
                case 1:
                    print("KICK SAVE AND A BEAUTY BY " + bs[6] + "\n");  // 打印 "KICK SAVE AND A BEAUTY BY " 后面跟着 bs 数组中索引为 6 的元素
                    print("CLEARED OUT BY " + bs[3] + "\n");  // 打印 "CLEARED OUT BY " 后面跟着 bs 数组中索引为 3 的元素
                    l--;  // l 减一
                    continue;  // 继续下一次循环
                case 2:
                    print("WHAT A SPECTACULAR GLOVE SAVE BY " + bs[6] + "\n");  // 打印 "WHAT A SPECTACULAR GLOVE SAVE BY " 后面跟着 bs 数组中索引为 6 的元素
                    case 0:
                        # 打印出球员 bs[6] 的进球信息
                        print("AND " + bs[6] + " GOLFS IT INTO THE CROWD\n");
                        break;
                    case 3:
                        # 打印出球员 bs[6] 的低位射门被挽救的信息
                        print("SKATE SAVE ON A LOW STEAMER BY " + bs[6] + "\n");
                        l--;
                        continue;
                    case 4:
                        # 打印出球员 bs[6] 的射门被挡出并由球员 as[g] 接住的信息
                        print("PAD SAVE BY " + bs[6] + " OFF THE STICK\n");
                        print("OF " + as[g] + " AND " + bs[6] + " COVERS UP\n");
                        break;
                    case 5:
                        # 打印出球员 bs[6] 的射门越过了球门的信息
                        print("WHISTLES ONE OVER THE HEAD OF " + bs[6] + "\n");
                        l--;
                        continue;
                    case 6:
                        # 打印出球员 bs[6] 的面部挽救并受伤的信息，以及由球员 bs[5] 接替的信息
                        print(bs[6] + " MAKES A FACE SAVE!! AND HE IS HURT\n");
                        print("THE DEFENSEMAN " + bs[5] + " COVERS UP FOR HIM\n");
                        break;
                }
            } else {
# 开始一个 switch 语句，根据 s1 的值执行不同的操作
                switch (s1) {
                    # 如果 s1 的值为 1，则打印 "STICK SAVE BY " 和 as[6] 的值，并换行
                    case 1:
                        print("STICK SAVE BY " + as[6] +"\n");
                        # 打印 "AND CLEARED OUT BY " 和 as[4] 的值，并换行
                        print("AND CLEARED OUT BY " + as[4] + "\n");
                        # l 减一
                        l--;
                        # 继续下一次循环
                        continue;
                    # 如果 s1 的值为 2，则打印 "OH MY GOD!! " 和 bs[g] 的值，以及其他相关信息
                    case 2:
                        print("OH MY GOD!! " + bs[g] + " RATTLES ONE OFF THE POST\n");
                        print("TO THE RIGHT OF " + as[6] + " AND " + as[6] + " COVERS ");
                        print("ON THE LOOSE PUCK!\n");
                        # 跳出 switch 语句
                        break;
                    # 如果 s1 的值为 3，则打印 "SKATE SAVE BY " 和 as[6] 的值，并换行
                    case 3:
                        print("SKATE SAVE BY " + as[6] + "\n");
                        # 打印 as[6] 的值和其他相关信息
                        print(as[6] + " WHACKS THE LOOSE PUCK INTO THE STANDS\n");
                        # 跳出 switch 语句
                        break;
                    # 如果 s1 的值为 4，则打印 "STICK SAVE BY " 和 as[6] 的值，并其他相关信息
                    case 4:
                        print("STICK SAVE BY " + as[6] + " AND HE CLEARS IT OUT HIMSELF\n");
                        # l 减一
                        l--;
                        # 继续下一次循环
                        continue;
                    # 如果 s1 的值为 5，则执行其他操作
                        # 打印被第6个元素踢出的信息
                        print("KICKED OUT BY " + as[6] + "\n");
                        # 打印信息并且减少l的值
                        print("AND IT REBOUNDS ALL THE WAY TO CENTER ICE\n");
                        l--;
                        # 继续循环
                        continue;
                    case 6:
                        # 打印第6个元素的信息
                        print("GLOVE SAVE " + as[6] + " AND HE HANGS ON\n");
                        # 结束switch语句
                        break;
                }
            }
        }
        # 打印信息
        print("AND WE'RE READY FOR THE FACE-OFF\n");
    }
    // Bells chime
    # 打印信息
    print("THAT'S THE SIREN\n");
    # 打印空行
    print("\n");
    # 打印最终比分
    print(tab(15) + "FINAL SCORE:\n");
    # 如果ha[8]小于等于ha[9]，打印as[7]、ha[9]、bs[7]、ha[8]的信息
    if (ha[8] <= ha[9]) {
        print(as[7] + ": " + ha[9] + "\t" + bs[7] + ": " + ha[8] + "\n");
    } else {
        # 否则打印bs[7]、ha[8]、as[7]、ha[9]的信息
        print(bs[7] + ": " + ha[8] + "\t" + as[7] + ": " + ha[9] + "\n");
    }
    # 打印换行
    print("\n");
    # 打印制表符和"SCORING SUMMARY"字符串
    print(tab(10) + "SCORING SUMMARY\n");
    # 打印换行
    print("\n");
    # 打印制表符和as列表中索引为7的元素
    print(tab(25) + as[7] + "\n");
    # 打印表头"NAME", "GOALS", "ASSISTS"
    print("\tNAME\tGOALS\tASSISTS\n");
    # 打印表格分隔线
    print("\t----\t-----\t-------\n");
    # 遍历1到5的数字，打印球员名字、进球数和助攻数
    for (i = 1; i <= 5; i++) {
        print("\t" + as[i] + "\t" + ta[i] + "\t" + t1[i] + "\n");
    }
    # 打印换行
    print("\n");
    # 打印制表符和bs列表中索引为7的元素
    print(tab(25) + bs[7] + "\n");
    # 打印表头"NAME", "GOALS", "ASSISTS"
    print("\tNAME\tGOALS\tASSISTS\n");
    # 打印表格分隔线
    print("\t----\t-----\t-------\n");
    # 遍历1到5的数字，打印球员名字、进球数和助攻数
    for (t = 1; t <= 5; t++) {
        print("\t" + bs[t] + "\t" + t2[t] + "\t" + t3[t] + "\n");
    }
    # 打印换行
    print("\n");
    # 打印"SHOTS ON NET"字符串
    print("SHOTS ON NET\n");
    # 打印as列表中索引为7的元素和s2变量的值
    print(as[7] + ": " + s2 + "\n");
    print(bs[7] + ": " + s3 + "\n");  # 打印 bs 列表中索引为 7 的元素和字符串 s3 的组合，然后换行
}

main();  # 调用 main 函数
```