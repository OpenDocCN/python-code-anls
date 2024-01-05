# `37_Football\javascript\football.js`

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
# 添加键盘按下事件监听器，当按下回车键时，获取输入字符串，移除输入元素，打印输入字符串，换行，解析输入字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束事件监听器
});
}

# 定义一个函数 tab，参数为 space
function tab(space)
{
    # 初始化一个空字符串 str
    var str = "";
    # 当 space 大于 0 时，循环执行以下操作
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var player_data = [17,8,4,14,19,3,10,1,7,11,15,9,5,20,13,18,16,2,12,6,  # 定义包含球员数据的数组
                   20,2,17,5,8,18,12,11,1,4,19,14,10,7,9,15,6,13,16,3];
var aa = [];  # 定义空数组
var ba = [];  # 定义空数组
var ca = [];  # 定义空数组
var ha = [];  # 定义空数组
var ta = [];  # 定义空数组
var wa = [];  # 定义空数组
var xa = [];  # 定义空数组
var ya = [];  # 定义空数组
var za = [];  # 定义空数组
var ms = [];  # 定义空数组
var da = [];  # 定义空数组
var ps = [, "PITCHOUT","TRIPLE REVERSE","DRAW","QB SNEAK","END AROUND",  # 定义包含不同类型传球策略的数组
          "DOUBLE REVERSE","LEFT SWEEP","RIGHT SWEEP","OFF TACKLE",
          "WISHBONE OPTION","FLARE PASS","SCREEN PASS",
# 定义一个包含不同足球战术的数组
var tactics = ["ROLL OUT OPTION","RIGHT CURL","LEFT CURL","WISHBONE OPTION",
          "SIDELINE PASS","HALF-BACK OPTION","RAZZLE-DAZZLE","BOMB!!!!"];
# 定义变量 p 和 t
var p;
var t;

# 定义函数 field_headers，用于打印足球场上的队伍头部信息
function field_headers()
{
    # 打印队伍头部的距离标记
    print("TEAM 1 [0   10   20   30   40   50   60   70   80   90");
    # 打印队伍头部的分隔线
    print("   100] TEAM 2\n");
    # 打印空行
    print("\n");
}

# 定义函数 separator，用于打印分隔线
function separator()
{
    # 初始化一个空字符串
    str = "";
    # 循环生成分隔线
    for (x = 1; x <= 72; x++)
        str += "+";
    # 打印分隔线
    print(str + "\n");
}
# 定义名为 show_ball 的函数，用于显示球的信息
function show_ball()
{
    # 打印球的位置、速度和加速度
    print(tab(da[t] + 5 + p / 2) + ms[t] + "\n");
    # 调用 field_headers 函数
    field_headers();
}

# 定义名为 show_scores 的函数，用于显示比分信息
function show_scores()
{
    # 打印空行
    print("\n");
    # 打印第一队的得分
    print("TEAM 1 SCORE IS " + ha[1] + "\n");
    # 打印第二队的得分
    print("TEAM 2 SCORE IS " + ha[2] + "\n");
    # 打印空行
    print("\n");
    # 如果某队得分大于等于设定的胜利分数
    if (ha[t] >= e) {
        # 打印该队获胜的信息
        print("TEAM " + t + " WINS*******************");
        # 返回 true
        return true;
    }
    # 返回 false
    return false;
}

# 定义名为 loss_posession 的函数，用于处理失去球权的情况
    print("\n");  # 打印空行
    print("** LOSS OF POSSESSION FROM TEAM " + t + " TO TEAM " + ta[t] + "\n");  # 打印失去球权的消息，包括球队信息
    print("\n");  # 打印空行
    separator();  # 调用名为separator的函数，可能是用来打印分隔线
    print("\n");  # 打印空行
    t = ta[t];  # 将变量t的值更新为ta[t]的值

function touchdown() {
    print("\n");  # 打印空行
    print("TOUCHDOWN BY TEAM " + t + " *********************YEA TEAM\n");  # 打印Touchdown的消息，包括球队信息
    q = 7;  # 将变量q的值设置为7
    g = Math.random();  # 生成一个0到1之间的随机数，赋值给变量g
    if (g <= 0.1) {  # 如果随机数小于等于0.1
        q = 6;  # 将变量q的值设置为6
        print("EXTRA POINT NO GOOD\n");  # 打印额外得分未成功的消息
    } else {
        print("EXTRA POINT GOOD\n");  # 打印额外得分成功的消息
    }
    ha[t] = ha[t] + q;  # 更新ha[t]的值，加上变量q的值
// 主程序
async function main()
{
    // 打印标题
    print(tab(32) + "FOOTBALL\n");
    // 打印创意计算的地址
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 打印游戏介绍
    print("PRESENTING N.F.U. FOOTBALL (NO FORTRAN USED)\n");
    print("\n");
    print("\n");
    // 循环直到用户输入有效指令
    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        // 等待用户输入
        str = await input();
        // 如果用户输入为"YES"或"NO"，跳出循环
        if (str == "YES" || str == "NO")
            break;
    }
    // 如果用户输入为"YES"，执行以下代码
    if (str == "YES") {
        # 打印游戏规则说明
        print("THIS IS A FOOTBALL GAME FOR TWO TEAMS IN WHICH PLAYERS MUST\n");
        print("PREPARE A TAPE WITH A DATA STATEMENT (1770 FOR TEAM 1,\n");
        print( "1780 FOR TEAM 2) IN WHICH EACH TEAM SCRAMBLES NOS. 1-20\n");
        print("THESE NUMBERS ARE THEN ASSIGNED TO TWENTY GIVEN PLAYS.\n");
        print("A LIST OF NOS. AND THEIR PLAYS IS PROVIDED WITH\n");
        print("BOTH TEAMS HAVING THE SAME PLAYS. THE MORE SIMILAR THE\n");
        print("PLAYS THE LESS YARDAGE GAINED.  SCORES ARE GIVEN\n");
        print("WHENEVER SCORES ARE MADE. SCORES MAY ALSO BE OBTAINED\n");
        print("BY INPUTTING 99,99 FOR PLAY NOS. TO PUNT OR ATTEMPT A\n");
        print("FIELD GOAL, INPUT 77,77 FOR PLAY NUMBERS. QUESTIONS WILL BE\n");
        print("ASKED THEN. ON 4TH DOWN, YOU WILL ALSO BE ASKED WHETHER\n");
        print("YOU WANT TO PUNT OR ATTEMPT A FIELD GOAL. IF THE ANSWER TO\n");
        print("BOTH QUESTIONS IS NO IT WILL BE ASSUMED YOU WANT TO\n");
        print("TRY AND GAIN YARDAGE. ANSWER ALL QUESTIONS YES OR NO.\n");
        print("THE GAME IS PLAYED UNTIL PLAYERS TERMINATE (CONTROL-C).\n");
        print("PLEASE PREPARE A TAPE AND RUN.\n");
    }
    # 打印空行
    print("\n");
    # 提示用户输入游戏的得分上限
    print("PLEASE INPUT SCORE LIMIT ON GAME");
    # 将用户输入的值转换为整数并赋值给变量 e
    e = parseInt(await input());
    # 使用循环将 player_data 中的数据存入不同的字典中
    for (i = 1; i <= 40; i++) {
        # 如果 i 小于等于 20，则将 player_data[i - 1] 对应的值存入 aa 字典中，键为 i
        if (i <= 20) {
            aa[player_data[i - 1]] = i;
        } 
        # 如果 i 大于 20，则将 player_data[i - 1] 对应的值存入 ba 字典中，键为 i - 20
        else {
            ba[player_data[i - 1]] = i - 20;
        }
        # 将 player_data[i - 1] 存入 ca 字典中，键为 i
        ca[i] = player_data[i - 1];
    }
    # 初始化变量 l 和 t
    l = 0;
    t = 1;
    # 执行循环
    do {
        # 打印输出团队 t 的比赛图表
        print("TEAM " + t + " PLAY CHART\n");
        # 打印输出表头
        print("NO.      PLAY\n");
        # 使用循环打印输出每个球员的比赛数据
        for (i = 1; i <= 20; i++) {
            # 将 ca[i + l] 转换为字符串
            str = "" + ca[i + l];
            # 如果字符串长度小于 6，则在末尾添加空格，使其长度为 6
            while (str.length < 6)
                str += " ";
            # 将 ps[i] 添加到字符串末尾
            str += ps[i];
            # 打印输出字符串
            print(str + "\n");
        }
        l += 20;  # 将变量 l 的值增加 20
        t = 2;  # 将变量 t 的值设为 2
        print("\n");  # 打印一个换行符
        print("TEAR OFF HERE----------------------------------------------\n");  # 打印指定的字符串
        for (x = 1; x <= 11; x++)  # 循环 11 次
            print("\n");  # 每次循环打印一个换行符
    } while (l == 20) ;  # 当 l 的值等于 20 时继续循环
    da[1] = 0;  # 将数组 da 的第一个元素设为 0
    da[2] = 3;  # 将数组 da 的第二个元素设为 3
    ms[1] = "--->";  # 将数组 ms 的第一个元素设为 "--->"
    ms[2] = "<---";  # 将数组 ms 的第二个元素设为 "<---"
    ha[1] = 0;  # 将数组 ha 的第一个元素设为 0
    ha[2] = 0;  # 将数组 ha 的第二个元素设为 0
    ta[1] = 2;  # 将数组 ta 的第一个元素设为 2
    ta[2] = 1;  # 将数组 ta 的第二个元素设为 1
    wa[1] = -1;  # 将数组 wa 的第一个元素设为 -1
    wa[2] = 1;  # 将数组 wa 的第二个元素设为 1
    xa[1] = 100;  # 将数组 xa 的第一个元素设为 100
    xa[2] = 0;  # 将数组 xa 的第二个元素设为 0
    ya[1] = 1;  # 将数组 ya 的第一个元素设为 1
    ya[2] = -1;  // 设置数组 ya 的第二个元素为 -1
    za[1] = 0;   // 设置数组 za 的第一个元素为 0
    za[2] = 100; // 设置数组 za 的第二个元素为 100
    p = 0;       // 将变量 p 设置为 0
    field_headers();  // 调用 field_headers 函数
    print("TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.\n");  // 打印字符串
    t = Math.floor(2 * Math.random() + 1);  // 生成一个随机数并赋值给变量 t
    print("\n");  // 打印换行
    print("THE COIN IS FLIPPED\n");  // 打印字符串
    routine = 1;  // 将变量 routine 设置为 1
    while (1) {   // 进入无限循环
        if (routine <= 1) {  // 如果 routine 小于等于 1
            p = xa[t] - ya[t] * 40;  // 计算并赋值给变量 p
            separator();  // 调用 separator 函数
            print("\n");  // 打印换行
            print("TEAM " + t + " RECEIVES KICK-OFF\n");  // 打印字符串和变量 t
            k = Math.floor(26 * Math.random() + 40);  // 生成一个随机数并赋值给变量 k
        }
        if (routine <= 2) {  // 如果 routine 小于等于 2
            p = p - ya[t] * k;  // 计算并赋值给变量 p
        }
        if (routine <= 3) {  # 如果当前的例行程序小于等于3
            if (wa[t] * p >= za[t] + 10) {  # 如果wa[t] * p大于等于za[t] + 10
                print("\n");  # 打印换行
                print("BALL WENT OUT OF ENDZONE --AUTOMATIC TOUCHBACK--\n");  # 打印球出了终区 --自动触地得分--
                p = za[t] - wa[t] * 20;  # p等于za[t]减去wa[t]乘以20
                if (routine <= 4)  # 如果当前的例行程序小于等于4
                    routine = 5;  # 将例行程序设置为5
            } else {
                print("BALL WENT " + k + " YARDS.  NOW ON " + p + "\n");  # 打印球飞了k码。现在在p位置
                show_ball();  # 显示球的位置
            }
        }
        if (routine <= 4) {  # 如果当前的例行程序小于等于4
            while (1) {  # 进入无限循环
                print("TEAM " + t + " DO YOU WANT TO RUNBACK");  # 打印队伍t，你想要回跑吗
                str = await input();  # 等待输入并将其赋值给str
                if (str == "YES" || str == "NO")  # 如果str等于"YES"或者str等于"NO"
                    break;  # 退出循环
            }
            # 如果条件满足，则执行以下代码块
            if (str == "YES") {
                # 生成一个1到9之间的随机整数
                k = Math.floor(9 * Math.random() + 1);
                # 生成一个随机整数，用于计算p的新值
                r = Math.floor(((xa[t] - ya[t] * p + 25) * Math.random() - 15) / k);
                # 更新p的值
                p = p - wa[t] * r;
                # 打印信息
                print("\n");
                print("RUNBACK TEAM " + t + " " + r + " YARDS\n");
                # 生成一个0到1之间的随机数
                g = Math.random();
                # 如果g小于0.25，则执行以下代码块
                if (g < 0.25) {
                    loss_posession();
                    routine = 4;
                    continue;
                # 如果ya[t] * p大于等于xa[t]，则执行以下代码块
                } else if (ya[t] * p >= xa[t]) {
                    touchdown();
                    # 如果show_scores()返回True，则结束函数
                    if (show_scores())
                        return;
                    # 更新t和routine的值
                    t = ta[t];
                    routine = 1;
                    continue;
                # 如果wa[t] * p大于等于za[t]，则执行以下代码块
                } else if (wa[t] * p >= za[t]) {
                    print("\n");
                    # 打印安全性对阵队伍的消息
                    print("SAFETY AGAINST TEAM " + t + " **********************OH-OH\n");
                    # 更新得分字典中对应队伍的得分
                    ha[ta[t]] = ha[ta[t]] + 2;
                    # 如果需要展示得分，则返回
                    if (show_scores())
                        return;
                    # 打印询问是否要进行开球而不是踢球
                    print("TEAM " + t + " DO YOU WANT TO PUNT INSTEAD OF A KICKOFF");
                    # 等待输入
                    str = await input();
                    # 计算需要踢球的位置
                    p = za[t] - wa[t] * 20;
                    # 如果答复是"YES"
                    if (str == "YES") {
                        # 打印踢球的消息
                        print("\n");
                        print("TEAM " + t + " WILL PUNT\n");
                        # 生成一个随机数
                        g = Math.random();
                        # 如果随机数小于0.25
                        if (g < 0.25) {
                            # 失去球权
                            loss_posession();
                            # 设置下一个动作为4
                            routine = 4;
                            # 继续循环
                            continue;
                        }
                        # 打印分隔线
                        print("\n");
                        separator();
                        # 计算踢球的距离
                        k = Math.floor(25 * Math.random() + 35);
                        # 更新队伍
                        t = ta[t];
                        routine = 2;  # 设置变量routine的值为2
                        continue;  # 继续下一次循环
                    }
                    touchdown();  # 调用touchdown函数
                    if (show_scores())  # 如果show_scores函数返回True
                        return;  # 返回
                    t = ta[t];  # 将t的值更新为ta[t]的值
                    routine = 1;  # 设置变量routine的值为1
                    continue;  # 继续下一次循环
                } else {
                    routine = 5;  # 设置变量routine的值为5
                    continue;  # 继续下一次循环
                }
            } else if (str == "NO") {  # 如果str的值为"NO"
                if (wa[t] * p >= za[t])  # 如果wa[t]乘以p的值大于等于za[t]
                    p = za[t] - wa[t] * 20;  # 更新p的值为za[t]减去wa[t]乘以20的结果
            }
        }
        if (routine <= 5) {  # 如果routine的值小于等于5
            d = 1;  # 设置变量d的值为1
            s = p;  // 将变量 p 的值赋给变量 s
        }
        if (routine <= 6) {  // 如果变量 routine 的值小于等于 6
            str = "";  // 初始化变量 str 为空字符串
            for (i = 1; i <= 72; i++)  // 循环 72 次
                str += "=";  // 将 "=" 添加到变量 str 中
            print(str + "\n");  // 打印变量 str 的值并换行
            print("TEAM " + t + " DOWN " + d + " ON " + p + "\n");  // 打印固定文本和变量 t、d、p 的值并换行
            if (d == 1) {  // 如果变量 d 的值等于 1
                if (ya[t] * (p + ya[t] * 10) >= xa[t])  // 如果条件成立
                    c = 8;  // 将变量 c 的值设为 8
                else
                    c = 4;  // 否则将变量 c 的值设为 4
            }
            if (c != 8) {  // 如果变量 c 的值不等于 8
                print(tab(27) + (10 - (ya[t] * p - ya[t] * s)) + " YARDS TO 1ST DOWN\n");  // 打印固定文本和计算结果并换行
            } else {  // 否则
                print(tab(27) + (xa[t] - ya[t] * p) + " YARDS\n");  // 打印固定文本和计算结果并换行
            }
            show_ball();  // 调用函数 show_ball()
            if (d == 4)  # 如果变量 d 的值等于 4
                routine = 8;  # 则将变量 routine 的值设为 8
        }
        if (routine <= 7) {  # 如果变量 routine 的值小于等于 7
            u = Math.floor(3 * Math.random() - 1);  # 生成一个随机数并赋值给变量 u
            while (1) {  # 进入一个无限循环
                print("INPUT OFFENSIVE PLAY, DEFENSIVE PLAY");  # 打印提示信息
                str = await input();  # 等待用户输入并将输入的值赋给变量 str
                if (t == 1) {  # 如果变量 t 的值等于 1
                    p1 = parseInt(str);  # 将输入的值转换为整数并赋给变量 p1
                    p2 = parseInt(str.substr(str.indexOf(",") + 1));  # 将逗号后的部分转换为整数并赋给变量 p2
                } else {  # 否则
                    p2 = parseInt(str);  # 将输入的值转换为整数并赋给变量 p2
                    p1 = parseInt(str.substr(str.indexOf(",") + 1));  # 将逗号后的部分转换为整数并赋给变量 p1
                }
                if (p1 == 99) {  # 如果变量 p1 的值等于 99
                    if (show_scores())  # 调用 show_scores() 函数并检查返回值
                        return;  # 如果返回值为真，则结束函数执行
                    if (p1 == 99)  # 如果变量 p1 的值等于 99
                        continue;  # 继续下一次循环
                }
                # 如果p1或p2小于1或大于20，则打印错误信息并继续循环
                if (p1 < 1 || p1 > 20 || p2 < 1 || p2 > 20) {
                    print("ILLEGAL PLAY NUMBER, CHECK AND\n");
                    continue;
                }
                # 跳出循环
                break;
            }
        }
        # 如果d等于4或p1等于77
        if (d == 4 || p1 == 77) {
            # 进入循环
            while (1) {
                print("DOES TEAM " + t + " WANT TO PUNT");
                # 等待输入
                str = await input();
                # 如果输入为"YES"或"NO"，跳出循环
                if (str == "YES" || str == "NO")
                    break;
            }
            # 如果输入为"YES"
            if (str == "YES") {
                print("\n");
                print("TEAM " + t + " WILL PUNT\n");
                # 生成一个0到1之间的随机数
                g = Math.random();
                # 如果随机数小于0.25
                if (g < 0.25) {
                    loss_posession();  # 失去球权
                    routine = 4;  # 将程序状态设置为4
                    continue;  # 继续执行下一轮循环
                }
                print("\n");  # 打印空行
                separator();  # 调用分隔符函数
                k = Math.floor(25 * Math.random() + 35);  # 生成一个随机数并赋值给k
                t = ta[t];  # 将t的值更新为ta[t]的值
                routine = 2;  # 将程序状态设置为2
                continue;  # 继续执行下一轮循环
            }
            while (1) {  # 进入无限循环
                print("DOES TEAM " + t + " WANT TO ATTEMPT A FIELD GOAL");  # 打印提示信息
                str = await input();  # 等待输入并将输入值赋给str
                if (str == "YES" || str == "NO")  # 如果输入值为"YES"或"NO"，则跳出循环
                    break;
            }
            if (str == "YES") {  # 如果输入值为"YES"
                print("\n");  # 打印空行
                print("TEAM " + t + " WILL ATTEMPT A FIELD GOAL\n");  # 打印提示信息
                g = Math.random();  # 生成一个 0 到 1 之间的随机数并赋值给变量 g
                if (g < 0.025) {  # 如果 g 小于 0.025
                    loss_posession();  # 调用 loss_posession() 函数
                    routine = 4;  # 将 routine 变量赋值为 4
                    continue;  # 跳过当前循环的剩余代码，继续下一次循环
                } else {  # 如果 g 不小于 0.025
                    f = Math.floor(35 * Math.random() + 20);  # 生成一个 20 到 55 之间的随机整数并赋值给变量 f
                    print("\n");  # 打印一个空行
                    print("KICK IS " + f + " YARDS LONG\n");  # 打印 "KICK IS " 后跟 f 的值再跟 " YARDS LONG" 的字符串
                    p = p - wa[t] * f;  # 计算并更新变量 p 的值
                    g = Math.random();  # 生成一个新的 0 到 1 之间的随机数并赋值给变量 g
                    if (g < 0.35) {  # 如果 g 小于 0.35
                        print("BALL WENT WIDE\n");  # 打印 "BALL WENT WIDE" 字符串
                    } else if (ya[t] * p >= xa[t]) {  # 如果 ya[t] 乘以 p 大于等于 xa[t]
                        print("FIELD GOLD GOOD FOR TEAM " + t + " *********************YEA");  # 打印 "FIELD GOLD GOOD FOR TEAM " 后跟 t 的值再跟 " *********************YEA" 的字符串
                        q = 3;  # 将变量 q 赋值为 3
                        ha[t] = ha[t] + q;  # 更新 ha[t] 的值
                        if (show_scores())  # 如果 show_scores() 函数返回真值
                            return;  # 返回当前函数
                        t = ta[t];  # 更新变量 t 的值
# 设置变量 routine 为 1
routine = 1
# 继续执行下一次循环
continue
# 打印提示信息，表示球门未进球
print("FIELD GOAL UNSUCCESFUL TEAM " + t + "-----------------TOO BAD\n")
# 打印换行符
print("\n")
# 调用函数 separator()，用于打印分隔符
separator()
# 如果球队 t 的得分乘以 p 小于球队 t 的进攻码 xa[t] 加上 10
if (ya[t] * p < xa[t] + 10):
    # 打印提示信息，表示球权转移到位置 p
    print("BALL NOW ON " + p + "\n")
    # 将 t 更新为 ta[t]
    t = ta[t]
    # 调用函数 show_ball()，用于展示球的位置
    show_ball()
    # 设置变量 routine 为 4
    routine = 4
    # 继续执行下一次循环
    continue
# 如果条件不满足
else:
    # 将 t 更新为 ta[t]
    t = ta[t]
    # 设置变量 routine 为 3
    routine = 3
    # 继续执行下一次循环
    continue
                routine = 7;  // 设置变量routine的值为7
                continue;  // 跳过当前循环的剩余代码，继续下一次循环
            }
        }
        y = Math.floor(Math.abs(aa[p1] - ba[p2]) / 19 * ((xa[t] - ya[t] * p + 25) * Math.random() - 15));  // 计算y的值
        print("\n");  // 打印换行符
        if (t == 1 && aa[p1] < 11 || t == 2 && ba[p2] < 11) {  // 如果条件成立
            print("THE BALL WAS RUN\n");  // 打印"The ball was run"
        } else if (u == 0) {  // 如果条件成立
            print("PASS INCOMPLETE TEAM " + t + "\n");  // 打印"PASS INCOMPLETE TEAM "和变量t的值，并换行
            y = 0;  // 设置变量y的值为0
        } else {  // 如果以上条件都不成立
            g = Math.random();  // 生成一个随机数并赋值给变量g
            if (g <= 0.025 && y > 2) {  // 如果条件成立
                print("PASS COMPLETED\n");  // 打印"PASS COMPLETED"
            } else {  // 如果以上条件不成立
                print("QUARTERBACK SCRAMBLED\n");  // 打印"QUARTERBACK SCRAMBLED"
            }
        }
        p = p - wa[t] * y;  // 计算p的值
        # 打印换行
        print("\n");
        # 打印“NET YARDS GAINED ON DOWN” + d + “ARE” + y + “\n”
        print("NET YARDS GAINED ON DOWN " + d + " ARE " + y + "\n");

        # 生成一个随机数 g
        g = Math.random();
        # 如果 g 小于等于 0.025，则执行失去球权的操作，设置 routine 为 4，并继续循环
        if (g <= 0.025) {
            loss_posession();
            routine = 4;
            continue;
        } else if (ya[t] * p >= xa[t]) {
            # 如果 ya[t] * p 大于等于 xa[t]，执行 Touchdown 操作
            touchdown();
            # 如果显示比分，则返回
            if (show_scores())
                return;
            # 设置 t 为 ta[t]，设置 routine 为 1，并继续循环
            t = ta[t];
            routine = 1;
            continue;
        } else if (wa[t] * p >= za[t]) {
            # 打印换行
            print("\n");
            # 打印“SAFETY AGAINST TEAM” + t + “**********************OH-OH\n”
            print("SAFETY AGAINST TEAM " + t + " **********************OH-OH\n");
            # 将 ha[ta[t]] 增加 2
            ha[ta[t]] = ha[ta[t]] + 2;
            # 如果显示比分，则...
            if (show_scores())
                return;  # 结束当前函数的执行并返回
            print("TEAM " + t + " DO YOU WANT TO PUNT INSTEAD OF A KICKOFF");  # 打印提示信息，询问是否要选择开球后进行踢球
            str = await input();  # 等待用户输入，并将输入内容赋值给变量str
            p = za[t] - wa[t] * 20;  # 根据特定公式计算变量p的值
            if (str == "YES") {  # 如果用户输入为"YES"，则执行以下代码块
                print("\n");  # 打印换行符
                print("TEAM " + t + " WILL PUNT\n");  # 打印提示信息，表示球队将进行踢球
                g = Math.random();  # 生成一个随机数并赋值给变量g
                if (g < 0.25) {  # 如果随机数小于0.25，则执行以下代码块
                    loss_posession();  # 调用loss_posession函数
                    routine = 4;  # 将变量routine的值设为4
                    continue;  # 跳过当前循环的剩余代码，继续下一次循环
                }
                print("\n");  # 打印换行符
                separator();  # 调用separator函数
                k = Math.floor(25 * Math.random() + 35);  # 生成一个随机数并赋值给变量k
                t = ta[t];  # 根据特定规则更新变量t的值
                routine = 2;  # 将变量routine的值设为2
                continue;  # 跳过当前循环的剩余代码，继续下一次循环
            }
            touchdown();  # 调用touchdown函数，执行touchdown动作
            if (show_scores())  # 如果show_scores函数返回True
                return;  # 返回空值，结束函数
            t = ta[t];  # 将t赋值为ta[t]的值
            routine = 1;  # 将routine赋值为1
        } else if (ya[t] * p - ya[t] * s >= 10) {  # 否则如果ya[t] * p - ya[t] * s大于等于10
            routine = 5;  # 将routine赋值为5
        } else {  # 否则
            d++;  # d自增1
            if (d != 5) {  # 如果d不等于5
                routine = 6;  # 将routine赋值为6
            } else {  # 否则
                print("\n");  # 打印换行符
                print("CONVERSION UNSUCCESSFUL TEAM " + t + "\n");  # 打印CONVERSION UNSUCCESSFUL TEAM和t的值
                t = ta[t];  # 将t赋值为ta[t]的值
                print("\n");  # 打印换行符
                separator();  # 调用separator函数
                routine = 5;  # 将routine赋值为5
            }
        }
    }
}
```
这部分代码是一个函数的结束和一个程序的结束。在Python中，函数的结束需要使用"}"来表示，而程序的结束需要调用main()函数来执行程序的主要逻辑。
```