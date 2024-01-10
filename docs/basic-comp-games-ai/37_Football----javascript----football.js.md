# `basic-computer-games\37_Football\javascript\football.js`

```
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
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听输入框的按键事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入框的值，移除输入框，打印输入的值，然后解析 Promise
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 初始化一些数组
var player_data = [17,8,4,14,19,3,10,1,7,11,15,9,5,20,13,18,16,2,12,6,
                   20,2,17,5,8,18,12,11,1,4,19,14,10,7,9,15,6,13,16,3];
var aa = [];
var ba = [];
var ca = [];
var ha = [];
var ta = [];
var wa = [];
var xa = [];
var ya = [];
var za = [];
var ms = [];
var da = [];
# 定义包含不同足球战术的数组
var ps = [, "PITCHOUT","TRIPLE REVERSE","DRAW","QB SNEAK","END AROUND",
          "DOUBLE REVERSE","LEFT SWEEP","RIGHT SWEEP","OFF TACKLE",
          "WISHBONE OPTION","FLARE PASS","SCREEN PASS",
          "ROLL OUT OPTION","RIGHT CURL","LEFT CURL","WISHBONE OPTION",
          "SIDELINE PASS","HALF-BACK OPTION","RAZZLE-DAZZLE","BOMB!!!!"];
# 定义变量 p
var p;
# 定义变量 t
var t;

# 定义输出球场标头的函数
function field_headers()
{
    # 输出球场标头
    print("TEAM 1 [0   10   20   30   40   50   60   70   80   90");
    print("   100] TEAM 2\n");
    print("\n");
}

# 定义分隔线函数
function separator()
{
    # 初始化空字符串
    str = "";
    # 循环生成分隔线
    for (x = 1; x <= 72; x++)
        str += "+";
    # 输出分隔线
    print(str + "\n");
}

# 定义显示球的位置函数
function show_ball()
{
    # 输出球的位置
    print(tab(da[t] + 5 + p / 2) + ms[t] + "\n");
    # 调用输出球场标头的函数
    field_headers();
}

# 定义显示比分函数
function show_scores()
{
    # 输出空行
    print("\n");
    # 输出TEAM 1的得分
    print("TEAM 1 SCORE IS " + ha[1] + "\n");
    # 输出TEAM 2的得分
    print("TEAM 2 SCORE IS " + ha[2] + "\n");
    # 输出空行
    print("\n");
    # 如果某队得分大于等于e，则输出该队获胜，并返回true
    if (ha[t] >= e) {
        print("TEAM " + t + " WINS*******************");
        return true;
    }
    # 否则返回false
    return false;
}

# 定义失去控球函数
function loss_posession() {
    # 输出失去控球的信息
    print("\n");
    print("** LOSS OF POSSESSION FROM TEAM " + t + " TO TEAM " + ta[t] + "\n");
    print("\n");
    # 调用分隔线函数
    separator();
    print("\n");
    # 将t设置为失去控球后的队伍
    t = ta[t];
}

# 定义Touchdown函数
function touchdown() {
    # 输出Touchdown的信息
    print("\n");
    print("TOUCHDOWN BY TEAM " + t + " *********************YEA TEAM\n");
    # 设置q为7
    q = 7;
    # 生成一个随机数g
    g = Math.random();
    # 如果g小于等于0.1，则将q设置为6，并输出额外得分不好的信息，否则输出额外得分好的信息
    if (g <= 0.1) {
        q = 6;
        print("EXTRA POINT NO GOOD\n");
    } else {
        print("EXTRA POINT GOOD\n");
    }
    # 将t队伍的得分增加q
    ha[t] = ha[t] + q;
}

# 主程序
async function main()
{
    # 输出标题
    print(tab(32) + "FOOTBALL\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("PRESENTING N.F.U. FOOTBALL (NO FORTRAN USED)\n");
    print("\n");
    print("\n");
    # 循环询问是否需要说明
    while (1) {
        print("DO YOU WANT INSTRUCTIONS");
        # 等待输入
        str = await input();
        # 如果输入为"YES"或"NO"，则跳出循环
        if (str == "YES" || str == "NO")
            break;
    }
    # 如果输入的字符串为"YES"，则执行以下代码块
    if (str == "YES") {
        # 打印关于足球比赛规则的说明
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
    # 提示用户输入游戏的得分限制
    print("PLEASE INPUT SCORE LIMIT ON GAME");
    # 将用户输入的值转换为整数并赋给变量e
    e = parseInt(await input());
    # 循环，初始化i为1，当i小于等于40时执行循环，每次i加1
    for (i = 1; i <= 40; i++) {
        # 如果i小于等于20，执行以下代码块
        if (i <= 20) {
            # 将player_data中的第i-1个元素作为键，i作为值，存入字典aa中
            aa[player_data[i - 1]] = i;
        } else {
            # 将player_data中的第i-1个元素作为键，i-20作为值，存入字典ba中
            ba[player_data[i - 1]] = i - 20;
        }
        # 将player_data中的第i-1个元素作为值，i作为键，存入字典ca中
        ca[i] = player_data[i - 1];
    }
    # 初始化变量l为0，变量t为1
    l = 0;
    t = 1;
    # 执行循环，直到条件不满足
    do {
        # 打印团队 t 的比赛图表
        print("TEAM " + t + " PLAY CHART\n");
        # 打印表头
        print("NO.      PLAY\n");
        # 遍历 1 到 20
        for (i = 1; i <= 20; i++) {
            # 将 ca[i + l] 转换为字符串
            str = "" + ca[i + l];
            # 如果字符串长度小于 6，则在末尾添加空格
            while (str.length < 6)
                str += " ";
            # 将 ps[i] 添加到字符串末尾
            str += ps[i];
            # 打印字符串
            print(str + "\n");
        }
        # l 增加 20
        l += 20;
        # t 赋值为 2
        t = 2;
        # 打印空行
        print("\n");
        # 打印分隔线
        print("TEAR OFF HERE----------------------------------------------\n");
        # 打印 11 个空行
        for (x = 1; x <= 11; x++)
            print("\n");
    # 条件为 l 等于 20 时继续循环
    } while (l == 20) ;
    # 设置数组元素的值
    da[1] = 0;
    da[2] = 3;
    ms[1] = "--->";
    ms[2] = "<---";
    ha[1] = 0;
    ha[2] = 0;
    ta[1] = 2;
    ta[2] = 1;
    wa[1] = -1;
    wa[2] = 1;
    xa[1] = 100;
    xa[2] = 0;
    ya[1] = 1;
    ya[2] = -1;
    za[1] = 0;
    za[2] = 100;
    p = 0;
    # 调用函数 field_headers
    field_headers();
    # 打印信息
    print("TEAM 1 DEFEND 0 YD GOAL -- TEAM 2 DEFENDS 100 YD GOAL.\n");
    # t 被赋值为 1 或 2
    t = Math.floor(2 * Math.random() + 1);
    # 打印信息
    print("\n");
    print("THE COIN IS FLIPPED\n");
    # 设置变量 routine 的值为 1
    routine = 1;
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```