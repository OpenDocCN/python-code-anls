# `basic-computer-games\49_Hockey\javascript\hockey.js`

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
                       // 创建一个输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       // 设置输入元素类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入元素获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入元素
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

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一些数组变量
var as = [];
var bs = [];
var ha = [];
var ta = [];
var t1 = [];
var t2 = [];
var t3 = [];

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "HOCKEY\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化数组 ha
    // Robert Puopolo Alg. 1 140 McCowan 6/7/73 Hockey
    for (c = 0; c <= 20; c++)
        ha[c] = 0;
    # 初始化四个数组，每个数组包含5个元素，初始值都为0
    for (c = 1; c <= 5; c++) {
        ta[c] = 0;
        t1[c] = 0;
        t2[c] = 0;
        t3[c] = 0;
    }
    # 初始化变量 x 为 1
    x = 1;
    # 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    # 进入无限循环，直到用户输入 YES 或 NO 为止
    while (1) {
        # 提示用户是否需要游戏说明
        print("WOULD YOU LIKE THE INSTRUCTIONS");
        # 等待用户输入
        str = await input();
        print("\n");
        # 如果用户输入为 YES 或 NO，则跳出循环
        if (str == "YES" || str == "NO")
            break;
        # 如果用户输入不是 YES 或 NO，则提示用户重新输入
        print("ANSWER YES OR NO!!\n");
    }
    # 如果用户输入为 YES，则打印游戏说明
    if (str == "YES") {
        print("\n");
        print("THIS IS A SIMULATED HOCKEY GAME.\n");
        print("QUESTION     RESPONSE\n");
        print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD\n");
        print("             LIKE TO MAKE, FROM 0 TO 3.\n");
        print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT\n");
        print("             YOU WANT TO MAKE.  ENTER:\n");
        print("             1 FOR A SLAPSHOT\n");
        print("             2 FOR A WRISTSHOT\n");
        print("             3 FOR A BACKHAND\n");
        print("             4 FOR A SNAP SHOT\n");
        print("AREA         TYPE IN THE NUMBER CORRESPONDING TO\n");
        print("             THE AREA YOU ARE AIMING AT.  ENTER:\n");
        print("             1 FOR UPPER LEFT HAND CORNER\n");
        print("             2 FOR UPPER RIGHT HAND CORNER\n");
        print("             3 FOR LOWER LEFT HAND CORNER\n");
        print("             4 FOR LOWER RIGHT HAND CORNER\n");
        print("\n");
        print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES\n");
        print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: \n");
        print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,\n");
        print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL\n");
        print("HAVE EXPLANATORY INSTRUCTIONS.\n");
    }
    # 提示用户输入两支队伍的名称
    print("ENTER THE TWO TEAMS");
    # 等待用户输入
    str = await input();
    # 获取输入字符串中逗号的位置
    c = str.indexOf(",");
    # 将输入字符串分割成两部分，分别存入数组 as 和 bs 的第七个元素
    as[7] = str.substr(0, c);
    bs[7] = str.substr(c + 1);
    print("\n");
    // 循环，直到输入的分钟数大于等于1
    do {
        // 打印提示信息，输入分钟数
        print("ENTER THE NUMBER OF MINUTES IN A GAME");
        // 将输入的字符串转换为整数
        t6 = parseInt(await input());
        // 打印换行符
        print("\n");
    } while (t6 < 1) ;
    // 打印换行符
    print("\n");
    // 打印提示信息，询问第7个元素的值是否会进入球队
    print("WOULD THE " + as[7] + " COACH ENTER HIS TEAM\n");
    // 打印换行符
    print("\n");
    // 循环，输入6个球员的信息
    for (i = 1; i <= 6; i++) {
        // 打印提示信息，输入球员信息
        print("PLAYER " + i + " ");
        // 等待输入球员信息
        as[i] = await input();
    }
    // 打印换行符
    print("\n");
    // 打印提示信息，询问第7个元素的值是否会进入球队
    print("WOULD THE " + bs[7] + " COACH DO THE SAME\n");
    // 打印换行符
    print("\n");
    // 循环，输入6个球员的信息
    for (t = 1; t <= 6; t++) {
        // 打印提示信息，输入球员信息
        print("PLAYER " + t + " ");
        // 等待输入球员信息
        bs[t] = await input();
    }
    // 打印换行符
    print("\n");
    // 打印提示信息，输入裁判信息
    print("INPUT THE REFEREE FOR THIS GAME");
    // 等待输入裁判信息
    rs = await input();
    // 打印换行符
    print("\n");
    // 打印第7个元素的值和起始阵容
    print(tab(10) + as[7] + " STARTING LINEUP\n");
    // 循环，打印起始阵容
    for (t = 1; t <= 6; t++) {
        // 打印球员信息
        print(as[t] + "\n");
    }
    // 打印换行符
    print("\n");
    // 打印第7个元素的值和起始阵容
    print(tab(10) + bs[7] + " STARTING LINEUP\n");
    // 循环，打印起始阵容
    for (t = 1; t <= 6; t++) {
        // 打印球员信息
        print(bs[t] + "\n");
    }
    // 打印换行符
    print("\n");
    // 打印提示信息，比赛准备好了
    print("WE'RE READY FOR TONIGHTS OPENING FACE-OFF.\n");
    // 打印裁判和球员信息
    print(rs + " WILL DROP THE PUCK BETWEEN " + as[2] + " AND " + bs[2] + "\n");
    // 初始化变量
    s2 = 0;
    s3 = 0;
    }
    // 打印提示信息，比赛结束
    print("THAT'S THE SIREN\n");
    // 打印换行符
    print("\n");
    // 打印最终比分
    print(tab(15) + "FINAL SCORE:\n");
    // 判断主队得分和客队得分，打印不同的比分信息
    if (ha[8] <= ha[9]) {
        print(as[7] + ": " + ha[9] + "\t" + bs[7] + ": " + ha[8] + "\n");
    } else {
        print(bs[7] + ": " + ha[8] + "\t" + as[7] + ": " + ha[9] + "\n");
    }
    // 打印换行符
    print("\n");
    // 打印提示信息，比分摘要
    print(tab(10) + "SCORING SUMMARY\n");
    // 打印换行符
    print("\n");
    // 打印主队球员的得分情况
    print(tab(25) + as[7] + "\n");
    // 打印表头
    print("\tNAME\tGOALS\tASSISTS\n");
    print("\t----\t-----\t-------\n");
    // 循环，打印球员的得分情况
    for (i = 1; i <= 5; i++) {
        print("\t" + as[i] + "\t" + ta[i] + "\t" + t1[i] + "\n");
    }
    // 打印换行符
    print("\n");
    // 打印客队球员的得分情况
    print(tab(25) + bs[7] + "\n");
    // 打印表头
    print("\tNAME\tGOALS\tASSISTS\n");
    print("\t----\t-----\t-------\n");
    // 循环，打印球员的得分情况
    for (t = 1; t <= 5; t++) {
        print("\t" + bs[t] + "\t" + t2[t] + "\t" + t3[t] + "\n");
    }
    // 打印换行符
    print("\n");
    // 打印提示信息，射门次数
    print("SHOTS ON NET\n");
    // 打印主队的射门次数
    print(as[7] + ": " + s2 + "\n");
    # 打印 bs 列表中索引为 7 的元素和字符串 s3 的拼接结果，再加上换行符
    print(bs[7] + ": " + s3 + "\n");
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```