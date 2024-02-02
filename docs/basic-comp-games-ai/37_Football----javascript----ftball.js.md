# `basic-computer-games\37_Football\javascript\ftball.js`

```py
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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象并传递输入的字符串
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

// 定义变量 os, sa, ls, p, x, x1
var os = [];
var sa = [];
var ls = [, "KICK","RECEIVE"," YARD ","RUN BACK FOR ","BALL ON ",
          "YARD LINE"," SIMPLE RUN"," TRICKY RUN"," SHORT PASS",
          " LONG PASS","PUNT"," QUICK KICK "," PLACE KICK"," LOSS ",
          " NO GAIN","GAIN "," TOUCHDOWN "," TOUCHBACK ","SAFETY***",
          "JUNK"];
var p;
var x;
var x1;

// 定义函数 fnf，返回 1 - 2 * p
function fnf(x)
{
    return 1 - 2 * p;
}

// 定义函数 fng，返回 p * (x1 - x) + (1 - p) * (x - x1)
function fng(z)
{
    return p * (x1 - x) + (1 - p) * (x - x1);
}
// 显示比分
function show_score()
{
    // 打印空行
    print("\n");
    // 打印比分信息
    print("SCORE:  " + sa[0] + " TO " + sa[1] + "\n");
    // 打印两个空行
    print("\n");
    print("\n");
}

// 显示位置
function show_position()
{
    // 如果位置小于等于50
    if (x <= 50) {
        // 打印位置信息
        print(ls[5] + os[0] + " " + x + " " + ls[6] + "\n");
    } else {
        // 否则打印位置信息
        print(ls[5] + os[1] + " " + (100 - x) + " " + ls[6] + "\n");
    }
}

// 进攻得分
function offensive_td()
{
    // 打印进攻得分信息
    print(ls[17] + "***\n");
    // 如果随机数小于等于0.8
    if (Math.random() <= 0.8) {
        // 更新得分
        sa[p] = sa[p] + 7;
        // 打印进攻得分信息
        print("KICK IS GOOD.\n");
    } else {
        // 否则打印进攻得分信息
        print("KICK IS OFF TO THE SIDE\n");
        // 更新得分
        sa[p] = sa[p] + 6;
    }
    // 显示比分
    show_score();
    // 打印进攻信息
    print(os[p] + " KICKS OFF\n");
    // 更新球队
    p = 1 - p;
}

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "FTBALL\n");
    // 打印信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印两个空行
    print("\n");
    print("\n");
    // 打印比赛信息
    print("THIS IS DARTMOUTH CHAMPIONSHIP FOOTBALL.\n");
    // 打印提示信息
    print("YOU WILL QUARTERBACK DARTMOUTH. CALL PLAYS AS FOLLOWS:\n");
    print("1= SIMPLE RUN; 2= TRICKY RUN; 3= SHORT PASS;\n");
    print("4= LONG PASS; 5= PUNT; 6= QUICK KICK; 7= PLACE KICK.\n");
    // 打印两个空行
    print("\n");
    print("CHOOSE YOUR OPPONENT");
    // 获取对手信息
    os[1] = await input();
    // 设置对手信息
    os[0] = "DARMOUTH";
    // 初始化得分
    sa[0] = 0;
    sa[1] = 0;
    // 随机决定先发球的球队
    p = Math.floor(Math.random() * 2);
    // 打印谁赢得了抛硬币
    print(os[p] + " WON THE TOSS\n");
    // 如果先发球的不是DARMOUTH
    if (p != 0) {
        // 打印对手选择接球
        print(os[1] + " ELECTS TO RECEIVE.\n");
        // 打印两个空行
        print("\n");
    } else {
        // 否则询问是否选择踢球还是接球
        print("DO YOU ELECT TO KICK OR RECEIVE");
        // 循环直到输入正确的选项
        while (1) {
            str = await input();
            print("\n");
            // 如果输入的是踢球或接球
            if (str == ls[1] || str == ls[2])
                break;
            // 否则提示输入错误
            print("INCORRECT ANSWER.  PLEASE TYPE 'KICK' OR 'RECEIVE'");
        }
        // 根据选择更新p的值
        e = (str == ls[1]) ? 1 : 2;
        if (e == 1)
            p = 1;
    }
    // 初始化时间和开始标志
    t = 0;
    start = 1;
    // 打印比赛结束信息
    print("END OF GAME  ***\n");
    // 打印最终比分
    print("FINAL SCORE:  " + os[0] + ": " + sa[0] + "  " + os[1] + ": " + sa[1] + "\n");
}

// 调用主程序
main();
```