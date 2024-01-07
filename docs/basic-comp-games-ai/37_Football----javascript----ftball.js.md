# `basic-computer-games\37_Football\javascript\ftball.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    // 声明变量
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
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 声明变量
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

// 定义函数，返回对手的得分
function fnf(x)
{
    return 1 - 2 * p;
}

// 定义函数，返回对手的进攻得分
function fng(z)
{
    return p * (x1 - x) + (1 - p) * (x - x1);
}

// 定义函数，显示比分
function show_score()
{
    print("\n");
    print("SCORE:  " + sa[0] + " TO " + sa[1] + "\n");
    print("\n");
    print("\n");
}

// 定义函数，显示位置
function show_position()
{
    if (x <= 50) {
        print(ls[5] + os[0] + " " + x + " " + ls[6] + "\n");
    } else {
        print(ls[5] + os[1] + " " + (100 - x) + " " + ls[6] + "\n");
    }
}

// 定义函数，进攻得分
function offensive_td()
{
    print(ls[17] + "***\n");
    if (Math.random() <= 0.8) {
        sa[p] = sa[p] + 7;
        print("KICK IS GOOD.\n");
    } else {
        print("KICK IS OFF TO THE SIDE\n");
        sa[p] = sa[p] + 6;
    }
    show_score();
    print(os[p] + " KICKS OFF\n");
    p = 1 - p;
}

// 主程序
async function main()
}

// 调用主程序
main();

```