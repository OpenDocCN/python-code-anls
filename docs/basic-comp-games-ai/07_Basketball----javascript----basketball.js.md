# `basic-computer-games\07_Basketball\javascript\basketball.js`

```

// BASKETBALL
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串输出到指定元素
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，将输入的字符串返回
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义缩进函数，返回指定数量空格组成的字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义全局变量
var s = [0, 0];
var z;
var d;
var p;
var your_turn;
var game_restart;

// 定义两分钟剩余时间函数
function two_minutes()
{
    print("\n");
    print("   *** TWO MINUTES LEFT IN THE GAME ***\n");
    print("\n");
}

// 定义显示比分函数
function show_scores()
{
    print("SCORE: " + s[1] + " TO " + s[0] + "\n");
}

// 定义电脑得分函数
function score_computer()
{
    s[0] = s[0] + 2;
    show_scores();
}

// 定义玩家得分函数
function score_player()
{
    s[1] = s[1] + 2;
    show_scores();
}

// 定义半场结束函数
function half_time()
{
    print("\n");
    print("   ***** END OF FIRST HALF *****\n");
    print("SCORE: DARMOUTH: " + s[1] + "  " + os + ": " + s[0] + "\n");
    print("\n");
    print("\n");
}

// 定义犯规函数
function foul()
{
    if (Math.random() <= 0.49) {
        print("SHOOTER MAKES BOTH SHOTS.\n");
        s[1 - p] = s[1 - p] + 2;
        show_scores();
    } else if (Math.random() <= 0.75) {
        print("SHOOTER MAKES ONE SHOT AND MISSES ONE.\n");
        s[1 - p] = s[1 - p] + 1;
        show_scores();
    } else {
        print("BOTH SHOTS MISSED.\n");
        show_scores();
    }
}

// 定义玩家出手函数
function player_play()
}

// 定义电脑出手函数
function computer_play()
}

// 主程序
async function main()
}

// 调用主程序
main();

```