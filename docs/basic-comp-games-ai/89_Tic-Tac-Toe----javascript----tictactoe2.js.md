# `basic-computer-games\89_Tic-Tac-Toe\javascript\tictactoe2.js`

```

// TIC TAC TOE 2
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串添加到输出元素中
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
                       // 创建输入元素
                       input_element = document.createElement("INPUT");

                       // 打印提示符
                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，移除输入元素，打印输入的字符串，并解析 Promise
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

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义全局变量
var s = [];

// 判断胜者并打印结果
function who_win(piece)
{
    if (piece == -1) {
        print("I WIN, TURKEY!!!\n");
    } else if (piece == 1) {
        print("YOU BEAT ME!! GOOD GAME.\n");
    }
}

// 显示游戏棋盘
function show_board()
{
    print("\n");
    for (i = 1; i <= 9; i++) {
        print(" ");
        if (s[i] == -1) {
            print(qs + " ");
        } else if (s[i] == 0) {
            print("  ");
        } else {
            print(ps + " ");
        }
        if (i == 3 || i == 6) {
            print("\n");
            print("---+---+---\n");
        } else if (i != 9) {
            print("!");
        }
    }
    print("\n");
    print("\n");
    print("\n");
    for (i = 1; i <= 7; i += 3) {
        if (s[i] && s[i] == s[i + 1] && s[i] == s[i + 2]) {
            who_win(s[i]);
            return true;
        }
    }
    for (i = 1; i <= 3; i++) {
        if (s[i] && s[i] == s[i + 3] && s[i] == s[i + 6]) {
            who_win(s[i]);
            return true;
        }
    }
    if (s[1] && s[1] == s[5] && s[1] == s[9]) {
        who_win(s[1]);
        return true;
    }
    if (s[3] && s[3] == s[5] && s[3] == s[7]) {
        who_win(s[3]);
        return true;
    }
    for (i = 1; i <= 9; i++) {
        if (s[i] == 0)
            break;
    }
    if (i > 9) {
        print("IT'S A DRAW. THANK YOU.\n");
        return true;
    }
    return false;
}

// 主控制部分，使用 async 函数
async function main()
}

// 调用主函数
main();

```