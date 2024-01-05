# `d:/src/tocomm/basic-computer-games\89_Tic-Tac-Toe\javascript\tictactoe2.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上输出提示符 "? "
// 设置 INPUT 元素的类型为文本输入
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环执行以下操作
    while (space-- > 0)
        str += " ";  # 将空格字符添加到字符串变量str的末尾
    return str;  # 返回处理后的字符串

var s = [];  # 声明一个空数组变量s

function who_win(piece)  # 定义一个名为who_win的函数，参数为piece
{
    if (piece == -1) {  # 如果piece等于-1
        print("I WIN, TURKEY!!!\n");  # 打印"I WIN, TURKEY!!!"并换行
    } else if (piece == 1) {  # 否则如果piece等于1
        print("YOU BEAT ME!! GOOD GAME.\n");  # 打印"YOU BEAT ME!! GOOD GAME."并换行
    }
}

function show_board()  # 定义一个名为show_board的函数
{
    print("\n");  # 打印换行符
    for (i = 1; i <= 9; i++) {  # 循环9次，i从1到9
        print(" ");  # 打印空格字符
        if (s[i] == -1) {  // 如果数组中当前位置的值为-1
            print(qs + " ");  // 打印qs和一个空格
        } else if (s[i] == 0) {  // 否则，如果数组中当前位置的值为0
            print("  ");  // 打印两个空格
        } else {  // 否则
            print(ps + " ");  // 打印ps和一个空格
        }
        if (i == 3 || i == 6) {  // 如果当前位置是3或6
            print("\n");  // 打印换行
            print("---+---+---\n");  // 打印---+---+---
        } else if (i != 9) {  // 否则，如果当前位置不是9
            print("!");  // 打印!
        }
    }
    print("\n");  // 打印三个换行
    print("\n");
    print("\n");
    for (i = 1; i <= 7; i += 3) {  // 循环i从1到7，每次增加3
        if (s[i] && s[i] == s[i + 1] && s[i] == s[i + 2]) {  // 如果数组中当前位置的值不为0且等于下一个两个位置的值
            who_win(s[i]);  // 调用who_win函数并传入s[i]作为参数
            return true;  # 如果有一方获胜，返回真
        }
    }
    for (i = 1; i <= 3; i++) {  # 遍历行
        if (s[i] && s[i] == s[i + 3] && s[i] == s[i + 6]) {  # 如果某一列有相同的标记，表示获胜
            who_win(s[i]);  # 调用函数宣布获胜者
            return true;  # 返回真
        }
    }
    if (s[1] && s[1] == s[5] && s[1] == s[9]) {  # 如果左上到右下对角线有相同的标记，表示获胜
        who_win(s[1]);  # 调用函数宣布获胜者
        return true;  # 返回真
    }
    if (s[3] && s[3] == s[5] && s[3] == s[7]) {  # 如果右上到左下对角线有相同的标记，表示获胜
        who_win(s[3]);  # 调用函数宣布获胜者
        return true;  # 返回真
    }
    for (i = 1; i <= 9; i++) {  # 遍历整个棋盘
        if (s[i] == 0)  # 如果有空格
            break;  # 跳出循环
    }
    if (i > 9) {  # 如果游戏进行了9次，即棋盘已满
        print("IT'S A DRAW. THANK YOU.\n");  # 打印平局信息
        return true;  # 返回true，表示游戏结束
    }
    return false;  # 返回false，表示游戏继续
}

// Main control section
async function main()
{
    print(tab(30) + "TIC-TAC-TOE\n");  # 打印游戏标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印游戏信息
    print("\n");
    print("\n");
    print("\n");
    for (i = 1; i <= 9; i++)  # 初始化棋盘，将每个位置的状态设为0
        s[i] = 0;
    print("THE BOARD IS NUMBERED:\n");  # 打印棋盘编号
    print(" 1  2  3\n");  # 打印棋盘行号
    # 打印 4  5  6
    print(" 4  5  6\n");
    # 打印 7  8  9
    print(" 7  8  9\n");
    # 打印空行
    print("\n");
    # 打印空行
    print("\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("DO YOU WANT 'X' OR 'O'");
    # 获取用户输入的字符串
    str = await input();
    # 如果用户输入为 "X"，则设置 ps 为 "X"，qs 为 "O"，并将 first_time 设置为 true
    if (str == "X") {
        ps = "X";
        qs = "O";
        first_time = true;
    } else {
        # 如果用户输入不为 "X"，则设置 ps 为 "O"，qs 为 "X"，并将 first_time 设置为 false
        ps = "O";
        qs = "X";
        first_time = false;
    }
    # 进入循环
    while (1) {
        # 如果不是第一次循环
        if (!first_time) {
            # 设置 g 为 -1
            g = -1;
            # 设置 h 为 1
            h = 1;
            # 如果 s[5] 等于 0，则将其赋值为 -1
            if (s[5] == 0) {
                s[5] = -1;
            # 如果 s[5] 等于 1 并且 s[1] 等于 0，则将 s[1] 赋值为 -1
            } else if (s[5] == 1 && s[1] == 0) {
                s[1] = -1;
            # 如果 s[5] 不等于 1 并且 s[2] 等于 1 并且 s[1] 等于 0 或者 s[5] 不等于 1 并且 s[4] 等于 1 并且 s[1] 等于 0，则将 s[1] 赋值为 -1
            } else if (s[5] != 1 && s[2] == 1 && s[1] == 0 || s[5] != 1 && s[4] == 1 && s[1] == 0) {
                s[1] = -1;
            # 如果 s[5] 不等于 1 并且 s[6] 等于 1 并且 s[9] 等于 0 或者 s[5] 不等于 1 并且 s[8] 等于 1 并且 s[9] 等于 0，则将 s[9] 赋值为 -1
            } else if (s[5] != 1 && s[6] == 1 && s[9] == 0 || s[5] != 1 && s[8] == 1 && s[9] == 0) {
                s[9] = -1;
            # 否则，进入循环
            } else {
                while (1) {
                    played = false;
                    # 如果 g 等于 1，则执行以下操作
                    if (g == 1) {
                        j = 3 * Math.floor((m - 1) / 3) + 1;
                        # 如果 3 * Math.floor((m - 1) / 3) + 1 等于 m，则将 k 赋值为 1
                        if (3 * Math.floor((m - 1) / 3) + 1 == m)
                            k = 1;
                        # 如果 3 * Math.floor((m - 1) / 3) + 2 等于 m，则将 k 赋值为 2
                        if (3 * Math.floor((m - 1) / 3) + 2 == m)
                            k = 2;
                        # 如果 3 * Math.floor((m - 1) / 3) + 3 等于 m，则将 k 赋值为 3
                        if (3 * Math.floor((m - 1) / 3) + 3 == m)
                            k = 3;
                    # 否则，执行以下操作
                    } else {
# 初始化变量 j 和 k
j = 1;
k = 1;
```

```
# 进入无限循环
while (1) {
```

```
# 检查当前位置是否为 g
if (s[j] == g) {
    # 如果当前位置的下一个位置也为 g
    if (s[j + 2] == g) {
        # 如果当前位置的下一个位置的下一个位置为 0
        if (s[j + 1] == 0) {
            # 将当前位置的下一个位置设为 -1
            s[j + 1] = -1;
            # 设置 played 为 true
            played = true;
            # 退出循环
            break;
        }
    } else {
        # 如果当前位置的下一个位置为 0 并且下下个位置为 g
        if (s[j + 2] == 0 && s[j + 1] == g) {
            # 将当前位置的下一个位置的下一个位置设为 -1
            s[j + 2] = -1;
            # 设置 played 为 true
            played = true;
            # 退出循环
            break;
        }
    }
} else {
    # 如果当前位置不为 h 并且下一个位置为 g 并且下下个位置为 g
    if (s[j] != h && s[j + 2] == g && s[j + 1] == g) {
                                s[j] = -1;  // 如果当前位置为空，则将其标记为已下棋
                                played = true;  // 设置已下棋标记为 true
                                break;  // 跳出当前循环
                            }
                        }
                        if (s[k] == g) {  // 如果当前位置已经被对手占据
                            if (s[k + 6] == g) {  // 如果下一行同一列位置也被对手占据
                                if (s[k + 3] == 0) {  // 如果中间位置为空
                                    s[k + 3] = -1;  // 则将中间位置标记为已下棋
                                    played = true;  // 设置已下棋标记为 true
                                    break;  // 跳出当前循环
                                }
                            } else {
                                if (s[k + 6] == 0 && s[k + 3] == g) {  // 如果下一行同一列位置为空且中间位置被对手占据
                                    s[k + 6] = -1;  // 则将下一行同一列位置标记为已下棋
                                    played = true;  // 设置已下棋标记为 true
                                    break;  // 跳出当前循环
                                }
                            }
                        } else {  // 如果当前位置为空
                            # 如果 s[k] 不等于 h 并且 s[k + 6] 等于 g 并且 s[k + 3] 等于 g，则执行以下操作
                            if (s[k] != h && s[k + 6] == g && s[k + 3] == g) {
                                # 将 s[k] 设置为 -1
                                s[k] = -1;
                                # 将 played 设置为 true
                                played = true;
                                # 跳出循环
                                break;
                            }
                        }
                        # 如果 g 等于 1，则跳出循环
                        if (g == 1)
                            break;
                        # 如果 j 等于 7 并且 k 等于 3，则跳出循环
                        if (j == 7 && k == 3)
                            break;
                        # k 自增 1
                        k++;
                        # 如果 k 大于 3，则执行以下操作
                        if (k > 3) {
                            # 将 k 设置为 1
                            k = 1;
                            # j 自增 3
                            j += 3;
                            # 如果 j 大于 7，则跳出循环
                            if (j > 7)
                                break;
                        }
                    }
                    # 如果 played 为 false，则执行以下操作
                    if (!played) {
                        # 如果 s[5] 等于 g，则执行以下操作
# 如果条件满足，则将数组s中的特定位置赋值为-1，并将played标记为true
if (s[3] == g && s[7] == 0) {
    s[7] = -1;
    played = true;
} 
# 如果条件满足，则将数组s中的特定位置赋值为-1，并将played标记为true
else if (s[9] == g && s[1] == 0) {
    s[1] = -1;
    played = true;
} 
# 如果条件满足，则将数组s中的特定位置赋值为-1，并将played标记为true
else if (s[7] == g && s[3] == 0) {
    s[3] = -1;
    played = true;
} 
# 如果条件满足，则将数组s中的特定位置赋值为-1，并将played标记为true
else if (s[9] == 0 && s[1] == g) {
    s[9] = -1;
    played = true;
}
# 如果没有任何条件满足，则根据g的值进行赋值，并将h的值赋值为-1
if (!played) {
    if (g == -1) {
        g = 1;
        h = -1;
    }
}
                    }
                    # 如果played为真，则跳出循环
                    if (played)
                        break;
                }
                # 如果played为假
                if (!played) {
                    # 如果s[9]等于1且s[3]等于0且s[1]不等于1
                    if (s[9] == 1 && s[3] == 0 && s[1] != 1) {
                        # 将s[3]设为-1
                        s[3] = -1;
                    } else {
                        # 否则
                        for (i = 2; i <= 9; i++) {
                            # 如果s[i]等于0
                            if (s[i] == 0) {
                                # 将s[i]设为-1，并跳出循环
                                s[i] = -1;
                                break;
                            }
                        }
                        # 如果i大于9
                        if (i > 9) {
                            # 将s[1]设为-1
                            s[1] = -1;
                        }
                    }
                }
            }
            # 打印换行
            print("\n");
            # 打印计算机移动到的位置
            print("THE COMPUTER MOVES TO...");
            # 如果显示棋盘成功，则跳出循环
            if (show_board())
                break;
        }
        # 第一次移动标记设为假
        first_time = false;
        # 无限循环，直到玩家输入有效的移动位置
        while (1) {
            # 打印换行
            print("\n");
            # 打印提示玩家输入移动位置
            print("WHERE DO YOU MOVE");
            # 将输入的字符串转换为整数
            m = parseInt(await input());
            # 如果玩家输入0，则打印感谢信息并跳出循环
            if (m == 0) {
                print("THANKS FOR THE GAME.\n");
                break;
            }
            # 如果玩家输入的位置在1到9之间且该位置未被占据，则跳出循环
            if (m >= 1 && m <= 9 && s[m] == 0)
                break;
            # 如果玩家输入的位置已被占据，则打印提示信息
            print("THAT SQUARE IS OCCUPIED.\n");
            # 打印换行
            print("\n");
            # 打印两个换行
            print("\n");
        }
        g = 1;  # 初始化变量 g 为 1
        s[m] = 1;  # 将数组 s 的第 m 个元素赋值为 1
        if (show_board())  # 调用函数 show_board()，如果返回值为真（非零），则执行下面的语句
            break;  # 退出循环
    }
}

main();  # 调用函数 main()
```