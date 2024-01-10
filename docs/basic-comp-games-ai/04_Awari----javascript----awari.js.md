# `basic-computer-games\04_Awari\javascript\awari.js`

```
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
                       // 设置输入元素类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入元素添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 设置输入元素焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入值
                                                      input_str = input_element.value;
                                                      // 移除输入元素
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入值
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

// 打印游戏标题
print(tab(34) + "AWARI\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

// 初始化变量
n = 0;
b = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
g = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];
f = [];
for (i = 0; i <= 50; i++) {
    f[i] = 0;
}

// 定义显示数字函数
function show_number(number)
{
    if (number < 10)
        print("  " + number + " ");
    else
        print(" " + number + " ");
}

// 定义显示游戏板函数
function show_board()
{
    var i;

    // 打印换行
    print("\n");
    // 打印空格
    print("   ");
    # 从12到7递减循环，依次展示数组b中的数字
    for (i = 12; i >= 7; i--)
        show_number(b[i]);
    # 打印换行符
    print("\n");
    # 将i设置为13
    i = 13;
    # 展示数组b中索引为13的数字
    show_number(b[i]);
    # 打印空格和数组b中索引为6的数字
    print("                       " + b[6] + "\n");
    # 打印空格
    print("   ");
    # 从0到5递增循环，依次展示数组b中的数字
    for (i = 0; i <= 5; i++)
        show_number(b[i]);
    # 打印换行符
    print("\n");
    # 打印两个换行符
    print("\n");
}
// 结束函数定义

function do_move()
{
    // 将 m 的值赋给 k
    k = m;
    // 调用 adjust_board 函数
    adjust_board();
    // 将 e 的值设为 0
    e = 0;
    // 如果 k 大于 6，则将 k 减去 7
    if (k > 6)
        k -= 7;
    // c 值加一
    c++;
    // 如果 c 小于 9，则将 f[n] 的值乘以 6 再加上 k
    if (c < 9)
        f[n] = f[n] * 6 + k
        // 循环 i 从 0 到 5
        for (i = 0; i <= 5; i++) {
            // 如果 b[i] 不等于 0
            if (b[i] != 0) {
                // 循环 i 从 7 到 12
                for (i = 7; i <= 12; i++) {
                    // 如果 b[i] 不等于 0
                    if (b[i] != 0) {
                        // 将 e 的值设为 1
                        e = 1;
                        // 返回
                        return;
                    }
                }
            }
        }
}

function adjust_board()
{
    // 将 b[m] 的值赋给 p
    p = b[m];
    // 将 b[m] 的值设为 0
    b[m] = 0;
    // 当 p 大于等于 1 时循环
    while (p >= 1) {
        // m 值加一
        m++;
        // 如果 m 大于 13，则将 m 减去 14
        if (m > 13)
            m -= 14;
        // b[m] 的值加一
        b[m]++;
        // p 的值减一
        p--;
    }
    // 如果 b[m] 的值等于 1
    if (b[m] == 1) {
        // 如果 m 不等于 6 且 m 不等于 13
        if (m != 6 && m != 13) {
            // 如果 b[12 - m] 不等于 0
            if (b[12 - m] != 0) {
                // 将 b[12 - m] 加上 b[h] 再加上 1 赋给 b[h]
                b[h] += b[12 - m] + 1;
                // 将 b[m] 的值设为 0
                b[m] = 0;
                // 将 b[12 - m] 的值设为 0
                b[12 - m] = 0;
            }
        }
    }
}

function computer_move()
{
    // 将 -99 赋给 d
    d = -99;
    // 将 13 赋给 h
    h = 13;
    // 循环 i 从 0 到 13
    for (i = 0; i<= 13; i++)    // 备份棋盘
        // 将 b[i] 的值赋给 g[i]
        g[i] = b[i];
    // 循环 j 从 7 到 12
    for (j = 7; j <= 12) {
        // 如果 b[j] 等于 0，则继续下一次循环
        if (b[j] == 0)
            continue;
        // 将 0 赋给 q
        q = 0;
        // 将 j 的值赋给 m
        m = j;
        // 调用 adjust_board 函数
        adjust_board();
        // 循环 i 从 0 到 5
        for (i = 0; i <= 5; i++) {
            // 如果 b[i] 等于 0，则继续下一次循环
            if (b[i] == 0)
                continue;
            // 将 b[i] 加上 i 的值赋给 l
            l = b[i] + i;
            // 将 0 赋给 r
            r = 0;
            // 当 l 大于 13 时循环
            while (l > 13) {
                // 将 l 减去 14
                l -= 14;
                // 将 1 赋给 r
                r = 1;
            }
            // 如果 b[l] 等于 0
            if (b[l] == 0) {
                // 如果 l 不等于 6 且 l 不等于 13
                if (l != 6 && l != 13)
                    // 将 b[12 - l] 加上 r 赋给 r
                    r = b[12 - l] + r;
            }
            // 如果 r 大于 q，则将 r 赋给 q
            if (r > q)
                q = r;
        }
        // 将 b[13] 减去 b[6] 减去 q 的值赋给 q
        q = b[13] - b[6] - q;
        // 如果 c 小于 8
        if (c < 8) {
            // 将 j 的值赋给 k
            k = j;
            // 如果 k 大于 6，则将 k 减去 7
            if (k > 6)
                k -= 7;
            // 循环 i 从 0 到 n - 1
            for (i = 0; i <= n - 1; i++) {
                // 如果 f[n] * 6 + k 等于 Math.floor(f[i] / Math.pow(7 - c, 6) + 0.1)
                if (f[n] * 6 + k == Math.floor(f[i] / Math.pow(7 - c, 6) + 0.1))
                    // 将 q 减去 2
                    q -= 2;
            }
        }
        // 循环 i 从 0 到 13    // 恢复棋盘
        for (i = 0; i <= 13; i++)
            // 将 g[i] 的值赋给 b[i]
            b[i] = g[i];
        // 如果 q 大于等于 d
        if (q >= d) {
            // 将 j 的值赋给 a
            a = j;
            // 将 q 的值赋给 d
            d = q;
        }
    }
    // 将 a 的值赋给 m
    m = a;
    // 打印 m 减去 6
    print(m - 6);
    // 调用 do_move 函数
    do_move();
}

// 主程序
async function main()
{
    # 进入游戏主循环
    while (1) {
        # 打印两个空行
        print("\n");
        print("\n");
        # 初始化变量 e 为 0
        e = 0;
        # 将数组 b 的前 13 个元素全部赋值为 3
        for (i = 0; i <= 12; i++)
            b[i] = 3;

        # 初始化变量 c 为 0
        c = 0;
        # 初始化数组 f 的第 n 个元素为 0
        f[n] = 0;
        # 初始化数组 b 的第 13 个元素为 0
        b[13] = 0;
        # 初始化数组 b 的第 6 个元素为 0
        b[6] = 0;

        # 进入游戏内部循环
        while (1) {
            # 显示游戏板
            show_board();
            # 打印提示信息
            print("YOUR MOVE");
            # 进入循环，等待玩家输入合法的移动
            while (1) {
                # 从输入中获取玩家的移动
                m = parseInt(await input());
                # 如果玩家的移动小于 7
                if (m < 7) {
                    # 如果玩家的移动大于 0
                    if (m > 0) {
                        # 将玩家的移动减一
                        m--;
                        # 如果数组 b 中对应位置的值不为 0，则跳出循环
                        if (b[m] != 0)
                            break;
                    }
                }
                # 打印非法移动的提示信息
                print("ILLEGAL MOVE\n");
                print("AGAIN");
            }
            # 初始化变量 h 为 6
            h = 6;
            # 执行玩家的移动
            do_move();
            # 显示游戏板
            show_board();
            # 如果 e 等于 0，则跳出循环
            if (e == 0)
                break;
            # 如果玩家的移动等于 h
            if (m == h) {
                # 打印提示信息
                print("AGAIN");
                # 进入循环，等待玩家输入合法的移动
                while (1) {
                    # 从输入中获取玩家的移动
                    m = parseInt(await input());
                    # 如果玩家的移动小于 7
                    if (m < 7) {
                        # 如果玩家的移动大于 0
                        if (m > 0) {
                            # 将玩家的移动减一
                            m--;
                            # 如果数组 b 中对应位置的值不为 0，则跳出循环
                            if (b[m] != 0)
                                break;
                        }
                    }
                    # 打印非法移动的提示信息
                    print("ILLEGAL MOVE\n");
                    print("AGAIN");
                }
                # 初始化变量 h 为 6
                h = 6;
                # 执行玩家的移动
                do_move();
                # 显示游戏板
                show_board();
            }
            # 如果 e 等于 0，则跳出循环
            if (e == 0)
                break;
            # 打印提示信息
            print("MY MOVE IS ");
            # 计算电脑的移动
            computer_move();
            # 如果 e 等于 0，则跳出循环
            if (e == 0)
                break;
            # 如果玩家的移动等于 h
            if (m == h) {
                # 打印逗号
                print(",");
                # 计算电脑的移动
                computer_move();
            }
            # 如果 e 等于 0，则跳出循环
            if (e == 0)
                break;
        }
        # 打印两个空行
        print("\n");
        # 打印游戏结束的提示信息
        print("GAME OVER\n");
        # 计算玩家和电脑的得分差
        d = b[6] - b[13];
        # 如果得分差小于 0，则打印电脑获胜的提示信息
        if (d < 0)
            print("I WIN BY " + -d + " POINTS\n");
        # 如果得分差等于 0，则打印平局的提示信息
        else if (d == 0) {
            n++;
            print("DRAWN GAME\n");
        # 如果得分差大于 0，则打印玩家获胜的提示信息
        } else {
            n++;
            print("YOU WIN BY " + d + " POINTS\n");
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```