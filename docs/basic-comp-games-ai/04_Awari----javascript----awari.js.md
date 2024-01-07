# `basic-computer-games\04_Awari\javascript\awari.js`

```

// 定义打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    // 声明变量
    var input_element;
    var input_str;

    // 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       // 创建输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       // 初始化输入字符串
                       input_str = undefined;
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 输出标题
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

// 显示数字
function show_number(number)
{
    if (number < 10)
        print("  " + number + " ");
    else
        print(" " + number + " ");
}

// 显示游戏板
function show_board()
{
    var i;

    print("\n");
    print("   ");
    for (i = 12; i >= 7; i--)
        show_number(b[i]);
    print("\n");
    i = 13;
    show_number(b[i]);
    print("                       " + b[6] + "\n");
    print("   ");
    for (i = 0; i <= 5; i++)
        show_number(b[i]);
    print("\n");
    print("\n");
}

// 执行移动
function do_move()
{
    k = m;
    adjust_board();
    e = 0;
    if (k > 6)
        k -= 7;
    c++;
    if (c < 9)
        f[n] = f[n] * 6 + k
        for (i = 0; i <= 5; i++) {
            if (b[i] != 0) {
                for (i = 7; i <= 12; i++) {
                    if (b[i] != 0) {
                        e = 1;
                        return;
                    }
                }
            }
        }
}

// 调整游戏板
function adjust_board()
{
    p = b[m];
    b[m] = 0;
    while (p >= 1) {
        m++;
        if (m > 13)
            m -= 14;
        b[m]++;
        p--;
    }
    if (b[m] == 1) {
        if (m != 6 && m != 13) {
            if (b[12 - m] != 0) {
                b[h] += b[12 - m] + 1;
                b[m] = 0;
                b[12 - m] = 0;
            }
        }
    }
}

// 计算机移动
function computer_move()
{
    // 初始化变量
    d = -99;
    h = 13;
    for (i = 0; i<= 13; i++)	// 备份游戏板
        g[i] = b[i];
    for (j = 7; j <= 12; j++) {
        if (b[j] == 0)
            continue;
        q = 0;
        m = j;
        adjust_board();
        for (i = 0; i <= 5; i++) {
            if (b[i] == 0)
                continue;
            l = b[i] + i;
            r = 0;
            while (l > 13) {
                l -= 14;
                r = 1;
            }
            if (b[l] == 0) {
                if (l != 6 && l != 13)
                    r = b[12 - l] + r;
            }
            if (r > q)
                q = r;
        }
        q = b[13] - b[6] - q;
        if (c < 8) {
            k = j;
            if (k > 6)
                k -= 7;
            for (i = 0; i <= n - 1; i++) {
                if (f[n] * 6 + k == Math.floor(f[i] / Math.pow(7 - c, 6) + 0.1))
                    q -= 2;
            }
        }
        for (i = 0; i <= 13; i++)	// 恢复游戏板
            b[i] = g[i];
        if (q >= d) {
            a = j;
            d = q;
        }
    }
    m = a;
    print(m - 6);
    do_move();
}

// 主程序
async function main()
{
    while (1) {
        print("\n");
        print("\n");
        e = 0;
        for (i = 0; i <= 12; i++)
            b[i] = 3;

        c = 0;
        f[n] = 0;
        b[13] = 0;
        b[6] = 0;

        while (1) {
            show_board();
            print("YOUR MOVE");
            while (1) {
                m = parseInt(await input());
                if (m < 7) {
                    if (m > 0) {
                        m--;
                        if (b[m] != 0)
                            break;
                    }
                }
                print("ILLEGAL MOVE\n");
                print("AGAIN");
            }
            h = 6;
            do_move();
            show_board();
            if (e == 0)
                break;
            if (m == h) {
                print("AGAIN");
                while (1) {
                    m = parseInt(await input());
                    if (m < 7) {
                        if (m > 0) {
                            m--;
                            if (b[m] != 0)
                                break;
                        }
                    }
                    print("ILLEGAL MOVE\n");
                    print("AGAIN");
                }
                h = 6;
                do_move();
                show_board();
            }
            if (e == 0)
                break;
            print("MY MOVE IS ");
            computer_move();
            if (e == 0)
                break;
            if (m == h) {
                print(",");
                computer_move();
            }
            if (e == 0)
                break;
        }
        print("\n");
        print("GAME OVER\n");
        d = b[6] - b[13];
        if (d < 0)
            print("I WIN BY " + -d + " POINTS\n");
        else if (d == 0) {
            n++;
            print("DRAWN GAME\n");
        } else {
            n++;
            print("YOU WIN BY " + d + " POINTS\n");
        }
    }
}

// 调用主程序
main();

```