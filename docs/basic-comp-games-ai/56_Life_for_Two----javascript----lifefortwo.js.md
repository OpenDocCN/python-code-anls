# `basic-computer-games\56_Life_for_Two\javascript\lifefortwo.js`

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
                                                      // 当按下回车键时，获取输入框的值
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值
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

// 初始化变量
var na = [];
var ka = [, 3,102,103,120,130,121,112,111,12,
          21,30,1020,1030,1011,1021,1003,1002,1012];
var aa = [,-1,0,1,0,0,-1,0,1,-1,-1,1,-1,-1,1,1,1];
var xa = [];
var ya = [];
var j;
var k;
var m2;
var m3;

// 定义一个显示数据的函数
function show_data()
{
    // 初始化变量
    k = 0;
    m2 = 0;
    m3 = 0;
    # 循环变量 j 从 0 到 6
    for (j = 0; j <= 6; j++) {
        # 打印换行
        print("\n");
        # 循环变量 k 从 0 到 6
        for (k = 0; k <= 6; k++) {
            # 如果 j 等于 0 或者 j 等于 6
            if (j == 0 || j == 6) {
                # 如果 k 等于 6
                if (k == 6)
                    # 打印 0
                    print(" 0 ");
                else
                    # 打印空格和 k 的值
                    print(" " + k + " ");
            } else if (k == 0 || k == 6) {
                # 如果 k 等于 0 或者 k 等于 6
                if (j == 6)
                    # 打印 0 并换行
                    print(" 0\n");
                else
                    # 打印空格和 j 的值
                    print(" " + j + " ");
            } else {
                # 如果 na[j][k] 大于等于 3
                if (na[j][k] >= 3) {
                    # 循环变量 o1 从 1 到 18
                    for (o1 = 1; o1 <= 18; o1++) {
                        # 如果 na[j][k] 等于 ka[o1]
                        if (na[j][k] == ka[o1])
                            # 跳出循环
                            break;
                    }
                    # 如果 o1 小于等于 18
                    if (o1 <= 18) {
                        # 如果 o1 小于等于 9
                        if (o1 <= 9) {
                            # 将 na[j][k] 赋值为 100
                            na[j][k] = 100;
                            # m2 自增
                            m2++;
                            # 打印 *
                            print(" * ");
                        } else {
                            # 将 na[j][k] 赋值为 1000
                            na[j][k] = 1000;
                            # m3 自增
                            m3++;
                            # 打印 #
                            print(" # ");
                        }
                    } else {
                        # 将 na[j][k] 赋值为 0
                        na[j][k] = 0;
                        # 打印三个空格
                        print("   ");
                    }
                } else {
                    # 将 na[j][k] 赋值为 0
                    na[j][k] = 0;
                    # 打印三个空格
                    print("   ");
                }
            }
        }
    }
// 处理游戏板的函数
function process_board()
{
    // 循环遍历游戏板的每一个位置
    for (j = 1; j <= 5; j++) {
        for (k = 1; k <= 5; k++) {
            // 如果当前位置的值大于99
            if (na[j][k] > 99) {
                // 设置变量b为1
                b = 1;
                // 如果当前位置的值大于999，将变量b设置为10
                if (na[j][k] > 999)
                    b = 10;
                // 遍历固定的15个偏移量，更新周围位置的值
                for (o1 = 1; o1 <= 15; o1 += 2) {
                    na[j + aa[o1]][k + aa[o1 + 1]] = na[j + aa[o1]][k + aa[o1 + 1]] + b;
                }
            }
        }
    }
    // 调用显示数据的函数
    show_data();
}

// 主程序
async function main()
{
    // 打印游戏名称和信息
    print(tab(33) + "LIFE2\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print(tab(10) + "U.B. LIFE GAME\n");
    // 初始化变量m2和m3
    m2 = 0;
    m3 = 0;
    // 初始化二维数组na
    for (j = 0; j <= 6; j++) {
        na[j] = [];
        for (k = 0; k <= 6; k++)
            na[j][k] = 0;
    }
    // 循环两次，为两个玩家设置初始位置
    for (b = 1; b <= 2; b++) {
        // 根据玩家设置初始值p1
        p1 = (b == 2) ? 30 : 3;
        print("\n");
        print("PLAYER " + b + " - 3 LIVE PIECES.\n");
        // 循环三次，为每个玩家设置三个初始位置
        for (k1 = 1; k1 <= 3; k1++) {
            // 循环直到输入合法的坐标
            while (1) {
                print("X,Y\n");
                str = await input();
                ya[b] = parseInt(str);
                xa[b] = parseInt(str.substr(str.indexOf(",") + 1));
                if (xa[b] > 0 && xa[b] < 6 && ya[b] > 0 && ya[b] < 5 && na[xa[b]][ya[b]] == 0)
                    break;
                print("ILLEGAL COORDS. RETYPE\n");
            }
            // 如果不是第一个玩家，检查是否与第一个玩家的位置重合
            if (b != 1) {
                if (xa[1] == xa[2] && ya[1] == ya[2]) {
                    print("SAME COORD.  SET TO 0\n");
                    na[xa[b] + 1][ya[b] + 1] = 0;
                    b = 99;
                }
            }
            // 设置玩家的初始位置
            na[xa[b]][ya[b]] = p1;
        }
    }
    // 调用显示数据的函数
    show_data();
}
    # 进入游戏循环
    while (1) {
        # 打印空行
        print("\n");
        # 处理游戏板
        process_board();
        # 如果玩家2和玩家3都没有获胜
        if (m2 == 0 && m3 == 0) {
            # 打印空行
            print("\n");
            # 打印平局信息
            print("A DRAW\n");
            # 退出游戏循环
            break;
        }
        # 如果玩家3没有获胜
        if (m3 == 0) {
            # 打印空行
            print("\n");
            # 打印玩家1获胜信息
            print("PLAYER 1 IS THE WINNER\n");
            # 退出游戏循环
            break;
        }
        # 如果玩家2没有获胜
        if (m2 == 0) {
            # 打印空行
            print("\n");
            # 打印玩家2获胜信息
            print("PLAYER 2 IS THE WINNER\n");
            # 退出游戏循环
            break;
        }
        # 遍历玩家1和玩家2
        for (b = 1; b <= 2; b++) {
            # 打印空行
            print("\n");
            print("\n");
            # 打印玩家信息
            print("PLAYER " + b + " ");
            # 玩家输入坐标
            while (1) {
                # 打印提示信息
                print("X,Y\n");
                # 等待输入
                str = await input();
                # 解析输入的坐标
                ya[b] = parseInt(str);
                xa[b] = parseInt(str.substr(str.indexOf(",") + 1));
                # 如果坐标合法，跳出循环
                if (xa[b] > 0 && xa[b] < 6 && ya[b] > 0 && ya[b] < 5 && na[xa[b]][ya[b]] == 0)
                    break;
                # 打印非法坐标提示
                print("ILLEGAL COORDS. RETYPE\n");
            }
            # 如果不是玩家1
            if (b != 1) {
                # 如果玩家1和玩家2选择了相同的坐标
                if (xa[1] == xa[2] && ya[1] == ya[2]) {
                    # 打印相同坐标提示
                    print("SAME COORD.  SET TO 0\n");
                    # 设置坐标为0
                    na[xa[b] + 1][ya[b] + 1] = 0;
                    # 退出循环
                    b = 99;
                }
            }
            # 如果b等于99
            if (b == 99)
                # 退出循环
                break;
        }
        # 如果b小于等于2
        if (b <= 2) {
            # 设置玩家1和玩家2的坐标值
            na[x[1]][y[1]] = 100;
            na[x[2]][y[2]] = 1000;
        }
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```