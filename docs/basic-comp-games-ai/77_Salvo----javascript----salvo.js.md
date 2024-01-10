# `basic-computer-games\77_Salvo\javascript\salvo.js`

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
                                                      // 如果按下的是回车键
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析输入的字符串并返回
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定空格数的字符串的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一系列变量
var aa = [];
var ba = [];
var ca = [];
var da = [];
var ea = [];
var fa = [];
var ga = [];
var ha = [];
var ka = [];
var w;
var r3;
var x;
var y;
var v;
var v2;

// 定义一个返回数值的符号函数
function sgn(k)
{
    if (k < 0)
        return -1;
    if (k > 0)
        return 1;
    return 0;
}

// 定义一个函数，根据输入的数值返回计算结果
function fna(k)
{
    return (5 - k) * 3 - 2 * Math.floor(k / 4) + sgn(k - 1) - 1;
}

// 定义一个函数，根据输入的数值返回计算结果
function fnb(k)
{
    return k + Math.floor(k / 4) - sgn(k - 1);
}

// 定义一个生成随机数的函数
function generate_random()
    # 生成一个随机数并向下取整，范围在1到10之间
    x = Math.floor(Math.random() * 10 + 1);
    # 生成一个随机数并向下取整，范围在1到10之间
    y = Math.floor(Math.random() * 10 + 1);
    # 生成一个随机数并向下取整，范围在-1到2之间
    v = Math.floor(3 * Math.random() - 1);
    # 生成一个随机数并向下取整，范围在-1到2之间
    v2 = Math.floor(3 * Math.random() - 1);
// 主程序
async function main()
{
    // 打印 SALVO
    print(tab(33) + "SALVO\n");
    // 打印 CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印三个空行
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量 z8 为 0
    z8 = 0;
    // 初始化数组 ea 和 ha 的值为 -1
    for (w = 1; w <= 12; w++) {
        ea[w] = -1;
        ha[w] = -1;
    }
    // 初始化二维数组 ba 和 ka 的值为 0
    for (x = 1; x <= 10; x++) {
        ba[x] = [];
        ka[x] = [];
        for (y = 1; y <= 10; y++) {
            ba[x][y] = 0;
            ka[x][y] = 0;
        }
    }
    // 初始化数组 fa 和 ga 的值为 0
    for (x = 1; x <= 12; x++) {
        fa[x] = 0;
        ga[x] = 0;
    }
    // 初始化二维数组 aa 的值为 0
    for (x = 1; x <= 10; x++) {
        aa[x] = [];
        for (y = 1; y <= 10; y++) {
            aa[x][y] = 0;
        }
    }
    // 初始化变量 u6 为 0
    u6 = 0;
}
    # 从 4 循环到 1
    for (k = 4; k >= 1; k--) {
        # 生成随机数
        do {
            generate_random();
        } while (v + v2 + v * v2 == 0 || y + v * fnb(k) > 10 || y + v * fnb(k) < 1 || x + v2 * fnb(k) > 10 || x + v2 * fnb(k) < 1) ;
        # u6 自增
        u6++;
        # 如果 u6 大于 25
        if (u6 > 25) {
            # 重置 aa 数组
            for (x = 1; x <= 10; x++) {
                aa[x] = [];
                for (y = 1; y <= 10; y++) {
                    aa[x][y] = 0;
                }
            }
            # 重置 u6 和 k
            u6 = 0;
            k = 5;
            # 继续下一次循环
            continue;
        }
        # 循环计算坐标
        for (z = 0; z <= fnb(k); z++) {
            fa[z + fna(k)] = x + v2 * z;
            ga[z + fna(k)] = y + v * z;
        }
        # 设置 u8
        u8 = fna(k);
        # 如果 u8 小于等于 u8 + fnb(k)
        if (u8 <= u8 + fnb(k)) {
            # 重试标志
            retry = false;
            # 循环检查坐标
            for (z2 = u8; z2 <= u8 + fnb(k); z2++) {
                # 如果 u8 大于等于 2
                if (u8 >= 2) {
                    # 循环检查距离
                    for (z3 = 1; z3 < u8 - 1; z3++) {
                        if (Math.sqrt(Math.pow((fa[z3] - fa[z2]), 2)) + Math.pow((ga[z3] - ga[z2]), 2) < 3.59) {
                            # 设置重试标志
                            retry = true;
                            # 跳出循环
                            break;
                        }
                    }
                    # 如果重试标志为真，跳出循环
                    if (retry)
                        break;
                }
            }
            # 如果重试标志为真，增加 k 并继续下一次循环
            if (retry) {
                k++;
                continue;
            }
        }
        # 设置 aa 数组的值
        for (z = 0; z <= fnb(k); z++) {
            if (k - 1 < 0)
                sk = -1;
            else if (k - 1 > 0)
                sk = 1;
            else
                sk = 0;
            aa[fa[z + u8]][ga[z + u8]] = 0.5 + sk * (k - 1.5);
        }
        # 重置 u6
        u6 = 0;
    }
    # 打印提示信息
    print("ENTER COORDINATES FOR...\n");
    print("BATTLESHIP\n");
    # 循环获取输入坐标
    for (x = 1; x <= 5; x++) {
        str = await input();
        y = parseInt(str);
        z = parseInt(str.substr(str.indexOf(",") + 1));
        ba[y][z] = 3;
    }
    # 打印提示信息
    print("CRUISER\n");
    # 循环获取输入坐标
    for (x = 1; x <= 3; x++) {
        str = await input();
        y = parseInt(str);
        z = parseInt(str.substr(str.indexOf(",") + 1));
        ba[y][z] = 2;
    }
    # 打印字符串 "DESTROYER<A>"
    print("DESTROYER<A>\n");
    # 循环两次
    for (x = 1; x <= 2; x++) {
        # 等待输入并将输入的字符串转换为整数赋值给变量 y
        str = await input();
        y = parseInt(str);
        # 从输入的字符串中找到逗号后的部分并转换为整数赋值给变量 z
        z = parseInt(str.substr(str.indexOf(",") + 1));
        # 在二维数组 ba 中的位置 (y, z) 赋值为 1
        ba[y][z] = 1;
    }
    # 打印字符串 "DESTROYER<B>"
    print("DESTROYER<B>\n");
    # 循环两次
    for (x = 1; x <= 2; x++) {
        # 等待输入并将输入的字符串转换为整数赋值给变量 y
        str = await input();
        y = parseInt(str);
        # 从输入的字符串中找到逗号后的部分并转换为整数赋值给变量 z
        z = parseInt(str.substr(str.indexOf(",") + 1));
        # 在二维数组 ba 中的位置 (y, z) 赋值为 0.5
        ba[y][z] = 0.5;
    }
    # 无限循环
    while (1) {
        # 打印字符串 "DO YOU WANT TO START"
        print("DO YOU WANT TO START");
        # 等待输入并将输入的字符串赋值给变量 js
        js = await input();
        # 如果输入的字符串为 "WHERE ARE YOUR SHIPS?"
        if (js == "WHERE ARE YOUR SHIPS?") {
            # 打印不同类型的船只及其位置
            print("BATTLESHIP\n");
            for (z = 1; z <= 5; z++)
                print(" " + fa[z] + " " + ga[z] + "\n");
            print("CRUISER\n");
            print(" " + fa[6] + " " + ga[6] + "\n");
            print(" " + fa[7] + " " + ga[7] + "\n");
            print(" " + fa[8] + " " + ga[8] + "\n");
            print("DESTROYER<A>\n");
            print(" " + fa[9] + " " + ga[9] + "\n");
            print(" " + fa[10] + " " + ga[10] + "\n");
            print("DESTROYER<B>\n");
            print(" " + fa[11] + " " + ga[11] + "\n");
            print(" " + fa[12] + " " + ga[12] + "\n");
        } else {
            # 结束循环
            break;
        }
    }
    # 变量 c 赋值为 0
    c = 0;
    # 打印字符串 "DO YOU WANT TO SEE MY SHOTS"
    print("DO YOU WANT TO SEE MY SHOTS");
    # 等待输入并将输入的字符串赋值给变量 ks
    ks = await input();
    # 打印换行符
    print("\n");
    # 如果变量 js 不等于 "YES"，则变量 first_time 赋值为 true，否则赋值为 false
    if (js != "YES")
        first_time = true;
    else
        first_time = false;
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```