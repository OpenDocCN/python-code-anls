# `04_Awari\javascript\awari.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 INPUT 元素，用于用户输入
// 在页面上打印提示符 "? "
// 设置 INPUT 元素的类型为文本输入类型
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
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回处理后的字符串

print(tab(34) + "AWARI\n");  # 打印以34个空格开头的字符串 "AWARI"，并换行
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印以15个空格开头的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行

n = 0;  # 初始化变量 n 为 0

b = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];  # 创建一个长度为 14 的数组 b，所有元素初始化为 0
g = [0,0,0,0,0,0,0,0,0,0,0,0,0,0];  # 创建一个长度为 14 的数组 g，所有元素初始化为 0
f = [];  # 创建一个空数组 f
for (i = 0; i <= 50; i++) {  # 循环 51 次，i 从 0 到 50
    f[i] = 0;  # 将数组 f 的第 i 个元素初始化为 0
}

function show_number(number)  # 定义一个名为 show_number 的函数，接受一个参数 number
{
    if (number < 10)  # 如果参数 number 小于 10
        print("  " + number + " ");  # 打印两个空格、参数 number、一个空格
    else
        print(" " + number + " ");  # 如果条件不满足，打印数字并在前后加上空格

function show_board()
{
    var i;

    print("\n");  # 打印换行
    print("   ");  # 打印三个空格
    for (i = 12; i >= 7; i--)  # 循环遍历数组元素
        show_number(b[i]);  # 调用函数显示数组元素
    print("\n");  # 打印换行
    i = 13;  # 初始化变量 i
    show_number(b[i]);  # 调用函数显示数组元素
    print("                       " + b[6] + "\n");  # 打印固定字符串和数组元素，并换行
    print("   ");  # 打印三个空格
    for (i = 0; i <= 5; i++)  # 循环遍历数组元素
        show_number(b[i]);  # 调用函数显示数组元素
    print("\n");  # 打印换行
}
    print("\n");  # 打印一个换行符

}

function do_move()  # 定义一个名为 do_move 的函数
{
    k = m;  # 将变量 m 的值赋给变量 k
    adjust_board();  # 调用 adjust_board 函数
    e = 0;  # 将变量 e 的值设为 0
    if (k > 6)  # 如果 k 大于 6
        k -= 7;  # 则将 k 减去 7
    c++;  # 变量 c 的值加 1
    if (c < 9)  # 如果 c 小于 9
        f[n] = f[n] * 6 + k  # 计算 f[n] 的新值
        for (i = 0; i <= 5; i++) {  # 循环 i 从 0 到 5
            if (b[i] != 0) {  # 如果 b[i] 不等于 0
                for (i = 7; i <= 12; i++) {  # 再次循环 i 从 7 到 12
                    if (b[i] != 0) {  # 如果 b[i] 不等于 0
                        e = 1;  # 将变量 e 的值设为 1
                        return;  # 返回
                    }
function adjust_board()
{
    p = b[m];  // 将变量 b[m] 的值赋给变量 p
    b[m] = 0;  // 将变量 b[m] 的值设为 0
    while (p >= 1) {  // 当变量 p 大于等于 1 时执行循环
        m++;  // 变量 m 自增 1
        if (m > 13)  // 如果变量 m 大于 13
            m -= 14;  // 则将变量 m 减去 14
        b[m]++;  // 变量 b[m] 自增 1
        p--;  // 变量 p 自减 1
    }
    if (b[m] == 1) {  // 如果变量 b[m] 的值等于 1
        if (m != 6 && m != 13) {  // 如果变量 m 不等于 6 且不等于 13
            if (b[12 - m] != 0) {  // 如果变量 b[12 - m] 的值不等于 0
                b[h] += b[12 - m] + 1;  // 变量 b[h] 的值加上 b[12 - m] 的值再加 1
                b[m] = 0;  // 将数组 b 中索引为 m 的元素赋值为 0
                b[12 - m] = 0;  // 将数组 b 中索引为 12 - m 的元素赋值为 0
            }
        }
    }
}

function computer_move()
{
    d = -99;  // 将变量 d 赋值为 -99
    h = 13;  // 将变量 h 赋值为 13
    for (i = 0; i<= 13; i++)	// 备份棋盘
        g[i] = b[i];  // 将数组 b 中的元素复制到数组 g 中
    for (j = 7; j <= 12; j++) {  // 循环遍历数组 b 中索引从 7 到 12 的元素
        if (b[j] == 0)  // 如果数组 b 中索引为 j 的元素为 0
            continue;  // 跳过当前循环，继续下一次循环
        q = 0;  // 将变量 q 赋值为 0
        m = j;  // 将变量 m 赋值为 j
        adjust_board();  // 调用 adjust_board 函数
        for (i = 0; i <= 5; i++) {  // 循环遍历 i 从 0 到 5
            # 如果数组中当前位置的值为0，则跳过当前循环，继续下一次循环
            if (b[i] == 0)
                continue;
            # 计算l的值，为数组中当前位置的值加上i
            l = b[i] + i;
            # 初始化r为0
            r = 0;
            # 当l大于13时，执行循环
            while (l > 13) {
                l -= 14;
                r = 1;
            }
            # 如果数组中l位置的值为0
            if (b[l] == 0) {
                # 如果l不等于6且不等于13
                if (l != 6 && l != 13)
                    # 计算r的值，为12减去l位置的值再加上r
                    r = b[12 - l] + r;
            }
            # 如果r大于q
            if (r > q)
                # 将q的值更新为r
                q = r;
        }
        # 计算q的值，为数组中13位置的值减去数组中6位置的值再减去q
        q = b[13] - b[6] - q;
        # 如果c小于8
        if (c < 8) {
            # 将k的值更新为j
            k = j;
            # 如果k大于6
            if (k > 6)
                # 将k的值更新为k减去7
                k -= 7;
            for (i = 0; i <= n - 1; i++) {  // 循环遍历数组中的元素
                if (f[n] * 6 + k == Math.floor(f[i] / Math.pow(7 - c, 6) + 0.1))  // 如果条件成立
                    q -= 2;  // 对变量 q 进行减法操作
            }
        }
        for (i = 0; i <= 13; i++)	// 恢复棋盘状态
            b[i] = g[i];  // 将数组 g 的值赋给数组 b
        if (q >= d) {  // 如果条件成立
            a = j;  // 将 j 赋给 a
            d = q;  // 将 q 赋给 d
        }
    }
    m = a;  // 将 a 赋给 m
    print(m - 6);  // 打印 m 减去 6 的值
    do_move();  // 调用 do_move 函数
}

// 主程序
async function main()
{
    while (1) {  # 进入无限循环
        print("\n");  # 打印换行
        print("\n");  # 再次打印换行
        e = 0;  # 初始化变量 e 为 0
        for (i = 0; i <= 12; i++)  # 进入循环，i 从 0 到 12
            b[i] = 3;  # 将数组 b 的前 13 个元素赋值为 3

        c = 0;  # 初始化变量 c 为 0
        f[n] = 0;  # 初始化数组 f 的第 n 个元素为 0
        b[13] = 0;  # 初始化数组 b 的第 13 个元素为 0
        b[6] = 0;  # 初始化数组 b 的第 6 个元素为 0

        while (1) {  # 进入内部无限循环
            show_board();  # 调用函数显示游戏板
            print("YOUR MOVE");  # 打印提示信息
            while (1) {  # 进入内部无限循环
                m = parseInt(await input());  # 从输入中获取整数并赋值给变量 m
                if (m < 7) {  # 如果 m 小于 7
                    if (m > 0) {  # 且 m 大于 0
                        m--;  # m 减 1
# 如果数组 b 的第 m 个元素不等于 0，则跳出循环
                        if (b[m] != 0)
                            break;
                    }
                }
                # 打印 "ILLEGAL MOVE" 字符串
                print("ILLEGAL MOVE\n");
                # 打印 "AGAIN" 字符串
                print("AGAIN");
            }
            # 将 h 设为 6
            h = 6;
            # 执行 do_move() 函数
            do_move();
            # 执行 show_board() 函数
            show_board();
            # 如果 e 等于 0，则跳出循环
            if (e == 0)
                break;
            # 如果 m 等于 h，则执行以下操作
            if (m == h) {
                # 打印 "AGAIN" 字符串
                print("AGAIN");
                # 进入无限循环
                while (1) {
                    # 将用户输入的值转换为整数并赋给 m
                    m = parseInt(await input());
                    # 如果 m 小于 7，则执行以下操作
                    if (m < 7) {
                        # 如果 m 大于 0，则执行以下操作
                        if (m > 0) {
                            # 将 m 减 1
                            m--;
                            # 如果数组 b 的第 m 个元素不等于 0，则执行以下操作
                            if (b[m] != 0)
                break;  # 结束当前循环，跳出循环体
        }
    }
    print("ILLEGAL MOVE\n");  # 打印消息提示玩家移动非法
    print("AGAIN");  # 打印消息提示玩家再次移动
}
h = 6;  # 将变量h赋值为6
do_move();  # 调用函数执行移动操作
show_board();  # 调用函数展示游戏棋盘
if (e == 0)  # 如果e等于0
    break;  # 结束当前循环，跳出循环体
print("MY MOVE IS ");  # 打印消息提示轮到计算机移动
computer_move();  # 调用函数执行计算机移动
if (e == 0)  # 如果e等于0
    break;  # 结束当前循环，跳出循环体
if (m == h) {  # 如果m等于h
    print(",");  # 打印逗号
    computer_move();  # 调用函数执行计算机移动
}
            if (e == 0)  # 如果 e 等于 0
                break;  # 跳出循环
        }  # 结束循环
        print("\n");  # 打印空行
        print("GAME OVER\n");  # 打印游戏结束提示
        d = b[6] - b[13];  # 计算数组 b 中第6个元素和第13个元素的差，赋值给变量 d
        if (d < 0)  # 如果 d 小于 0
            print("I WIN BY " + -d + " POINTS\n");  # 打印“我以 x 分获胜”的提示，其中 x 为 d 的绝对值
        else if (d == 0) {  # 否则如果 d 等于 0
            n++;  # 变量 n 自增1
            print("DRAWN GAME\n");  # 打印“平局”的提示
        } else {  # 否则
            n++;  # 变量 n 自增1
            print("YOU WIN BY " + d + " POINTS\n");  # 打印“你以 x 分获胜”的提示，其中 x 为 d
        }
    }  # 结束循环
}

main();  # 调用主函数
```