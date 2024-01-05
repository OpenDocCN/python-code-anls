# `02_Amazing\javascript\amazing.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 input 元素，用于用户输入
// 在页面上打印提示符 "? "
// 设置 input 元素的类型为文本输入类型
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

print(tab(28) + "AMAZING PROGRAM\n");  # 打印带有缩进的字符串
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印带有缩进的字符串
print("\n");  # 打印空行
print("\n");  # 打印空行
print("\n");  # 打印空行
print("FOR EXAMPLE TYPE 10,10 AND PRESS ENTER\n");  # 打印提示信息
print("\n");  # 打印空行

// Main program
async function main()
{
    while (1) {  # 进入循环
        print("WHAT ARE YOUR WIDTH AND LENGTH");  # 打印提示信息
        a = await input();  # 等待用户输入
        h = parseInt(a);  # 将输入转换为整数并赋值给变量h
        v2 = parseInt(a.substr(a.indexOf(",") + 1));  # 从输入中提取逗号后的部分并转换为整数赋值给变量v2
        # 如果 h 大于 1 并且 v2 大于 1，则跳出循环
        if (h > 1 && v2 > 1)
            break;
        # 打印提示信息
        print("MEANINGLESS DIMENSIONS.  TRY AGAIN.\n");
    }
    # 初始化 w 和 v 为空数组
    w = [];
    v = [];
    # 循环遍历 h 次
    for (i = 1; i <= h; i++) {
        # 初始化 w[i] 和 v[i] 为空数组
        w[i] = [];
        v[i] = [];
        # 循环遍历 v2 次
        for (j = 1; j <= v2; j++) {
            # 将 w[i][j] 和 v[i][j] 初始化为 0
            w[i][j] = 0;
            v[i][j] = 0;
        }
    }
    # 打印空行
    print("\n");
    print("\n");
    print("\n");
    print("\n");
    # 将 q 和 z 初始化为 0
    q = 0;
    z = 0;
    x = Math.floor(Math.random() * h + 1);  # 生成一个随机数x，范围在1到h之间
    for (i = 1; i <= h; i++) {  # 循环h次
        if (i == x)  # 如果i等于x
            print(".  ");  # 打印".  "
        else
            print(".--");  # 否则打印".--"
    }
    print(".\n");  # 打印".\n"
    c = 1;  # 初始化变量c为1
    w[x][1] = c;  # 将c赋值给数组w的第x个元素的第一个位置
    c++;  # c自增1
    r = x;  # 变量r赋值为x
    s = 1;  # 变量s赋值为1
    entry = 0;  # 初始化变量entry为0
    while (1) {  # 进入无限循环
        if (entry == 2) {  # 如果entry等于2
            do {  # 执行以下操作
                if (r < h) {  # 如果r小于h
                    r++;  # r自增1
                } else if (s < v2) {  # 否则如果s小于v2
                    r = 1;  // 初始化 r 为 1
                    s++;    // s 自增
                } else {
                    r = 1;  // 初始化 r 为 1
                    s = 1;  // 初始化 s 为 1
                }
            } while (w[r][s] == 0) ;  // 当 w[r][s] 等于 0 时执行循环
        }
        if (entry == 0 && r - 1 > 0 && w[r - 1][s] == 0) {	// 是否可以向左移动？
            if (s - 1 > 0 && w[r][s - 1] == 0) {	// 是否可以向上移动？
                if (r < h && w[r + 1][s] == 0) {	// 是否可以向右移动？
                    // 选择向左/向上/向右移动
                    x = Math.floor(Math.random() * 3 + 1);  // 生成 1-3 之间的随机整数
                } else if (s < v2) {
                    if (w[r][s + 1] == 0) {	// 是否可以向下移动？
                        // 选择向左/向上/向下移动
                        x = Math.floor(Math.random() * 3 + 1);  // 生成 1-3 之间的随机整数
                        if (x == 3)
                            x = 4;  // 如果 x 等于 3，则将其赋值为 4
                    } else {
// 生成一个 1 到 3 之间的随机整数
x = Math.floor(Math.random() * 3 + 1);
// 如果 z 等于 0
if (z == 0) {
    // 如果 r 小于 h 并且 w[r + 1][s] 等于 0
    if (r < h && w[r + 1][s] == 0) {
        // 如果 s 小于 v2
        if (s < v2) {
            // 如果 w[r][s + 1] 等于 0
            if (w[r][s + 1] == 0) {
                // 生成一个 1 到 3 之间的随机整数
                x = Math.floor(Math.random() * 3 + 1);
            } else {
                // 生成一个 1 到 2 之间的随机整数
                x = Math.floor(Math.random() * 2 + 1);
            }
            // 如果 x 大于等于 2
            if (x >= 2) {
                // x 加 1
                x++;
            }
        } else {
            // 生成一个 1 到 2 之间的随机整数
            x = Math.floor(Math.random() * 2 + 1);
        }
    } else if (z == 1) {
        // 生成一个 1 到 2 之间的随机整数
        x = Math.floor(Math.random() * 2 + 1);
    } else {
        // q 赋值为 1
        q = 1;
        // 生成一个 1 到 3 之间的随机整数
        x = Math.floor(Math.random() * 3 + 1);
        // 如果 x 等于 3
        if (x == 3) {
            // x 赋值为 4
            x = 4;
        }
    }
}
            } else if (z == 1) {  # 如果 z 等于 1
                x = Math.floor(Math.random() * 2 + 1);  # 生成一个 1 或 2 的随机数并赋值给 x
                if (x >= 2)  # 如果 x 大于等于 2
                    x++;  # x 自增 1
            } else {  # 否则
                q = 1;  # 将 q 赋值为 1
                x = Math.floor(Math.random() * 3 + 1);  # 生成一个 1 到 3 的随机数并赋值给 x
                if (x >= 2)  # 如果 x 大于等于 2
                    x++;  # x 自增 1
            } else if (s < v2) {  # 否则如果 s 小于 v2
                if (w[r][s + 1] == 0) {  # 如果 w[r][s + 1] 等于 0
                    // Choose left/down  # 选择左/下方向
                    x = Math.floor(Math.random() * 2 + 1);  # 生成一个 1 或 2 的随机数并赋值给 x
                    if (x == 2)  # 如果 x 等于 2
                        x = 4;  # 将 x 赋值为 4
                } else {  # 否则
                    x = 1;  # 将 x 赋值为 1
                }
            } else if (z == 1) {  # 否则如果 z 等于 1
                x = 1;  # 初始化变量 x 为 1
            } else {
                q = 1;  # 初始化变量 q 为 1
                x = Math.floor(Math.random() * 2 + 1);  # 生成一个 1 或 2 的随机数并赋值给 x
                if (x == 2)  # 如果 x 等于 2
                    x = 4;  # 将 x 的值改为 4
            }
        } else if (s - 1 > 0 && w[r][s - 1] == 0) {  # 如果可以向上移动
            if (r < h && w[r + 1][s] == 0) {  # 如果可以向下移动
                if (s < v2) {  # 如果当前位置在 v2 之前
                    if (w[r][s + 1] == 0)  # 如果可以向右移动
                        x = Math.floor(Math.random() * 3 + 2);  # 生成一个 2、3 或 4 的随机数并赋值给 x
                    else
                        x = Math.floor(Math.random() * 2 + 2);  # 生成一个 2 或 3 的随机数并赋值给 x
                } else if (z == 1) {  # 如果 z 等于 1
                    x = Math.floor(Math.random() * 2 + 2);  # 生成一个 2 或 3 的随机数并赋值给 x
                } else {
                    q = 1;  # 初始化变量 q 为 1
                    x = Math.floor(Math.random() * 3 + 2);  # 生成一个 2、3 或 4 的随机数并赋值给 x
                }
            } else if (s < v2) {  # 如果当前列小于目标列
                if (w[r][s + 1] == 0) {  # 如果右侧格子的值为0
                    x = Math.floor(Math.random() * 2 + 2);  # 随机生成2或3
                    if (x == 3)  # 如果随机数为3
                        x = 4;  # 将x设为4
                } else {  # 如果右侧格子的值不为0
                    x = 2;  # 将x设为2
                }
            } else if (z == 1) {  # 如果z为1
                x = 2;  # 将x设为2
            } else {  # 其他情况
                q = 1;  # 将q设为1
                x = Math.floor(Math.random() * 2 + 2);  # 随机生成2或3
                if (x == 3)  # 如果随机数为3
                    x = 4;  # 将x设为4
            }
        } else if (r < h && w[r + 1][s] == 0) {  # 如果当前行小于目标行且下方格子的值为0
            if (s < v2) {  # 如果当前列小于目标列
                if (w[r][s + 1] == 0)  # 如果右侧格子的值为0
                    x = Math.floor(Math.random() * 2 + 3);  # 随机生成3或4
                else
                    x = 3;  // 如果z不等于1，x赋值为3
            } else if (z == 1) {
                x = 3;  // 如果z等于1，x赋值为3
            } else {
                q = 1;  // 否则，q赋值为1
                x = Math.floor(Math.random() * 2 + 3);  // x赋值为一个随机数，取值范围为3到4
            }
        } else if (s < v2) {
            if (w[r][s + 1] == 0) 	// 是否可以向下移动？
                x = 4;  // 如果可以，x赋值为4
            else {
                entry = 2;	// 被阻挡了！
                continue;  // 继续下一次循环
            }
        } else if (z == 1) {
            entry = 2;	// 被阻挡了！
            continue;  // 继续下一次循环
        } else {
            q = 1;  // 否则，q赋值为1
            x = 4;  // 初始化变量 x 为 4
        }
        if (x == 1) {   // 如果 x 等于 1，表示向左移动
            w[r - 1][s] = c;  // 将当前位置的值设置为 c
            c++;  // c 自增
            v[r - 1][s] = 2;  // 标记向左移动
            r--;  // 行数减一
            if (c == h * v2 + 1)  // 如果 c 等于 h * v2 + 1，表示已经完成赋值
                break;  // 跳出循环
            q = 0;  // 重置 q 为 0
            entry = 0;  // 重置 entry 为 0
        } else if (x == 2) {  // 如果 x 等于 2，表示向上移动
            w[r][s - 1] = c;  // 将当前位置的值设置为 c
            c++;  // c 自增
            v[r][s - 1] = 1;  // 标记向上移动
            s--;  // 列数减一
            if (c == h * v2 + 1)  // 如果 c 等于 h * v2 + 1，表示已经完成赋值
                break;  // 跳出循环
            q = 0;  // 重置 q 为 0
            entry = 0;  // 重置 entry 为 0
        } else if (x == 3) {	// Right
            w[r + 1][s] = c;  // 在迷宫中向右移动一步，并标记该位置的值为c
            c++;  // c自增1
            if (v[r][s] == 0)  // 如果当前位置的值为0
                v[r][s] = 2;  // 将当前位置的值标记为2
            else
                v[r][s] = 3;  // 否则将当前位置的值标记为3
            r++;  // 行数自增1，向下移动一格
            if (c == h * v2 + 1)  // 如果c等于h * v2 + 1
                break;  // 跳出循环
            entry = 1;  // entry赋值为1
        } else if (x == 4) {	// Down
            if (q != 1) {	// 只有当不被阻挡时
                w[r][s + 1] = c;  // 在迷宫中向下移动一步，并标记该位置的值为c
                c++;  // c自增1
                if (v[r][s] == 0)  // 如果当前位置的值为0
                    v[r][s] = 1;  // 将当前位置的值标记为1
                else
                    v[r][s] = 3;  // 否则将当前位置的值标记为3
                s++;  // 列数自增1，向右移动一格
                # 如果条件成立，跳出循环
                if (c == h * v2 + 1)
                    break;
                # 重置 entry 为 0
                entry = 0;
            # 如果条件不成立
            } else {
                # 设置 z 为 1
                z = 1;
                # 如果 v[r][s] 等于 0
                if (v[r][s] == 0) {
                    # 将 v[r][s] 设置为 1
                    v[r][s] = 1;
                    # 设置 q 为 0
                    q = 0;
                    # 设置 r 为 1
                    r = 1;
                    # 设置 s 为 1
                    s = 1;
                    # 当 w[r][s] 等于 0 时执行循环
                    while (w[r][s] == 0) {
                        # 如果 r 小于 h
                        if (r < h) {
                            # r 自增
                            r++;
                        # 如果 s 小于 v2
                        } else if (s < v2) {
                            # r 重置为 1
                            r = 1;
                            # s 自增
                            s++;
                        # 否则
                        } else {
                            # r 重置为 1
                            r = 1;
                            # s 重置为 1
                            s = 1;
                        }
                    }
                    entry = 0;  # 重置 entry 变量为 0
                } else {
                    v[r][s] = 3;  # 将二维数组 v 中索引为 (r, s) 的元素赋值为 3
                    q = 0;  # 将 q 变量重置为 0
                    entry = 2;  # 将 entry 变量设置为 2
                }
            }
        }
    }
    for (j = 1; j <= v2; j++) {  # 遍历 j 从 1 到 v2
        str = "I";  # 初始化 str 变量为 "I"
        for (i = 1; i <= h; i++) {  # 遍历 i 从 1 到 h
            if (v[i][j] < 2)  # 如果二维数组 v 中索引为 (i, j) 的元素小于 2
                str += "  I";  # 在 str 后面添加 "  I"
            else
                str += "   ";  # 在 str 后面添加 "   "
        }
        print(str + "\n");  # 打印 str 并换行
        str = "";  # 重置 str 变量为空字符串
        for (i = 1; i <= h; i++) {  // 从第1行开始循环到第h行
            if (v[i][j] == 0 || v[i][j] == 2)  // 如果v[i][j]的值为0或2
                str += ":--";  // 在str后面添加":--"
            else
                str += ":  ";  // 否则在str后面添加":  "
        }
        print(str + ".\n");  // 打印str并换行
    }
// 如果你想看到访问单元格的顺序
//    for (j = 1; j <= v2; j++) {  // 从第1列开始循环到第v2列
//        str = "I";  // 将str设置为"I"
//        for (i = 1; i <= h; i++) {  // 从第1行开始循环到第h行
//            str += w[i][j] + " ";  // 在str后面添加w[i][j]的值和一个空格
//        }
//        print(str + "\n");  // 打印str并换行
//    }
}

main();  // 调用main函数
```