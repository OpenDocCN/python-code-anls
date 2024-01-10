# `basic-computer-games\06_Banner\javascript\banner.js`

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
                                                      // 获取输入框的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      // 打印换行符
                                                      print("\n");
                                                      // 解析 Promise 对象
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
# 定义一个包含字母和对应的点阵数据的数组
var letters = [" ",0,0,0,0,0,0,0,
               "A",505,37,35,34,35,37,505,
               "G",125,131,258,258,290,163,101,
               "E",512,274,274,274,274,258,258,
               "T",2,2,2,512,2,2,2,
               "W",256,257,129,65,129,257,256,
               "L",512,257,257,257,257,257,257,
               "S",69,139,274,274,274,163,69,
               "O",125,131,258,258,258,131,125,
               "N",512,7,9,17,33,193,512,
               "F",512,18,18,18,18,2,2,
               "K",512,17,17,41,69,131,258,
               "B",512,274,274,274,274,274,239,
               "D",512,258,258,258,258,131,125,
               "H",512,17,17,17,17,17,512,
               "M",512,7,13,25,13,7,512,
               "?",5,3,2,354,18,11,5,
               "U",128,129,257,257,257,129,128,
               "R",512,18,18,50,82,146,271,
               "P",512,18,18,18,18,18,15,
               "Q",125,131,258,258,322,131,381,
               "Y",8,9,17,481,17,9,8,
               "V",64,65,129,257,129,65,64,
               "X",388,69,41,17,41,69,388,
               "Z",386,322,290,274,266,262,260,
               "I",258,258,258,512,258,258,258,
               "C",125,131,258,258,258,131,69,
               "J",65,129,257,257,257,129,128,
               "1",0,0,261,259,512,257,257,
               "2",261,387,322,290,274,267,261,
               "*",69,41,17,512,17,41,69,
               "3",66,130,258,274,266,150,100,
               "4",33,49,41,37,35,512,33,
               "5",160,274,274,274,274,274,226,
               "6",194,291,293,297,305,289,193,
               "7",258,130,66,34,18,10,8,
               "8",69,171,274,274,274,171,69,
               "9",263,138,74,42,26,10,7,
               "=",41,41,41,41,41,41,41,
               "!",1,1,1,384,1,1,1,
               "0",57,69,131,258,131,69,57,
               ".",1,1,129,449,129,1,1];

# 定义三个空数组
f = [];
j = [];
s = [];

# 主程序
async function main()
{
    # 打印提示信息
    print("HORIZONTAL");
    # 将用户输入的值转换为整数并赋给变量x
    x = parseInt(await input());
    # 打印提示信息
    print("VERTICAL");
    # 从输入中获取一个整数并赋值给变量 y
    y = parseInt(await input());
    # 打印 "CENTERED"
    print("CENTERED");
    # 从输入中获取一个字符串并赋值给变量 ls
    ls = await input();
    # 初始化变量 g1 为 0
    g1 = 0;
    # 如果 ls 大于 "P"，则将 g1 赋值为 1
    if (ls > "P")
        g1 = 1;
    # 打印 "CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)"
    print("CHARACTER (TYPE 'ALL' IF YOU WANT CHARACTER BEING PRINTED)");
    # 从输入中获取一个字符串并赋值给变量 ms
    ms = await input();
    # 打印 "STATEMENT"
    print("STATEMENT");
    # 从输入中获取一个字符串并赋值给变量 as
    as = await input();
    # 打印 "SET PAGE"，这意味着准备打印机，只需按 Enter 键
    print("SET PAGE");
    # 从输入中获取一个字符串并赋值给变量 os
    os = await input();

    # 循环遍历字符串 as 的每个字符
    for (t = 0; t < as.length; t++) {
        # 获取字符串 as 中索引为 t 的字符并赋值给变量 ps
        ps = as.substr(t, 1);
        # 循环遍历 letters 数组
        for (o = 0; o < 50 * 8; o += 8) {
            # 如果 letters[o] 等于 ps，则将 s 数组中的元素赋值为 letters[o+1] 到 letters[o+7]
            if (letters[o] == ps) {
                for (u = 1; u <= 7; u++)
                    s[u] = letters[o + u];
                # 跳出循环
                break;
            }
        }
        # 如果 o 等于 50*8，则将 ps 赋值为空格，o 赋值为 0
        if (o == 50 * 8) {
            ps = " ";
            o = 0;
        }
// 打印正在进行的操作
// 如果 o 等于 0
if (o == 0) {
    // 打印换行符，重复 7*x 次
    for (h = 1; h <= 7 * x; h++)
        print("\n");
} else {
    // 将 ms 赋值给 xs
    xs = ms;
    // 如果 ms 等于 "ALL"，则将 ps 赋值给 xs
    if (ms == "ALL")
        xs = ps;
    // 循环 7 次
    for (u = 1; u <= 7; u++) {
        // 从 8 开始递减到 0，提取位的一种低效方法
        // 但在 BASIC 中足够好，因为没有位移运算符
        for (k = 8; k >= 0; k--) {
            // 如果 2 的 k 次方大于等于 s[u]
            if (Math.pow(2, k) >= s[u]) {
                j[9 - k] = 0;
            } else {
                j[9 - k] = 1;
                s[u] -= Math.pow(2, k);
                // 如果 s[u] 等于 1
                if (s[u] == 1) {
                    f[u] = 9 - k;
                    break;
                }
            }
        }
        // 循环 x 次
        for (t1 = 1; t1 <= x; t1++) {
            // 生成空格字符串，长度为 (63 - 4.5 * y) * g1 / xs.length + 1
            str = tab((63 - 4.5 * y) * g1 / xs.length + 1);
            // 循环 f[u] 次
            for (b = 1; b <= f[u]; b++) {
                // 如果 j[b] 等于 0
                if (j[b] == 0) {
                    // 循环 y 次，打印 xs.length 个空格
                    for (i = 1; i <= y; i++)
                        str += tab(xs.length);
                } else {
                    // 循环 y 次，打印 xs
                    for (i = 1; i <= y; i++)
                        str += xs;
                }
            }
            // 打印 str 和换行符
            print(str + "\n");
        }
    }
    // 打印换行符，重复 2*x 次
    for (h = 1; h <= 2 * x; h++)
        print("\n");
}
// 调用 main 函数
main();
```