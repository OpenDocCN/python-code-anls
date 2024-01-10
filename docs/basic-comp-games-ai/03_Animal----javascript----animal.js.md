# `basic-computer-games\03_Animal\javascript\animal.js`

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
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      // 如果按下回车键
                                                      if (event.keyCode == 13) {
                                                        // 获取输入框的值
                                                        input_str = input_element.value;
                                                        // 移除输入框
                                                        document.getElementById("output").removeChild(input_element);
                                                        // 打印输入的值
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

// 打印标题
print(tab(32) + "ANIMAL\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");
// 打印游戏提示
print("PLAY 'GUESS THE ANIMAL'\n");
print("\n");
print("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.\n");
print("\n");

// 定义变量
var k;
var n;
var str;
var q;
var z;
var c;
var t;
// 定义一个包含动物信息的数组
var animals = [
               "\\QDOES IT SWIM\\Y1\\N2\\",
               "\\AFISH",
               "\\ABIRD",
               ];

// 获取动物数组的长度
n = animals.length;

// 定义一个函数用于展示已知的动物信息
function show_animals() {
    var x;

    // 打印空行和标题
    print("\n");
    print("ANIMALS I ALREADY KNOW ARE:\n");
    // 初始化一个空字符串
    str = "";
    // 初始化 x 的值为 0
    x = 0;
    // 遍历动物数组
    for (var i = 0; i < n; i++) {
        // 判断动物信息是否以 "\\A" 开头
        if (animals[i].substr(0, 2) == "\\A") {
            // 当字符串长度小于 15*x 时，向字符串末尾添加空格
            while (str.length < 15 * x)
                str += " ";
            // 遍历动物信息，提取动物名称
            for (var z = 2; z < animals[i].length; z++) {
                // 如果遇到 "\\" 则跳出循环
                if (animals[i][z] == "\\")
                    break;
                // 将动物名称添加到字符串中
                str += animals[i][z];
            }
            // 增加 x 的值
            x++;
            // 当 x 等于 4 时，重置 x 的值为 0，打印字符串并重置字符串为空
            if (x == 4) {
                x = 0;
                print(str + "\n");
                str = "";
            }
        }
    }
    // 如果字符串不为空，则打印字符串
    if (str != "")
        print(str + "\n");
}

// 主控制部分
async function main()
{
    while (1) {
        while (1) {
            // 打印提示信息，询问用户是否在想一个动物
            print("ARE YOU THINKING OF AN ANIMAL");
            // 获取用户输入的字符串
            str = await input();
            // 如果用户输入的是"LIST"，则展示动物列表
            if (str == "LIST")
                show_animals();
            // 如果用户输入的字符串以"Y"开头，则跳出内层循环
            if (str[0] == "Y")
                break;
        }

        // 初始化变量k为0
        k = 0;
        do {
            // 子程序，用于打印问题
            q = animals[k];
            while (1) {
                // 初始化字符串str为空
                str = "";
                // 遍历问题字符串q，直到遇到"\"
                for (z = 2; z < q.length; z++) {
                    if (q[z] == "\\")
                        break;
                    str += q[z];
                }
                // 打印问题字符串
                print(str);
                // 获取用户输入的字符
                c = await input();
                // 如果用户输入的第一个字符是"Y"或"N"，则跳出内层循环
                if (c[0] == "Y" || c[0] == "N")
                    break;
            }
            // 获取用户输入的字符，并构造新的索引k
            t = "\\" + c[0];
            x = q.indexOf(t);
            k = parseInt(q.substr(x + 2));
        } while (animals[k].substr(0,2) == "\\Q") ;

        // 打印猜测的动物
        print("IS IT A " + animals[k].substr(2));
        // 获取用户输入的字符
        a = await input();
        // 如果用户输入的第一个字符是"Y"，则提示用户尝试另一个动物
        if (a[0] == "Y") {
            print("WHY NOT TRY ANOTHER ANIMAL?\n");
            continue;
        }
        // 打印用户所想的动物
        print("THE ANIMAL YOU WERE THINKING OF WAS A ");
        v = await input();
        // 提示用户输入一个可以区分两个动物的问题
        print("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A\n");
        print(v + " FROM A " + animals[k].substr(2) + "\n");
        x = await input();
        while (1) {
            // 提示用户对于动物v的答案
            print("FOR A " + v + " THE ANSWER WOULD BE ");
            // 获取用户输入的字符
            a = await input();
            // 截取用户输入的第一个字符
            a = a.substr(0, 1);
            // 如果用户输入的字符是"Y"或"N"，则跳出内层循环
            if (a == "Y" || a == "N")
                break;
        }
        // 根据用户输入的答案构造新的问题字符串
        if (a == "Y")
            b = "N";
        if (a == "N")
            b = "Y";
        z1 = animals.length;
        animals[z1] = animals[k];
        animals[z1 + 1] = "\\A" + v;
        animals[k] = "\\Q" + x + "\\" + a + (z1 + 1) + "\\" + b + z1 + "\\";
    }
# 结束 main 函数的定义
}

# 调用 main 函数
main();
```