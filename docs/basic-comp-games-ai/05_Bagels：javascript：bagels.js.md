# `d:/src/tocomm/basic-computer-games\05_Bagels\javascript\bagels.js`

```
// BAGELS
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    // 在页面上输出文本
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型为文本
                       input_element.setAttribute("type", "text");
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
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回添加空格后的字符串
}

print(tab(33) + "BAGELS\n");  # 打印字符串"BAGELS"，并在前面添加33个空格
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加15个空格

// *** Bagles number guessing game
// *** Original source unknown but suspected to be
// *** Lawrence Hall of Science, U.C. Berkeley

a1 = [0,0,0,0];  # 初始化数组a1
a = [0,0,0,0];  # 初始化数组a
b = [0,0,0,0];  # 初始化数组b

y = 0;  # 初始化变量y为0
t = 255;  # 初始化变量t为255

print("\n");  # 打印一个空行
print("\n");  # 打印一个空行
print("\n");  # 打印一个空行

# Main program
async function main()
{
    while (1) {
        print("WOULD YOU LIKE THE RULES (YES OR NO)");  # 打印提示信息
        str = await input();  # 等待用户输入
        if (str.substr(0, 1) != "N") {  # 如果用户输入的第一个字符不是"N"
            print("\n");  # 打印一个空行
            print("I AM THINKING OF A THREE-DIGIT NUMBER.  TRY TO GUESS\n");  # 打印提示信息
            print("MY NUMBER AND I WILL GIVE YOU CLUES AS FOLLOWS:\n");  # 打印提示信息
            print("   PICO   - ONE DIGIT CORRECT BUT IN THE WRONG POSITION\n");  # 打印提示信息
            print("   FERMI  - ONE DIGIT CORRECT AND IN THE RIGHT POSITION\n");  # 打印提示信息
            print("   BAGELS - NO DIGITS CORRECT\n");  # 打印提示信息
        }
        for (i = 1; i <= 3; i++) {  # 循环3次
            do {
                a[i] = Math.floor(Math.random() * 10);  # 生成一个0到9之间的随机整数并赋值给a[i]
                for (j = i - 1; j >= 1; j--) {  # 循环，j从i-1开始递减到1
                    if (a[i] == a[j])  # 如果数组a中索引为i和j的元素相等
                        break;  # 跳出当前循环
                }  # 结束内层循环
            } while (j >= 1) ;  # 当j大于等于1时，继续执行循环
        }  # 结束外层循环
        print("\n");  # 打印换行
        print("O.K.  I HAVE A NUMBER IN MIND.\n");  # 打印提示信息
        for (i = 1; i <= 20; i++) {  # 循环20次
            while (1) {  # 无限循环
                print("GUESS #" + i);  # 打印猜测次数
                str = await input();  # 等待用户输入并赋值给str
                if (str.length != 3) {  # 如果输入的字符串长度不等于3
                    print("TRY GUESSING A THREE-DIGIT NUMBER.\n");  # 打印提示信息
                    continue;  # 继续下一次循环
                }
                for (z = 1; z <= 3; z++)  # 循环3次
                    a1[z] = str.charCodeAt(z - 1);  # 将输入字符串的每个字符的 ASCII 值赋给数组a1
                for (j = 1; j <= 3; j++) {  # 循环3次
                    if (a1[j] < 48 || a1[j] > 57)  # 如果数组a1中索引为j的元素不在ASCII码范围内
                        break;  # 跳出当前循环
                b[j] = a1[j] - 48;  # 将字符转换为数字，将a1中的字符转换为数字并存储在b中
            }
            if (j <= 3) {  # 如果j小于等于3
                print("WHAT?");  # 打印"WHAT?"
                continue;  # 继续下一次循环
            }
            if (b[1] == b[2] || b[2] == b[3] || b[3] == b[1]) {  # 如果b中的数字有重复
                print("OH, I FORGOT TO TELL YOU THAT THE NUMBER I HAVE IN MIND\n");  # 打印提示信息
                print("HAS NO TWO DIGITS THE SAME.\n");  # 打印提示信息
                continue;  # 继续下一次循环
            }
            break;  # 跳出循环
        }
        c = 0;  # 初始化c为0
        d = 0;  # 初始化d为0
        for (j = 1; j <= 2; j++) {  # 循环两次
            if (a[j] == b[j + 1])  # 如果a和b中对应位置的数字相等
                c++;  # c加1
            if (a[j + 1] == b[j])  # 如果a和b中对应位置的数字相等
                c++;  # c加1
            }  # 结束 if 语句块
            if (a[1] == b[3])  # 如果 a 数组的第二个元素等于 b 数组的第四个元素
                c++;  # c 自增1
            if (a[3] == b[1])  # 如果 a 数组的第四个元素等于 b 数组的第二个元素
                c++;  # c 自增1
            for (j = 1; j <= 3; j++) {  # 循环遍历 j 从 1 到 3
                if (a[j] == b[j])  # 如果 a 数组的第 j 个元素等于 b 数组的第 j 个元素
                    d++;  # d 自增1
            }
            if (d == 3)  # 如果 d 等于 3
                break;  # 跳出循环
            for (j = 0; j < c; j++)  # 循环遍历 j 从 0 到 c
                print("PICO ");  # 打印 "PICO "
            for (j = 0; j < d; j++)  # 循环遍历 j 从 0 到 d
                print("FERMI ");  # 打印 "FERMI "
            if (c + d == 0)  # 如果 c 和 d 的和等于 0
                print("BAGELS");  # 打印 "BAGELS"
            print("\n");  # 打印换行符
        }  # 结束 for 循环块
        if (i <= 20) {  # 如果 i 小于等于 20
            print("YOU GOT IT!!!\n");  # 打印出玩家猜对了的消息
            print("\n");  # 打印一个空行
        } else {
            print("OH WELL.\n");  # 打印出玩家猜错了的消息
            print("THAT'S A TWENTY GUESS.  MY NUMBER WAS " + a[1] + a[2] + a[3]);  # 打印出正确的数字
        }
        y++;  # 将 y 的值加一
        print("PLAY AGAIN (YES OR NO)");  # 打印出是否要再玩一次的提示
        str = await input();  # 等待用户输入并将输入的字符串赋值给变量 str
        if (str.substr(0, 1) != "Y")  # 如果用户输入的字符串的第一个字符不是 Y
            break;  # 跳出循环
    }
    if (y == 0)  # 如果 y 的值为 0
        print("HOPE YOU HAD FUN.  BYE.\n");  # 打印出祝福语和再见的消息
    else
        print("\nA " + y + " POINT BAGELS BUFF!!\n");  # 打印出玩家得分的消息

}

main();  # 调用主函数
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```