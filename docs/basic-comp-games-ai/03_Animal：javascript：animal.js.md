# `d:/src/tocomm/basic-computer-games\03_Animal\javascript\animal.js`

```
// 创建一个新的 Promise 对象，用于处理输入操作
// 创建一个 INPUT 元素，用于接收用户输入
// 在页面上打印提示符 "? "
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
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将字符串 str 后面添加一个空格
    return str;  # 返回处理后的字符串

print(tab(32) + "ANIMAL\n");  # 打印一个制表符加上字符串 "ANIMAL"，并换行
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印一个制表符加上字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行
print("\n");  # 打印一个空行
print("\n");  # 打印一个空行
print("\n");  # 打印一个空行
print("PLAY 'GUESS THE ANIMAL'\n");  # 打印字符串 "PLAY 'GUESS THE ANIMAL'"，并换行
print("\n");  # 打印一个空行
print("THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT.\n");  # 打印字符串 "THINK OF AN ANIMAL AND THE COMPUTER WILL TRY TO GUESS IT."，并换行
print("\n");  # 打印一个空行

var k;  # 声明变量 k
var n;  # 声明变量 n
var str;  # 声明变量 str
var q;  # 声明变量 q
var z;  # 声明变量 z
var c;  # 声明变量 c
# 声明变量 t
var t;

# 声明并初始化包含特殊字符的字符串数组
var animals = [
               "\\QDOES IT SWIM\\Y1\\N2\\",
               "\\AFISH",
               "\\ABIRD",
               ];

# 获取数组长度
n = animals.length;

# 定义函数 show_animals
function show_animals() {
    # 声明变量 x
    var x;

    # 打印换行符
    print("\n");
    # 打印已知的动物
    print("ANIMALS I ALREADY KNOW ARE:\n");
    # 初始化字符串
    str = "";
    # 初始化变量 x
    x = 0;
    # 循环遍历数组
    for (var i = 0; i < n; i++) {
        # 检查数组元素的前两个字符是否为 "\\A"
        if (animals[i].substr(0, 2) == "\\A") {
            # 循环直到字符串长度达到特定条件
            while (str.length < 15 * x)
                str += " ";  // 在字符串末尾添加空格
            for (var z = 2; z < animals[i].length; z++) {  // 循环遍历 animals[i] 中的元素
                if (animals[i][z] == "\\")  // 如果当前元素为反斜杠，则跳出循环
                    break;
                str += animals[i][z];  // 将当前元素添加到字符串末尾
            }
            x++;  // x 值加一
            if (x == 4) {  // 如果 x 等于 4
                x = 0;  // 重置 x 为 0
                print(str + "\n");  // 打印字符串并换行
                str = "";  // 重置字符串为空
            }
        }
    }
    if (str != "")  // 如果字符串不为空
        print(str + "\n");  // 打印字符串并换行
}

// Main control section
async function main()  // 主控制部分的异步函数
{
    while (1) {  # 进入无限循环
        while (1) {  # 进入内部无限循环
            print("ARE YOU THINKING OF AN ANIMAL");  # 打印提示信息
            str = await input();  # 等待用户输入并将输入内容赋值给变量str
            if (str == "LIST")  # 如果输入内容为"LIST"
                show_animals();  # 调用show_animals函数
            if (str[0] == "Y")  # 如果输入内容的第一个字符为"Y"
                break;  # 退出内部循环
        }

        k = 0;  # 初始化变量k为0
        do {  # 进入do-while循环
            // Subroutine to print questions  # 打印问题的子程序
            q = animals[k];  # 将animals列表中索引为k的元素赋值给变量q
            while (1) {  # 进入内部无限循环
                str = "";  # 初始化变量str为空字符串
                for (z = 2; z < q.length; z++) {  # 遍历q中索引从2到q长度的元素
                    if (q[z] == "\\")  # 如果当前元素为"\"
                        break;  # 退出for循环
# 初始化一个空字符串
str = "";
# 遍历数组q，将其中的元素拼接到字符串str中
for z in range(len(q)):
    str += q[z];
# 打印字符串str
print(str);
# 等待用户输入，并将输入内容赋值给变量c
c = await input();
# 如果用户输入的第一个字符是"Y"或"N"，则跳出循环
if (c[0] == "Y" || c[0] == "N")
    break;
# 将"\\"和用户输入的第一个字符拼接成字符串t
t = "\\" + c[0];
# 获取字符串q中t的索引位置，并赋值给变量x
x = q.indexOf(t);
# 将从索引x+2开始的子字符串转换为整数，并赋值给变量k
k = parseInt(q.substr(x + 2));
# 当数组animals中索引为k的元素的前两个字符是"\Q"时，执行循环
while (animals[k].substr(0,2) == "\\Q") ;
    # 打印提示信息和数组animals中索引为k的元素去掉前两个字符的部分
    print("IS IT A " + animals[k].substr(2));
    # 等待用户输入，并将输入内容赋值给变量a
    a = await input();
    # 如果用户输入的第一个字符是"Y"，则打印提示信息并继续循环
    if (a[0] == "Y") {
        print("WHY NOT TRY ANOTHER ANIMAL?\n");
        continue;
    # 如果用户输入的第一个字符不是"Y"，则打印提示信息
    print("THE ANIMAL YOU WERE THINKING OF WAS A ");
    # 等待用户输入，并将输入内容赋值给变量v
    v = await input();
# 打印提示信息，要求用户输入一个问题来区分两个事物
print("PLEASE TYPE IN A QUESTION THAT WOULD DISTINGUISH A\n")
# 打印提示信息，包括一个变量和动物名称的子字符串
print(v + " FROM A " + animals[k].substr(2) + "\n")
# 等待用户输入，并将输入的内容赋给变量x
x = await input()
# 进入循环，直到用户输入合法的答案
while (1):
    # 打印提示信息，要求用户输入答案
    print("FOR A " + v + " THE ANSWER WOULD BE ")
    # 等待用户输入，并将输入的内容赋给变量a
    a = await input()
    # 只取答案的第一个字符
    a = a.substr(0, 1)
    # 如果答案是Y或N，则跳出循环
    if (a == "Y" || a == "N"):
        break
# 根据用户的答案，确定另一个变量b的值
if (a == "Y"):
    b = "N"
if (a == "N"):
    b = "Y"
# 获取动物数组的长度
z1 = animals.length
# 将原来的动物复制到数组的末尾
animals[z1] = animals[k]
# 在数组的末尾添加一个新的元素，表示问题
animals[z1 + 1] = "\\A" + v
# 修改原来动物的表示方式，包括问题、答案和指向新元素的索引
animals[k] = "\\Q" + x + "\\" + a + (z1 + 1) + "\\" + b + z1 + "\\"
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，所以这行代码会导致错误。
```