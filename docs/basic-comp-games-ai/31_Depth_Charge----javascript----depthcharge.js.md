# `31_Depth_Charge\javascript\depthcharge.js`

```
# DEPTH CHARGE
# 
# Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
# 

# 定义一个打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义一个输入函数，返回一个 Promise 对象
def input():
    var input_element
    var input_str

    return new Promise(function (resolve):
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT")

                       # 打印提示符
                       print("? ")

                       # 设置输入元素的类型为文本
                       input_element.setAttribute("type", "text")
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
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

// Main program
async function main()
{
    print(tab(30) + "DEPTH CHARGE\n");  // 打印带有缩进的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有缩进的字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("DIMENSION OF THE SEARCH AREA");  // 打印提示信息
    g = Math.floor(await input());  // 获取用户输入并向下取整
    n = Math.floor(Math.log(g) / Math.log(2)) + 1;  // 计算n的值
    print("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER\n");  // 打印提示信息
    print("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR\n");  // 打印提示信息
    print("MISSION IS TO DESTROY IT.  YOU HAVE " + n + " SHOTS.\n");  // 打印提示信息
    print("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A\n");  // 打印提示信息
    print("TRIO OF NUMBERS -- THE FIRST TWO ARE THE\n");  // 打印提示信息
    # 打印提示信息
    print("SURFACE COORDINATES; THE THIRD IS THE DEPTH.\n")
    # 循环开始
    do:
        # 打印提示信息
        print("\n")
        print("GOOD LUCK !\n")
        print("\n")
        # 生成随机的坐标值
        a = Math.floor(Math.random() * g)
        b = Math.floor(Math.random() * g)
        c = Math.floor(Math.random() * g)
        # 循环n次
        for (d = 1; d <= n; d++):
            # 打印提示信息
            print("\n")
            print("TRIAL #" + d + " ")
            # 等待用户输入
            str = await input()
            # 解析用户输入的坐标值
            x = parseInt(str)
            y = parseInt(str.substr(str.indexOf(",") + 1))
            z = parseInt(str.substr(str.lastIndexOf(",") + 1))
            # 如果用户输入的坐标与随机生成的坐标相同，则跳出循环
            if (Math.abs(x - a) + Math.abs(y - b) + Math.abs(z - c) == 0)
                break
            # 如果用户输入的y坐标大于随机生成的y坐标，则打印提示信息
            if (y > b)
                print("NORTH")
            # 如果用户输入的y坐标小于随机生成的y坐标，则打印提示信息
            if (y < b)
# 如果 x 大于 a，则打印 "EAST"
if (x > a)
    print("EAST");
# 如果 x 小于 a，则打印 "WEST"
if (x < a)
    print("WEST");
# 如果 y 不等于 b 或者 x 不等于 a，则打印 " AND"
if (y != b || x != a)
    print(" AND");
# 如果 z 大于 c，则打印 " TOO LOW.\n"
if (z > c)
    print(" TOO LOW.\n");
# 如果 z 小于 c，则打印 " TOO HIGH.\n"
if (z < c)
    print(" TOO HIGH.\n");
# 如果 z 等于 c，则打印 " DEPTH OK.\n"
if (z == c)
    print(" DEPTH OK.\n");
# 打印换行符
print("\n");
# 如果 d 小于等于 n，则打印 "B O O M ! ! YOU FOUND IT IN " + d + " TRIES!\n"
if (d <= n) {
    print("\n");
    print("B O O M ! ! YOU FOUND IT IN " + d + " TRIES!\n");
} else {
    print("\n");
# 打印“YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!”
print("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!\n");
# 打印“THE SUBMARINE WAS AT ”和变量 a、b、c 的值
print("THE SUBMARINE WAS AT " + a + "," + b + "," + c + "\n");
# 打印两个空行
print("\n");
print("\n");
# 打印“ANOTHER GAME (Y OR N)”
print("ANOTHER GAME (Y OR N)");
# 从用户输入中获取字符串
str = await input();
# 当用户输入的字符串的第一个字符是“Y”时，执行循环
} while (str.substr(0, 1) == "Y") ;
# 打印“OK.  HOPE YOU ENJOYED YOURSELF.”
print("OK.  HOPE YOU ENJOYED YOURSELF.\n");
# 调用主函数
main();
```