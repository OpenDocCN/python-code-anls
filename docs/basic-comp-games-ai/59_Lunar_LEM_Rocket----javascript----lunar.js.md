# `59_Lunar_LEM_Rocket\javascript\lunar.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
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
```
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串
}

var l;  # 声明变量 l
var t;  # 声明变量 t
var m;  # 声明变量 m
var s;  # 声明变量 s
var k;  # 声明变量 k
var a;  # 声明变量 a
var v;  # 声明变量 v
var i;  # 声明变量 i
var j;  # 声明变量 j
var q;  # 声明变量 q
var g;  # 声明变量 g
var z;  # 声明变量 z
var d;  # 声明变量 d

function formula_set_1()  # 定义名为 formula_set_1 的函数
{
    l = l + s;  // 将变量 l 的值加上变量 s 的值
    t = t - s;  // 将变量 t 的值减去变量 s 的值
    m = m - s * k;  // 将变量 m 的值减去变量 s 乘以变量 k 的值
    a = i;  // 将变量 a 的值设置为变量 i 的值
    v = j;  // 将变量 v 的值设置为变量 j 的值
}

function formula_set_2()
{
    q = s * k / m;  // 计算变量 q 的值
    j = v + g * s + z * (-q - q * q / 2 - Math.pow(q, 3) / 3 - Math.pow(q, 4) / 4 - Math.pow(q, 5) / 5);  // 计算变量 j 的值
    i = a - g * s * s / 2 - v * s + z * s * (q / 2 + Math.pow(q, 2) / 6 + Math.pow(q, 3) / 12 + Math.pow(q, 4) / 20 + Math.pow(q, 5) / 30);  // 计算变量 i 的值
}

function formula_set_3()
{
    while (s >= 5e-3) {  // 当变量 s 大于等于 5e-3 时执行循环
        d = v + Math.sqrt(v * v + 2 * a * (g - z * k / m));  // 计算变量 d 的值
        s = 2 * a / d;  // 计算变量 s 的值
        formula_set_2();  // 调用 formula_set_2 函数
        formula_set_1(); // 调用 formula_set_1 函数
    }
}

// Main program
async function main()
{
    print(tab(33) + "LUNAR\n"); // 打印带有制表符的字符串 "LUNAR"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n"); // 打印带有制表符的字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    print("THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR\n"); // 打印字符串 "THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR"
    print("LANDING CAPSULE.\n"); // 打印字符串 "LANDING CAPSULE."
    print("\n"); // 打印空行
    print("\n"); // 打印空行
    print("THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY\n"); // 打印字符串 "THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY"
    print("XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY.\n"); // 打印字符串 "XEROX) SO YOU HAVE TO LAND THE CAPSULE MANUALLY."
    while (1) { // 进入无限循环
        print("\n"); // 打印空行
        # 打印设置RETRO火箭燃烧速率的提示信息
        print("SET BURN RATE OF RETRO ROCKETS TO ANY VALUE BETWEEN\n");
        # 打印设置RETRO火箭燃烧速率范围的提示信息
        print("0 (FREE FALL) AND 200 (MAXIMUM BURN) POUNDS PER SECOND.\n");
        # 打印设置每10秒更新一次燃烧速率的提示信息
        print("SET NEW BURN RATE EVERY 10 SECONDS.\n");
        # 打印空行
        print("\n");
        # 打印胶囊重量和燃料重量的提示信息
        print("CAPSULE WEIGHT 32,500 LBS; FUEL WEIGHT 16,000 LBS.\n");
        # 打印多个空行
        print("\n");
        print("\n");
        print("\n");
        # 打印祝愿好运的提示信息
        print("GOOD LUCK\n");
        # 初始化变量l为0
        l = 0;
        # 打印表头信息
        print("\n");
        print("SEC\tMI + FT\t\tMPH\tLB FUEL\tBURN RATE\n");
        print("\n");
        # 初始化变量a为120
        a = 120;
        # 初始化变量v为1
        v = 1;
        # 初始化变量m为32500
        m = 32500;
        # 初始化变量n为16500
        n = 16500;
        # 初始化变量g为0.001
        g = 1e-3;
        # 初始化变量z为1.8
        z = 1.8;
        # 进入无限循环
        while (1) {
            # 打印输出一行数据，包括 l, a, v, m, n 的计算结果
            print(l + "\t" + Math.floor(a) + " + " + Math.floor(5280 * (a - Math.floor(a))) + " \t" + Math.floor(3600 * v * 100) / 100 + "\t" + (m - n) + "\t");
            # 从用户输入获取一个浮点数并转换为浮点数
            k = parseFloat(await input());
            # 初始化变量 t 为 10
            t = 10;
            # 初始化变量 should_exit 为 false
            should_exit = false;
            # 进入循环，条件为永远为真
            while (1) {
                # 如果 m - n 小于 1e-3，则跳出循环
                if (m - n < 1e-3)
                    break;
                # 如果 t 小于 1e-3，则跳出循环
                if (t < 1e-3)
                    break;
                # 将变量 s 初始化为 t
                s = t;
                # 如果 m 小于 n + s * k，则将 s 更新为 (m - n) / k
                if (m < n + s * k)
                    s = (m - n) / k;
                # 调用 formula_set_2 函数
                formula_set_2();
                # 如果 i 小于等于 0，则调用 formula_set_3 函数，设置 should_exit 为 true，并跳出循环
                if (i <= 0) {
                    formula_set_3();
                    should_exit = true;
                    break;
                }
                # 如果 v 大于 0 且 j 小于 0
                if (v > 0) {
                    if (j < 0) {
# 开始一个 do-while 循环，条件是 v 大于 0
do {
    # 计算 w 的值
    w = (1 - m * g / (z * k)) / 2;
    # 计算 s 的值
    s = m * v / (z * k * (w + Math.sqrt(w * w + v / z))) + 0.05;
    # 调用 formula_set_2 函数
    formula_set_2();
    # 如果 i 小于等于 0，则执行以下操作
    if (i <= 0) {
        # 调用 formula_set_3 函数
        formula_set_3();
        # 设置 should_exit 为 true
        should_exit = true;
        # 跳出循环
        break;
    }
    # 调用 formula_set_1 函数
    formula_set_1();
    # 如果 j 大于 0，则跳出循环
    if (j > 0)
        break;
} while (v > 0) ;  # 循环条件结束

# 如果 should_exit 为 true，则跳出循环
if (should_exit)
    break;
# 继续下一次循环
continue;
# 结束 do-while 循环

# 调用 formula_set_1 函数
formula_set_1();
            # 如果应该退出，则跳出循环
            if (should_exit)
                break;
            # 如果燃料少于1e-3，输出燃料用尽的信息，并计算着陆速度和时间
            if (m - n < 1e-3) {
                print("FUEL OUT AT " + l + " SECOND\n");
                s = (-v * Math.sqrt(v * v + 2 * a * g)) / g;
                v = v + g * s;
                l = l + s;
                break;
            }
        }
        # 计算着陆速度并转换为英里每小时
        w = 3600 * v;
        # 输出着陆信息，包括着陆时间和着陆速度
        print("ON MOON AT " + l + " SECONDS - IMPACT VELOCITY " + w + " MPH\n");
        # 根据着陆速度判断着陆质量
        if (w <= 1.2) {
            print("PERFECT LANDING!\n");
        } else if (w <= 10) {
            print("GOOD LANDING (COULD BE BETTER)\n");
        } else if (w <= 60) {
            print("CRAFT DAMAGE... YOU'RE STRANDED HERE UNTIL A RESCUE\n");
            print("PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!\n");
        } else {
# 打印字符串，表示没有幸存者
print("SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!\n");
# 打印字符串，表示实际上你炸出了一个新的月球撞击坑的深度
print("IN FACT, YOU BLASTED A NEW LUNAR CRATER " + (w * 0.227) + " FEET DEEP!\n");
# 打印空行
print("\n");
# 打印空行
print("\n");
# 打印空行
print("\n");
# 打印字符串，提示再次尝试
print("TRY AGAIN??\n");
# 结束当前函数
}
# 结束当前函数
}

# 调用主函数
main();
```