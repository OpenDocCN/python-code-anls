# `.\pytorch\c10\util\LeftRight.h`

```py
// 包含 C10 库中的必要头文件
#include <c10/macros/Macros.h>
#include <c10/util/Synchronized.h>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

// 进入 c10 命名空间
namespace c10 {

// 进入 detail 命名空间
namespace detail {

// IncrementRAII 结构，用于自动递增和递减原子计数器
struct IncrementRAII final {
 public:
  // 构造函数，接收一个指向 std::atomic<int32_t> 的指针，对计数器进行递增操作
  explicit IncrementRAII(std::atomic<int32_t>* counter) : _counter(counter) {
    _counter->fetch_add(1);
  }

  // 析构函数，对计数器进行递减操作
  ~IncrementRAII() {
    _counter->fetch_sub(1);
  }

 private:
  std::atomic<int32_t>* _counter; // 指向原子计数器的指针

  C10_DISABLE_COPY_AND_ASSIGN(IncrementRAII); // 禁止复制和赋值操作
};

} // namespace detail

// LeftRight 类模板，实现左右分离法的等待无锁读取同步机制
template <class T>
class LeftRight final {
 public:
  // 构造函数，接收可变参数 Args，并初始化成员变量
  template <class... Args>
  explicit LeftRight(const Args&... args)
      : _counters{{{0}, {0}}}, // 初始化计数器数组
        _foregroundCounterIndex(0), // 初始化前台计数器索引
        _foregroundDataIndex(0), // 初始化前台数据索引
        _data{{T{args...}, T{args...}}}, // 使用参数初始化数据数组
        _writeMutex() {} // 初始化写入互斥量

  // 禁用拷贝和移动构造函数及赋值操作符
  LeftRight(const LeftRight&) = delete;
  LeftRight(LeftRight&&) noexcept = delete;
  LeftRight& operator=(const LeftRight&) = delete;
  LeftRight& operator=(LeftRight&&) noexcept = delete;

  // 析构函数，等待所有写入和读取操作结束
  ~LeftRight() {
    // 等待所有写入操作结束
    { std::unique_lock<std::mutex> lock(_writeMutex); }

    // 等待所有读取操作结束
    while (_counters[0].load() != 0 || _counters[1].load() != 0) {
      std::this_thread::yield();
    }
  }

  // 读取函数模板，接收一个可调用对象 readFunc，并在读取期间自动递增和递减前台计数器
  template <typename F>
  auto read(F&& readFunc) const {
    detail::IncrementRAII _increment_counter(
        &_counters[_foregroundCounterIndex.load()]);

    return std::forward<F>(readFunc)(_data[_foregroundDataIndex.load()]);
  }

  // 写入函数模板，接收一个可调用对象 writeFunc，并在写入期间获取写入互斥量，执行写入操作
  template <typename F>
  auto write(F&& writeFunc) {
    std::unique_lock<std::mutex> lock(_writeMutex);

    return _write(std::forward<F>(writeFunc));
  }

 private:
  // 内部写入函数模板，接收一个可调用对象 writeFunc，执行实际的写入操作
  template <class F>
  auto _write(const F& writeFunc) {
    /*
     * 假设，A 是后台数据，B 是前台数据。简化来说，我们想要执行以下操作：
     * 1. 向 A 写入数据（旧后台）
     * 2. 切换 A/B
     * 3. 向 B 写入数据（新后台）
     *
     * 更详细的算法（为什么这些步骤重要的解释在下面的代码中）：
     * 1. 向 A 写入数据
     * 2. 切换 A/B 数据指针
     * 3. 等待直到 A 计数器为零
     * 4. 切换 A/B 计数器
     * 5. 等待直到 B 计数器为零
     * 6. 向 B 写入数据
     */

    // 获取当前的前台数据索引
    auto localDataIndex = _foregroundDataIndex.load();

    // 1. 向 A 写入数据
    _callWriteFuncOnBackgroundInstance(writeFunc, localDataIndex);

    // 2. 切换 A/B 数据指针
    localDataIndex = localDataIndex ^ 1;
    _foregroundDataIndex = localDataIndex;

    /*
     * 3. 等待直到 A 计数器为零
     *
     * 在前一个写入运行时，A 是前台数据而 B 是后台数据。
     * 在切换 _foregroundDataIndex（将 B 切换到前台）之后，以及切换 _foregroundCounterIndex 之前，
     * 新的读取者可能已经读取了 B，但是增加了 A 的计数器。
     *
     * 在当前运行中，我们刚刚切换了 _foregroundDataIndex（将 A 切换回前台），但在向新的后台 B 写入之前，
     * 我们必须确保 A 的计数器短暂地为零，以便所有这些旧的读取者消失。
     */
    auto localCounterIndex = _foregroundCounterIndex.load();
    _waitForBackgroundCounterToBeZero(localCounterIndex);

    /*
     * 4. 切换 A/B 计数器
     *
     * 现在我们知道所有读取 B 的读者确实已经消失了，我们可以切换计数器，并且让新的读取者再次增加 A 的计数器，
     * 因为他们正在读取 A。
     */
    localCounterIndex = localCounterIndex ^ 1;
    _foregroundCounterIndex = localCounterIndex;

    /*
     * 5. 等待直到 B 计数器为零
     *
     * 这等待所有在 B 上的读取者，这些读取者是在数据和计数器都在前台时发生的，即在切换数据和计数器之间的短暂间隙外的正常读取者。
     */
    _waitForBackgroundCounterToBeZero(localCounterIndex);

    // 6. 向 B 写入数据
    return _callWriteFuncOnBackgroundInstance(writeFunc, localDataIndex);
  }

  template <class F>
  auto _callWriteFuncOnBackgroundInstance(
      const F& writeFunc,
      uint8_t localDataIndex) {
    try {
      // 尝试调用写函数在后台实例上
      return writeFunc(_data[localDataIndex ^ 1]);
    } catch (...) {
      // 恢复不变性，通过从前台实例复制
      _data[localDataIndex ^ 1] = _data[localDataIndex];
      // 重新抛出异常
      throw;
    }
  }

  void _waitForBackgroundCounterToBeZero(uint8_t counterIndex) {
    // 当反向计数器不为零时，持续等待
    while (_counters[counterIndex ^ 1].load() != 0) {
      std::this_thread::yield();
    }
  }
  }



  // 结束了一个代码块的定义，这里应该是一个类或函数的结束标志
}



  mutable std::array<std::atomic<int32_t>, 2> _counters;



  // 可变的数组，包含两个原子整型，用于多线程中的计数器



  std::atomic<uint8_t> _foregroundCounterIndex;



  // 原子变量，存储前台计数器的索引，使用无符号8位整数



  std::atomic<uint8_t> _foregroundDataIndex;



  // 原子变量，存储前台数据的索引，使用无符号8位整数



  std::array<T, 2> _data;



  // 存储模板类型 T 的数组，包含两个元素，用于存储数据



  std::mutex _writeMutex;



  // 互斥锁，用于控制对 _data 的写操作，确保线程安全
};

// RWSafeLeftRightWrapper is API compatible with LeftRight and uses a
// read-write lock to protect T (data).
// RWSafeLeftRightWrapper 类提供了与 LeftRight 兼容的 API，并使用读写锁来保护 T 类型的数据。

template <class T>
class RWSafeLeftRightWrapper final {
 public:
  template <class... Args>
  // 构造函数，接受任意参数 Args，并将其传递给成员变量 data_ 的构造函数
  explicit RWSafeLeftRightWrapper(const Args&... args) : data_{args...} {}

  // RWSafeLeftRightWrapper 不能进行复制或移动，因为 LeftRight 也不能复制或移动。
  // 删除复制构造函数
  RWSafeLeftRightWrapper(const RWSafeLeftRightWrapper&) = delete;
  // 删除移动构造函数，但是 noexcept 说明移动操作不会抛出异常
  RWSafeLeftRightWrapper(RWSafeLeftRightWrapper&&) noexcept = delete;
  // 删除复制赋值运算符
  RWSafeLeftRightWrapper& operator=(const RWSafeLeftRightWrapper&) = delete;
  // 删除移动赋值运算符
  RWSafeLeftRightWrapper& operator=(RWSafeLeftRightWrapper&&) noexcept = delete;

  template <typename F>
  // 使用 const 成员函数 read，接受一个函数对象 readFunc，并在 data_ 上加锁，以只读方式访问数据
  auto read(F&& readFunc) const {
    return data_.withLock(
        [&readFunc](T const& data) { return std::forward<F>(readFunc)(data); });
  }

  template <typename F>
  // 使用非 const 成员函数 write，接受一个函数对象 writeFunc，并在 data_ 上加锁，以读写方式访问数据
  auto write(F&& writeFunc) {
    return data_.withLock(
        [&writeFunc](T& data) { return std::forward<F>(writeFunc)(data); });
  }

 private:
  c10::Synchronized<T> data_; // 使用 c10::Synchronized 包装的数据成员
};

} // namespace c10
```