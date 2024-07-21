# `.\pytorch\functorch\csrc\dim\arena.h`

```
// 版权声明，此代码受到 BSD 风格许可证保护，详见根目录下的 LICENSE 文件
#pragma once
// 引入 ATen 库
#include <ATen/ATen.h>
// 引入 minpybind.h 头文件
#include "minpybind.h"

#ifdef _WIN32
// 如果是 Windows 平台，在此处引入 intrin.h 头文件
#include <intrin.h>
// 定义 __builtin_clz 函数，使用 _BitScanReverse 实现，详见链接中的解释
inline unsigned int __builtin_clz(unsigned int x) {
    unsigned long r = 0;
    _BitScanReverse(&r, x);
    return (31 - r);
}
#endif

// 定义 round2min8 函数，将输入的整数 num 转换为大于等于 num 的最小的 2 的幂，且不小于 8
inline int round2min8(int num) {
   // 计算 (num - 1) | 4 的前导零个数
   int nzeros = __builtin_clz((num - 1)|4);
   // 返回 2 的 (32 - nzeros) 次幂
   return 1 << (32 - nzeros);
}

// 定义 Slice 结构模板
struct Arena;
template<typename T>
struct OwnedSlice;

template<typename T>
struct Slice {
    // 默认构造函数，初始化 begin_ 为 nullptr，size_ 和 capacity_ 均为 0
    Slice()
    :  begin_(nullptr), size_(0), capacity_(0) {}

    // Slice 的构造函数模板，使用 Arena 和参数列表初始化
    template<typename... Args>
    Slice(Arena& arena, Args&&... args);

    // 返回 Slice 开始的指针
    T* begin() const {
        return begin_;
    }
    // 返回 Slice 结束后的指针
    T* end() const {
        return begin_ + size_;
    }
    // 返回 Slice 的大小
    int size() const {
        return size_;
    }
    // 返回 Slice 的容量
    int capacity() const {
        return capacity_;
    }

    // 返回 Slice 的倒数第 i 个元素的引用
    T& back(int i=-1) {
        return begin_[size_ + i];
    }

    // 返回 Slice 的第 i 个元素的引用
    T& operator[](int i) const {
        return begin_[i];
    }
    // 查找值为 value 的元素，返回其索引值，如果不存在则返回 std::nullopt
    std::optional<int> index(const T& value) {
        for (int i : enumerate()) {
            if (begin_[i] == value) {
                return i;
            }
        }
        return c10::nullopt;
    }
    // 判断 Slice 是否包含值为 value 的元素
    bool contains(const T& value) {
        return index(value).has_value();
    }

    // 在指定位置插入另一个 Slice 到当前 Slice 中
    void insert(Arena& arena, Slice where, Slice to_insert);
    // 在指定位置插入值为 v 的元素到当前 Slice 中
    void insert(Arena& arena, Slice where, T v) {
        return insert(arena, where, Slice(&v, &v + 1));
    }
    // 在指定位置插入值为 v 的元素到当前 Slice 中
    void insert(Arena& arena, int where, T v) {
        return insert(arena, slice(where, where), v);
    }
    // 向当前 Slice 末尾追加一个值为 value 的元素
    void append(Arena& arena, T value);
    // 扩展当前 Slice，将另一个 Slice to_insert 的内容插入到当前 Slice 的末尾
    void extend(Arena& arena, Slice to_insert);
    // 扩展当前 Slice，将指定范围内的数组元素插入到当前 Slice 的末尾
    void extend(Arena& arena, const T* begin, const T* end) {
        return extend(arena, Slice<T>((T*)begin, (T*)end));
    }

    // 从当前 Slice 中移除值为 value 的元素
    bool remove(Arena& A, T value) {
        auto idx = index(value);
        if (idx) {
            insert(A, slice(*idx, *idx + 1), Slice());
        }
        return idx.has_value();
    }

    // 返回从 begin 开始到末尾的 Slice
    Slice slice(int begin) {
        return slice(begin, size_);
    }

    // 返回从 begin 到 end 的 Slice
    Slice slice(int begin, int end) {
        if (begin < 0) {
            begin += size_;
        }
        if (end < 0) {
            end += size_;
        }
        // 构造一个新的 Slice 对象 result
        Slice result;
        result.begin_ = begin_ + begin;
        result.size_ = end - begin;
        result.capacity_ = result.size_;
        return result;
    }

    // 判断当前 Slice 是否包含在 where 指定的范围内
    bool inside(Slice where) {
        return begin() <= where.begin() && where.end() <= end();
    }

    // 返回一个范围对象，用于迭代整个 Slice
    irange enumerate() const {
        return irange(size_);
    }

    // 返回一个反向范围对象，用于反向迭代整个 Slice
    irange reversed_enumerate() const {
        return irange(size_ - 1, -1, -1);
    }
    # 定义比较操作符==，用于比较当前 Slice 对象与另一个 Slice 对象 rhs 是否相等
    bool operator==(const Slice<T>& rhs) const {
        # 如果当前 Slice 的大小与 rhs 的大小不相等，则返回 false
        if (size() != rhs.size()) {
            return false;
        }
        # 使用 std::equal 函数比较当前 Slice 的起始位置到结束位置的元素与 rhs 的元素是否相等
        return std::equal(begin(), end(), rhs.begin());
    }
    
    # Slice 类的构造函数，用于初始化一个 Slice 对象
    Slice(T* begin, T* end)
    # 初始化 Slice 对象的起始位置为 begin，大小为 end - begin，容量也为 size_，即当前大小
    : begin_(begin), size_(end - begin), capacity_(size_) {}
protected:
    // 返回长度为1，用于非 Slice 类型的长度计算
    static int _length(const T& t) {
        return 1;
    }
    // 返回 Slice 对象的长度，用于 Slice 类型的长度计算
    static int _length(Slice t) {
        return t.size_;
    }
    // 将类型 T 的对象 t 插入到 dst 指向的位置，并返回插入后的新位置
    static T* _insert(T*& dst, T t) {
        *dst = std::move(t);
        return ++dst;
    }
    // 将 Slice 对象 t 的数据复制到 dst 指向的位置，并返回插入后的新位置
    static T* _insert(T*& dst, Slice t) {
        std::memcpy(dst, t.begin_, sizeof(T) * t.size_);
        dst += t.size_;
        return dst;
    }
    // 指向数据开始的指针
    T* begin_;
    // 数据当前大小
    int size_;
    // 数据容量
    int capacity_;
    // 友元结构体 OwnedSlice<T>
    friend struct OwnedSlice<T>;
};

template<typename T>
struct OwnedSlice {
    typedef void (*deleter_t)(Slice<T>);
    // 空的删除函数
    static void _no_delete(Slice<T>) {}
    // 构造函数，默认使用空的删除函数
    OwnedSlice()
    : deleter_(_no_delete) {}
    // 禁用复制构造函数
    OwnedSlice(const OwnedSlice&) = delete;
    // 禁用赋值操作符
    OwnedSlice& operator=(const OwnedSlice&) = delete;
    // 析构函数，释放资源
    ~OwnedSlice() {
        deleter_(slice_);
        // 如果数据大小大于8，则释放动态分配的内存
        if (slice_.size_ > 8) {
            delete [] slice_.begin_;
        }
    }
    // 设置拥有的 Slice 对象和删除器
    void set(Slice<T> to_own, deleter_t deleter = _no_delete) {
        slice_.size_ = slice_.capacity_ = to_own.size();
        // 如果大小大于8，则动态分配内存，否则使用内部的 small_buf
        slice_.begin_ = (slice_.size_ > 8) ? new T[slice_.size_] : &small_buf[0];
        // 复制数据到新的内存中
        std::memcpy(slice_.begin_, to_own.begin(), slice_.size_ * sizeof(T));
        deleter_ = deleter;
    }
    // 返回当前拥有的 Slice 对象
    Slice<T> slice() const {
        return slice_;
    }
private:
    // 拥有的 Slice 对象
    Slice<T> slice_;
    // 删除器函数指针
    deleter_t deleter_;
    // 内部小缓冲区
    T small_buf[8];
};

template<typename T>
// 输出 Slice<T> 对象的重载操作符
inline std::ostream& operator<<(std::ostream& s, const Slice<T>& v) {
    s << "[";
    // 遍历并输出 Slice 中的元素
    for (int i : v.enumerate()) {
        if (i > 0) {
            s << ", ";
        }
        s << v[i];
    }
    s << "]";
    return s;
}

// 引用封装，用于访问 Tensor 对象
struct TensorRef {
    // 默认构造函数，初始化为 nullptr
    TensorRef()
    : impl_(nullptr){}
    // 构造函数，使用给定的 Tensor 对象初始化
    TensorRef(const at::Tensor& t)
    : impl_(t.unsafeGetTensorImpl()) {}
    // 解引用操作符，返回内部的 Tensor 对象引用
    const at::Tensor& operator*() const {
        return *(at::Tensor*)this;
    }
    // 成员访问操作符，返回内部的 Tensor 对象指针
    at::Tensor* operator->() const {
        return (at::Tensor*)this;
    }
    // 转换为布尔值，检查是否为空指针
    operator bool() const {
        return impl_ != nullptr;
    }
private:
    // 内部的 TensorImpl 指针
    at::TensorImpl* impl_;
};

// 定义 Arena 类
constexpr int ARENA_MAX_SIZE = 4096;
constexpr int ALIGNMENT = 8;
struct Arena {
    // 构造函数，初始化已分配的大小为0
    Arena()
    : allocated_(0) {}
    // 分配内存，返回 T 类型的指针
    template<typename T>
    T* allocate(int n) {
        // 如果 n 为0，直接返回 nullptr
        if (!n) {
            return nullptr;
        }
        int to_allocate = sizeof(T) * n;
        // 对齐到 ALIGNMENT 的倍数
        int to_allocate_rounded = ALIGNMENT * ((to_allocate - 1) / ALIGNMENT + 1);
        auto prev_allocated = allocated_;
        allocated_ += to_allocate_rounded;
        // 如果超过了最大允许大小，直接返回一个指向新分配内存的指针
        if (C10_UNLIKELY_OR_CONST(allocated_ > ARENA_MAX_SIZE)) {
            overflow_.emplace_back(new char[to_allocate]);
            return (T*) &overflow_.back()[0];
        }
        // 否则返回指向已分配内存的指针
        return (T*) (buffer_ + prev_allocated);
    }
    // 自动释放 Tensor 对象，将其添加到 autorelease 的列表中
    TensorRef autorelease(at::Tensor s) {
        auto ref = TensorRef(s);
        s.unsafeReleaseTensorImpl();
        ar_tensors_.append(*this, ref);
        return ref;
    }
    // 自动释放 mpy::object 对象，将其添加到 autorelease 的列表中
    mpy::handle autorelease(mpy::object obj) {
        ar_objects_.append(*this, obj);
        obj.release();
        return ar_objects_.back();
    }
    
    // 已分配的总大小
    int allocated_;
    // 溢出的内存块列表
    std::vector<std::unique_ptr<char[]>> overflow_;
    // autorelease 的 Tensor 对象列表
    std::vector<TensorRef> ar_tensors_;
    // autorelease 的 mpy::object 对象列表
    std::vector<mpy::handle> ar_objects_;
    // 内部缓冲区
    alignas(ALIGNMENT) char buffer_[ARENA_MAX_SIZE];
};
    ~Arena() {
        // 析构函数，用于清理对象
        // 遍历 ar_tensors_ 中的每个 TensorRef 对象
        for(TensorRef t: ar_tensors_) {
            // 获取 TensorRef 中的 TensorImpl 对象指针，并释放其内存
            c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(t->unsafeGetTensorImpl());
        }
        // 遍历 ar_objects_ 中的每个 mpy::handle 对象
        for(mpy::handle h: ar_objects_) {
            // 从 ar_objects_ 中移除对象所有权
            mpy::object::steal(h);
        }
    }
// 存储已分配的内存大小
private:
    int64_t allocated_;
    // 缓冲区，用于存储数据，大小为 ARENA_MAX_SIZE
    char buffer_[ARENA_MAX_SIZE];
    // 存储 TensorRef 类型的切片
    Slice<TensorRef> ar_tensors_;
    // 存储 mpy::handle 类型的切片
    Slice<mpy::handle> ar_objects_;
    // 存储 unique_ptr<char[]> 类型的向量，用于处理溢出数据
    std::vector<std::unique_ptr<char[]>> overflow_;
};

// 在 Slice 类模板中定义的 insert 函数
template<typename T>
inline void Slice<T>::insert(Arena& arena, Slice where, Slice to_insert) {
    // 确保插入位置 where 在当前切片内部
    AT_ASSERT(inside(where));
    // 将当前切片保存到 result 中
    Slice result = *this;
    /// b------sb---se-----e,  0----n
    // 定义插入位置的起始地址
    T* body_dest = where.begin();
    // 如果待插入内容和插入位置的大小不相等
    if (where.size() != to_insert.size()) {
        // 计算新的切片大小
        int new_size = size() - where.size() + to_insert.size();
        // 计算尾部的目标地址
        T* tail_dest = where.begin() + to_insert.size();
        // 如果新的大小超过了当前切片的容量
        if (new_size >= capacity_) {
            // 计算新的容量，至少为 8 的倍数
            int new_capacity = new_size ? round2min8(new_size) : 0;
            result.capacity_ = new_capacity;
            // 在 arena 中分配新的存储空间
            result.begin_ = arena.allocate<T>(new_capacity);
            // 计算新的 body_dest 和 tail_dest
            body_dest = result.begin_ + (where.begin() - begin());
            tail_dest = body_dest + to_insert.size();
            // 复制插入位置之前的数据到新的切片中
            std::copy(begin_, begin_ + (where.begin() - begin()), result.begin_);
        }
        // 移动尾部数据到新的位置
        std::memmove(tail_dest, where.end(), sizeof(T)*(end() - where.end()));
        // 更新结果切片的大小
        result.size_ = new_size;
    }

    // 复制待插入数据到目标位置
    std::copy(to_insert.begin(), to_insert.end(), body_dest);
    // 更新当前切片为结果切片
    *this = result;
}

// 在 Slice 类模板中定义的 append 函数
template<typename T>
inline void Slice<T>::append(Arena& arena, T value) {
    // 将当前切片保存到 result 中
    Slice result = *this;
    // 如果当前大小等于容量，需要扩展存储空间
    if (size_ == capacity_) {
        // 计算新的大小，至少为 8 的倍数
        int new_size = size_ ? round2min8(size_)*2 : 8;
        // 在 arena 中分配新的存储空间
        T* n = arena.allocate<T>(new_size);
        // 复制当前数据到新的存储空间
        std::copy(begin_, begin_ + size_, n);
        // 更新 result 的起始地址和容量
        result.begin_ = n;
        result.capacity_ = new_size;
    }
    // 将 value 添加到当前切片的末尾
    result[result.size_++] = std::move(value);
    // 更新当前切片为结果切片
    *this = result;
}

// 在 Slice 类模板中定义的 extend 函数
template<typename T>
inline void Slice<T>::extend(Arena& arena, Slice<T> rhs) {
    // 将当前切片保存到 result 中
    Slice result = *this;
    // 更新 result 的大小为当前大小加上 rhs 的大小
    result.size_ = size_ + rhs.size();
    // 如果结果大小超过容量，则需要扩展存储空间
    if (result.size_ > capacity_) {
        // 计算新的容量，至少为当前大小的 2 倍且是 8 的倍数
        int new_size = round2min8(result.size_);
        // 在 arena 中分配新的存储空间
        T* n = arena.allocate<T>(new_size);
        // 复制当前数据到新的存储空间
        std::copy(begin_, begin_ + size_, n);
        // 更新 result 的起始地址和容量
        result.begin_ = n;
        result.capacity_ = new_size;
    }
    // 将 rhs 的数据添加到当前切片的末尾
    std::copy(rhs.begin(), rhs.end(), result.begin_ + size_);
    // 更新当前切片为结果切片
    *this = result;
}

// 在 Slice 类模板中定义的 Slice 构造函数模板
template<typename T>
template<typename... Args>
Slice<T>::Slice(Arena& arena, Args&&... args) {
    // 计算每个参数的长度
    int lens[] = {_length(args)...};
    size_ = 0;
    // 计算总的大小
    for (auto i : lens) {
        size_ += i;
    }
    // 计算容量，至少为总大小的 8 的倍数
    capacity_ = size_ ? round2min8(size_) : 0;
    // 在 arena 中分配存储空间
    begin_ = arena.allocate<T>(capacity_);
    T* dst_ = begin_;
    // 插入每个参数的数据到当前切片
    T* unused[] = {_insert(dst_, args)...};
    (void) unused;
}
```