# `.\pytorch\torch\csrc\jit\tensorexpr\stmt.h`

```
    // 块语句类，继承自 StmtNode<Block>
class TORCH_API Block : public StmtNode<Block> {
 public:
    // 创建一个新的 Block 对象，输入参数为语句的向量
    static BlockPtr make(const std::vector<StmtPtr>& stmts) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        std::vector<StmtPtr> valid_stmts;
        // 过滤掉空语句
        for (auto& stmt : stmts) {
            if (!stmt) {
                continue;
            }
            valid_stmts.push_back(stmt);
        }
        // 如果有效语句为空，返回空指针
        if (valid_stmts.empty()) {
            return nullptr;
        }
        // 返回一个包含有效语句的新的 Block 对象
        return alloc<Block>(valid_stmts);
    }

    // 返回当前 Block 包含的语句数量
    int nstmts() const {
        return stmts_.size();
    }

    // 检查当前 Block 是否为空
    bool empty() const {
        return stmts_.empty();
    }

    // 在 Block 的开头添加一个语句
    void prepend_stmt(StmtPtr s) {
        // 如果待添加的语句已有父节点，抛出异常
        if (s->get_parent()) {
            throw malformed_input(
                "Block prepend Stmt with existing parent", std::move(s));
        }

        // 将语句添加到 Block 开头
        stmts_.push_front(s);
        // 设置当前 Block 为语句的父节点
        set_parent(std::move(s), this);
    }

    // 在 Block 的末尾添加一个语句
    void append_stmt(StmtPtr s) {
        // 如果待添加的语句已有父节点，抛出异常
        if (s->get_parent()) {
            throw malformed_input(
                "Block append Stmt with existing parent", std::move(s));
        }

        // 将语句添加到 Block 末尾
        stmts_.push_back(s);
        // 设置当前 Block 为语句的父节点
        set_parent(std::move(s), this);
    }

    // 在指定语句 `before` 前插入一个新的语句 `s`
    void insert_stmt_before(StmtPtr s, StmtPtr before) {
        // 如果待插入的语句已有父节点，抛出异常
        if (s->get_parent()) {
            throw malformed_input(
                "Block append Stmt with existing parent", std::move(s));
        }

        // 在 Block 中查找 `before` 语句的位置
        auto pos = std::find(stmts_.begin(), stmts_.end(), before);
        // 如果未找到 `before` 语句，抛出异常
        if (pos == stmts_.end()) {
            throw malformed_input(
                "Inserting after statement that is not in block", std::move(s));
        }

        // 在 `before` 语句前插入新的语句 `s`
        stmts_.insert(pos, s);
        // 设置当前 Block 为语句的父节点
        set_parent(std::move(s), this);
    }
    // 将给定的语句 `s` 移动到本块的尾部，并设置其父节点为当前块 `this`
    void append_stmt(StmtPtr s) {
      set_parent(std::move(s), this);
    }
    
    // 在指定语句 `after` 后面插入语句 `s`
    void insert_stmt_after(StmtPtr s, StmtPtr after) {
      if (s->get_parent()) {
        throw malformed_input(
            "Block append Stmt with existing parent", std::move(s));
      }
    
      auto pos = std::find(stmts_.begin(), stmts_.end(), after);
      if (pos == stmts_.end()) {
        throw malformed_input(
            "Inserting after statement that is not in block", std::move(s));
      }
    
      ++pos;
    
      stmts_.insert(pos, s);
      set_parent(std::move(s), this);
    }
    
    // 将旧语句 `old_stmt` 替换为新语句 `new_stmt`
    bool replace_stmt(StmtPtr old_stmt, StmtPtr new_stmt) {
      if (new_stmt->get_parent()) {
        throw malformed_input(
            "Block replace Stmt with existing parent", std::move(new_stmt));
      }
    
      auto pos = std::find(stmts_.begin(), stmts_.end(), old_stmt);
      if (pos == stmts_.end()) {
        return false;
      }
      stmts_.insert(pos, new_stmt); // 插入新语句
      stmts_.erase(pos); // 移除旧语句
      set_parent(std::move(old_stmt), nullptr); // 清除旧语句的父节点
      set_parent(std::move(new_stmt), this); // 设置新语句的父节点为当前块
      return true;
    }
    
    // 克隆当前块，并替换其中的旧语句 `old_stmt` 为新语句 `new_stmt`
    // 如果未找到旧语句，则返回 `nullptr`
    BlockPtr clone_and_replace(StmtPtr old_stmt, StmtPtr new_stmt) {
      if (new_stmt->get_parent()) {
        throw malformed_input(
            "Block replace Stmt with existing parent", std::move(new_stmt));
      }
    
      std::vector<StmtPtr> stmts(stmts_.begin(), stmts_.end());
      std::vector<StmtPtr> cloned_stmts(stmts.size());
      bool found = false;
      for (int i = 0; i < static_cast<int>(stmts.size()); ++i) {
        if (stmts[i] == old_stmt) {
          found = true;
          cloned_stmts[i] = new_stmt;
        } else {
          cloned_stmts[i] = Stmt::clone(stmts[i]);
        }
      }
      if (!found) {
        return nullptr;
      }
      return alloc<Block>(cloned_stmts); // 返回克隆块的指针
    }
    
    // 移除当前块中的语句 `stmt`
    bool remove_stmt(StmtPtr stmt) {
      auto pos = std::find(stmts_.begin(), stmts_.end(), stmt);
      if (pos == stmts_.end()) {
        return false;
      }
    
      set_parent(std::move(stmt), nullptr); // 清除语句的父节点
      stmts_.erase(pos); // 移除语句
      return true;
    }
    
    // 返回当前块中的语句列表
    std::list<StmtPtr> stmts() const {
      return stmts_;
    }
    
    // 清空当前块，同时清除其中所有语句的父节点
    void clear() {
      for (const auto& s : stmts_) {
        set_parent(s, nullptr);
      }
      stmts_.clear();
    }
    
    // 设置当前块的语句列表为给定的语句列表 `stmts`
    void set_stmts(const std::vector<StmtPtr>& stmts) {
      clear();
      init(stmts);
    }
    
    // 构造函数，用给定的语句列表 `stmts` 初始化当前块
    explicit Block(const std::vector<StmtPtr>& stmts) {
      init(stmts);
    }
    
    // 迭代器类型定义
    typedef std::list<StmtPtr>::iterator iterator;
    typedef std::list<StmtPtr>::const_iterator const_iterator;
    
    // 返回当前块的起始迭代器
    iterator begin() {
      return stmts_.begin();
    }
    
    // 返回当前块的起始常量迭代器
    const_iterator begin() const {
      return stmts_.begin();
    }
    
    // 返回当前块的结束迭代器
    iterator end() {
      return stmts_.end();
    }
    
    // 返回当前块的结束常量迭代器
    const_iterator end() const {
  // 返回迭代器指向的语句链表的末尾
  return stmts_.end();
}

// 返回语句链表的第一个语句的指针
StmtPtr front() {
  return stmts_.front();
}

// 返回语句链表的第一个语句的指针（const版本）
StmtPtr front() const {
  return stmts_.front();
}

// 返回语句链表的最后一个语句的指针
StmtPtr back() {
  return stmts_.back();
}

// 返回语句链表的最后一个语句的指针（const版本）
StmtPtr back() const {
  return stmts_.back();
}

// 将另一个语句块中的语句插入到当前语句块的指定位置
void splice(Block::iterator it, BlockPtr other) {
  for (const StmtPtr& s : *other) {
    // 设置每个语句的父节点为当前语句块
    set_parent(s, this);
  }

  // 将另一个语句块中的语句链表插入到当前语句块的指定位置
  stmts_.splice(it, other->stmts_);
}

// 返回两个语句的最近公共父节点（Block类型）
static BlockPtr getSharedParent(StmtPtr p1, StmtPtr p2) {
  // 使用无序集合存储包含p1或p2的所有语句块
  std::unordered_set<BlockPtr> enclosing;

  // 查找包含p1的所有语句块，并加入enclosing集合
  StmtPtr p1_p = std::move(p1);
  while (p1_p) {
    if (BlockPtr b = to<Block>(p1_p)) {
      if (b) {
        enclosing.insert(b);
      }
    }
    p1_p = p1_p->get_parent();
  }

  // 查找包含p2的最近公共语句块
  StmtPtr p2_p = std::move(p2);
  while (p2_p) {
    if (BlockPtr b = to<Block>(p2_p)) {
      if (enclosing.count(b) != 0) {
        return b;
      }
    }
    p2_p = p2_p->get_parent();
  }

  // 如果没有找到最近公共父节点，返回空指针
  return nullptr;
}

// 返回包含指定语句的最近父节点（当前语句块为根节点）
// 如果语句s的父节点不是当前语句块，则向上遍历父节点直到找到当前语句块为止
StmtPtr getEnclosedRoot(StmtPtr s) const {
  while (s && s->get_parent().get() != this) {
    s = s->get_parent();
  }
  return s;
}

private:
std::list<StmtPtr> stmts_;

// 初始化语句块，将给定的语句列表添加到语句链表中
void init(const std::vector<StmtPtr>& stmts) {
  for (const StmtPtr& s : stmts) {
    if (!s) {
      continue;
    }
    if (!s->get_parent()) {
      // 如果语句的父节点为空，则设置其父节点为当前语句块
      // 注意：在构造函数中无法抛出错误，但IR验证器会捕获此类问题
      set_parent(s, this);
    }

    // 将语句添加到语句链表末尾
    stmts_.push_back(s);
  }
}
};

// Store 类继承自 StmtNode<Store>，表示一个存储操作
class TORCH_API Store : public StmtNode<Store> {
 public:
  // 返回存储操作的基础缓冲区句柄
  VarPtr base_handle() const {
    return buf_->base_handle();
  }
  
  // 返回存储操作的索引表达式列表
  std::vector<ExprPtr> indices() const {
    return indices_;
  }
  
  // 返回存储操作的扁平化索引表达式
  ExprPtr flat_index() const {
    TORCH_CHECK(indices_.size() == 1, "Indices haven't been flattened.");
    return indices_[0];
  }
  
  // 返回存储操作的值表达式
  ExprPtr value() const {
    return value_;
  }
  
  // 返回存储操作关联的缓冲区指针
  BufPtr buf() const {
    return buf_;
  }

  // 设置存储操作关联的缓冲区指针
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  // 设置存储操作的索引表达式列表
  void set_indices(std::vector<ExprPtr> indices) {
    indices_ = std::move(indices);
  }

  // 设置存储操作的值表达式
  void set_value(ExprPtr value) {
    value_ = std::move(value);
  }

  // 静态方法：创建一个存储操作，传入缓冲区句柄、索引表达式和值表达式
  static StorePtr make(
      const BufHandle& buf,
      const std::vector<ExprHandle>& indices,
      const ExprHandle& value);

  // 构造函数：初始化存储操作，传入缓冲区指针、索引表达式列表和值表达式
  Store(BufPtr buf, std::vector<ExprPtr> indices, ExprPtr value);

 private:
  BufPtr buf_;               // 存储操作关联的缓冲区指针
  std::vector<ExprPtr> indices_;  // 存储操作的索引表达式列表
  ExprPtr value_;            // 存储操作的值表达式
};

// Allocate 类表示一个分配内存的操作，继承自 StmtNode<Allocate>
// 绑定给定缓冲区变量，内存的生命周期至多为当前程序运行期间，直到显式释放为止。未释放的内存可能被视为错误。
class TORCH_API Allocate : public StmtNode<Allocate> {
 public:
  // 静态方法：创建一个分配内存的操作，传入缓冲区句柄
  static AllocatePtr make(const BufHandle& buf_handle) {
    return alloc<Allocate>(buf_handle.node());
  }

  // 返回缓冲区变量的句柄
  VarPtr buffer_var() const {
    return buf_->base_handle();
  }

  // 返回分配内存操作的数据类型
  Dtype dtype() const {
    return buf_->dtype();
  }

  // 返回分配内存操作的维度列表
  const std::vector<ExprPtr> dims() const {
    return buf_->dims();
  }

  // 返回分配内存操作关联的缓冲区指针
  BufPtr buf() const {
    return buf_;
  }

  // 设置分配内存操作关联的缓冲区指针
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  // 构造函数：初始化分配内存操作，传入缓冲区指针
  explicit Allocate(BufPtr buf) : buf_(std::move(buf)) {}

 private:
  BufPtr buf_;  // 分配内存操作关联的缓冲区指针
  // TODO: add memory types.
};

// PlacementAllocate 是 NNC IR 中 Allocate 操作的一种变体。
// 它不分配内存，而是重用另一个缓冲区的内存。
class TORCH_API PlacementAllocate : public StmtNode<PlacementAllocate> {
 public:
  // 静态方法：创建一个 PlacementAllocate 操作，传入目标缓冲区句柄和要重用的缓冲区句柄
  static PlacementAllocatePtr make(
      const BufHandle& buf_handle,
      const BufHandle& buf_handle_to_reuse) {
    return alloc<PlacementAllocate>(
        buf_handle.node(), buf_handle_to_reuse.node());
  }

  // 返回目标缓冲区的指针
  BufPtr buf() const {
    return buf_;
  }

  // 返回要重用的缓冲区的指针
  BufPtr buf_to_reuse() const {
    return buf_to_reuse_;
  }

  // 设置目标缓冲区的指针
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  // 设置要重用的缓冲区的指针
  void set_buf_to_reuse(BufPtr buf) {
    buf_to_reuse_ = std::move(buf);
  }

  // 构造函数：初始化 PlacementAllocate 操作，传入目标缓冲区指针和要重用的缓冲区指针
  explicit PlacementAllocate(BufPtr buf, BufPtr buf_to_reuse)
      : buf_(std::move(buf)), buf_to_reuse_(std::move(buf_to_reuse)) {}

 private:
  BufPtr buf_;            // 目标缓冲区的指针
  BufPtr buf_to_reuse_;   // 要重用的缓冲区的指针
};

// Free 类表示释放特定缓冲区的操作。未释放的内存被视为错误。
class TORCH_API Free : public StmtNode<Free> {
 public:
  // 静态方法：创建一个释放操作，传入缓冲区句柄
  static FreePtr make(const BufHandle& buf_handle) {
    return alloc<Free>(buf_handle.node());
  }

  // 返回缓冲区变量的句柄
  VarPtr buffer_var() const {
    return buf_->base_handle();
  }

  // 返回释放操作关联的缓冲区指针
  BufPtr buf() const {
    return buf_;
  }

  // 设置释放操作关联的缓冲区指针
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  explicit Free(BufPtr buf) : buf_(std::move(buf)) {}

 private:
  BufPtr buf_;



    buf_ = std::move(buf);

将参数 `buf` 移动给成员变量 `buf_`。


  }

  explicit Free(BufPtr buf) : buf_(std::move(buf)) {}

构造函数 `Free`，接受一个 `BufPtr` 类型的参数 `buf`，并将其移动给成员变量 `buf_`。使用 `explicit` 关键字来确保只有显式调用才能进行类型转换。


 private:
  BufPtr buf_;

私有成员变量 `buf_`，类型为 `BufPtr`，用于保存传入的缓冲区指针。
};

// 自定义类 FreeExt，继承自 StmtNode<FreeExt>
class TORCH_API FreeExt : public StmtNode<FreeExt> {
 public:
  // 创建 FreeExt 对象的静态方法，接受一个 BufHandle 的向量作为参数
  static FreeExtPtr make(const std::vector<BufHandle>& bufs);

  // 返回 bufs_ 成员变量，即 BufPtr 的向量
  std::vector<BufPtr> bufs() const {
    return bufs_;
  }

  // 设置 bufs_ 成员变量，接受一个 BufPtr 的向量作为参数
  void set_bufs(std::vector<BufPtr> bufs) {
    bufs_ = std::move(bufs);
  }

  // 构造函数，接受一个 BufPtr 的向量作为参数，初始化 bufs_ 成员变量
  explicit FreeExt(std::vector<BufPtr> bufs) : bufs_(std::move(bufs)) {}

 private:
  std::vector<BufPtr> bufs_; // 保存 BufPtr 的向量
};

// 自定义类 Let，继承自 StmtNode<Let>
class TORCH_API Let : public StmtNode<Let> {
 public:
  // 创建 Let 对象的静态方法，接受 VarHandle 和 ExprHandle 作为参数
  static LetPtr make(const VarHandle& var, const ExprHandle& val) {
    return alloc<Let>(var.node(), val.node());
  }

  // 构造函数，接受 VarPtr 和 ExprPtr 作为参数，初始化 var_ 和 val_ 成员变量
  Let(VarPtr var, ExprPtr val) : var_(std::move(var)), val_(std::move(val)) {}

  // 返回 var_ 成员变量，即 VarPtr 对象
  VarPtr var() const {
    return var_;
  }

  // 返回 val_ 成员变量，即 ExprPtr 对象
  ExprPtr value() const {
    return val_;
  }

  // 设置 var_ 成员变量，接受 VarPtr 对象作为参数
  void set_var(VarPtr var) {
    var_ = std::move(var);
  }

  // 设置 val_ 成员变量，接受 ExprPtr 对象作为参数
  void set_val(ExprPtr val) {
    val_ = std::move(val);
  }

 private:
  VarPtr var_;   // 保存 VarPtr 对象
  ExprPtr val_;  // 保存 ExprPtr 对象
};

// 自定义类 Cond，继承自 StmtNode<Cond>
class TORCH_API Cond : public StmtNode<Cond> {
 public:
  // 创建 Cond 对象的静态方法，接受 ExprHandle、StmtPtr 和 StmtPtr 作为参数
  static CondPtr make(
      const ExprHandle& condition,
      StmtPtr true_stmt,
      StmtPtr false_stmt) {
    return alloc<Cond>(condition.node(), true_stmt, false_stmt);
  }

  // 返回 condition_ 成员变量，即 ExprPtr 对象
  ExprPtr condition() const {
    return condition_;
  }

  // 返回 true_stmt_ 成员变量，即 BlockPtr 对象
  BlockPtr true_stmt() const {
    return true_stmt_;
  }

  // 返回 false_stmt_ 成员变量，即 BlockPtr 对象
  BlockPtr false_stmt() const {
    return false_stmt_;
  }

  // 设置 condition_ 成员变量，接受 ExprPtr 对象作为参数
  void set_condition(ExprPtr condition) {
    condition_ = std::move(condition);
  }

  // 设置 true_stmt_ 成员变量，接受 StmtPtr 对象作为参数
  void set_true_stmt(StmtPtr true_stmt) {
    if (true_stmt) {
      BlockPtr b = to<Block>(true_stmt);
      if (!b) {
        b = alloc<Block>(std::vector<StmtPtr>({std::move(true_stmt)}));
      }
      true_stmt_ = b;
      set_parent(true_stmt_, this);
    }
  }

  // 设置 false_stmt_ 成员变量，接受 StmtPtr 对象作为参数
  void set_false_stmt(StmtPtr false_stmt) {
    if (false_stmt) {
      BlockPtr b = to<Block>(false_stmt);
      if (!b) {
        b = alloc<Block>(std::vector<StmtPtr>({std::move(false_stmt)}));
      }
      false_stmt_ = b;
      set_parent(false_stmt_, this);
    }
  }

  // 构造函数，接受 ExprPtr、StmtPtr 和 StmtPtr 作为参数，初始化 condition_、true_stmt_ 和 false_stmt_ 成员变量
  Cond(ExprPtr condition, StmtPtr true_stmt, StmtPtr false_stmt)
      : condition_(std::move(condition)) {
    set_true_stmt(std::move(true_stmt));
    set_false_stmt(std::move(false_stmt));
  }

  // 克隆对象，替换 true_stmt_ 和 false_stmt_ 成员变量，并返回新的 Cond 对象
  CondPtr cloneWithNewBodies(StmtPtr true_stmt, StmtPtr false_stmt) {
    return alloc<Cond>(condition_, true_stmt, false_stmt);
  }

  // 克隆对象，替换 true_stmt_ 成员变量，并返回新的 Cond 对象
  CondPtr cloneWithNewBody(StmtPtr true_stmt) {
    return alloc<Cond>(condition_, true_stmt, nullptr);
  }

 private:
  ExprPtr condition_;   // 保存 ExprPtr 对象
  BlockPtr true_stmt_ = nullptr;   // 保存 true 分支的 BlockPtr 对象，默认为 nullptr
  BlockPtr false_stmt_ = nullptr;  // 保存 false 分支的 BlockPtr 对象，默认为 nullptr
};

// 自定义类 LoopOptions
class TORCH_API LoopOptions {
 public:
  // 枚举值定义
  enum {
    IDX_UNSET = -1,
    IDX_X = 0,
    IDX_Y = 1,
    IDX_Z = 2,
    IDX_W = 3,
    IDX_MAX = IDX_W,
  };
  
  // 判断是否有 GPU 块索引
  bool is_gpu_block_index() const {
    return gpu_block_index_ != IDX_UNSET;
  }

  // 返回 GPU 块索引值
  int gpu_block_index() const {
    return gpu_block_index_;
  }

  // 返回 GPU 块索引值的字符串形式
  std::string gpu_block_index_str() const {
    if (!is_gpu_block_index()) {
      throw malformed_input("Has no GPU block index");
    }
    // 定义一个静态常量数组，包含 GPU 块索引的名称
    static const char* kBlockIndexNames[] = {
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "blockIdx.w",
    };

    // 检查 GPU 块索引是否在有效范围内，如果不在范围内则抛出异常
    if (gpu_block_index_ < IDX_X || gpu_block_index_ > IDX_MAX) {
      throw malformed_input("invalid GPU block index");
    }

    // 返回当前 GPU 块索引对应的名称
    return kBlockIndexNames[gpu_block_index_];
  }

  // 设置 GPU 块索引值
  void set_gpu_block_index(int index) {
    // 如果索引为 IDX_UNSET，则将 GPU 块索引设置为未设置状态
    if (index == IDX_UNSET) {
      gpu_block_index_ = IDX_UNSET;
    }

    // 如果当前正在处理 GPU 线程索引，则抛出异常，因为不能同时设置 GPU 块和线程索引
    if (is_gpu_thread_index()) {
      throw std::runtime_error("Cannot set both gpu block and thread index");
    }

    // 如果已经设置了 GPU 块索引且试图设置不同的索引值，则抛出异常
    if (is_gpu_block_index() && gpu_block_index() != index) {
      throw std::runtime_error("Cannot set a previously set block index");
    }

    // 设置 GPU 块索引值
    gpu_block_index_ = index;
  }

  // 返回是否设置了 GPU 线程索引
  bool is_gpu_thread_index() const {
    return gpu_thread_index() != IDX_UNSET;
  }

  // 返回当前 GPU 线程索引值
  int gpu_thread_index() const {
    return gpu_thread_index_;
  }

  // 返回当前 GPU 线程索引值的字符串表示
  std::string gpu_thread_index_str() const {
    // 如果未设置 GPU 线程索引，则抛出异常
    if (!is_gpu_thread_index()) {
      throw malformed_input("has no GPU thread index");
    }

    // 定义一个静态常量数组，包含 GPU 线程索引的名称
    static const char* kThreadIndexNames[] = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z", "threadIdx.w"};

    // 检查 GPU 线程索引是否在有效范围内，如果不在范围内则抛出异常
    if (gpu_thread_index_ < IDX_X || gpu_thread_index_ > IDX_MAX) {
      throw malformed_input("invalid GPU thread index");
    }

    // 返回当前 GPU 线程索引对应的名称
    return kThreadIndexNames[gpu_thread_index_];
  }

  // 设置 GPU 线程索引值
  void set_gpu_thread_index(int index) {
    // 如果索引为 IDX_UNSET，则将 GPU 线程索引设置为未设置状态
    if (index == IDX_UNSET) {
      gpu_thread_index_ = IDX_UNSET;
    }

    // 如果当前已经设置了 GPU 块索引，则抛出异常，因为不能同时设置 GPU 线程和块索引
    if (is_gpu_block_index()) {
      throw std::runtime_error("Cannot set both gpu thread and block index");
    }

    // 如果已经设置了 GPU 线程索引且试图设置不同的索引值，则抛出异常
    if (is_gpu_thread_index() && gpu_thread_index() != index) {
      throw std::runtime_error("Cannot set a previously set thread index");
    }

    // 设置 GPU 线程索引值
    gpu_thread_index_ = index;
  }

  // 设置为并行处理状态
  void set_parallel() {
    is_parallel_ = true;
  }

  // 返回是否处于并行处理状态
  bool is_parallel() const {
    return is_parallel_;
  }

  // 根据当前对象的状态返回对应的字符串表示
  std::string ToString() const {
    // 如果已经设置了 GPU 块索引，则返回 GPU 块索引的字符串表示
    if (is_gpu_block_index()) {
      return gpu_block_index_str();
    } else if (is_gpu_thread_index()) {
      // 如果已经设置了 GPU 线程索引，则返回 GPU 线程索引的字符串表示
      return gpu_thread_index_str();
    } else if (is_parallel()) {
      // 如果处于并行处理状态，则返回 "parallel"
      return "parallel";
    }
    // 如果未设置任何索引并且不处于并行处理状态，则返回空字符串
    return "";
  }

  // 返回当前对象是否处于默认状态（即未设置任何 GPU 索引且不处于并行处理状态）
  bool isDefault() const {
    return gpu_block_index_ == IDX_UNSET && gpu_thread_index_ == IDX_UNSET &&
        !is_parallel_;
  }

  // 设置输入到张量缓冲映射关系
  void set_buffer_mapping(const std::unordered_map<std::string, BufPtr>& map) {
    map_input_to_tensor_bufs_ = map;
  }

  // 返回输入到张量缓冲映射关系
  std::unordered_map<std::string, BufPtr> get_buffer_mapping() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  // 成员变量，用于保存 GPU 块索引、GPU 线程索引、是否并行处理状态以及输入到张量缓冲映射关系
  int gpu_block_index_{IDX_UNSET};
  int gpu_thread_index_{IDX_UNSET};
  bool is_parallel_{false};
  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
};

// For 类，继承自 StmtNode<For>
class TORCH_API For : public StmtNode<For> {
 public:
  // 返回循环变量 var_
  VarPtr var() const {
    return var_;
  }
  // 返回循环起始表达式 start_
  ExprPtr start() const {
    return start_;
  }
  // 返回循环终止表达式 stop_
  ExprPtr stop() const {
    return stop_;
  }
  // 返回循环体 body_
  BlockPtr body() const {
    return body_;
  }
  // 静态工厂方法，创建 For 循环对象
  static ForPtr make(
      const VarHandle& var,
      const ExprHandle& start,
      const ExprHandle& stop,
      StmtPtr body) {
    if (!body) {
      return nullptr;
    }
    return alloc<For>(var.node(), start.node(), stop.node(), body);
  }
  // 带循环选项的静态工厂方法，创建 For 循环对象
  static ForPtr make(
      const VarHandle& var,
      const ExprHandle& start,
      const ExprHandle& stop,
      StmtPtr body,
      const LoopOptions& loop_options) {
    if (!body) {
      return nullptr;
    }
    return alloc<For>(
        var.node(), start.node(), stop.node(), body, loop_options);
  }
  // 返回循环选项 loop_options_
  const LoopOptions loop_options() const {
    return loop_options_;
  }

  // For 循环构造函数，初始化循环变量 var_、起始表达式 start_、终止表达式 stop_ 和循环体 body_
  For(VarPtr var, ExprPtr start, ExprPtr stop, StmtPtr body)
      : var_(std::move(var)), start_(std::move(start)), stop_(std::move(stop)) {
    // 将 body 转换为 BlockPtr 类型
    BlockPtr b = to<Block>(body);
    if (!b) {
      // 如果 body 不是 BlockPtr 类型，则创建一个只包含 body 的 BlockPtr
      b = alloc<Block>(std::vector<StmtPtr>({std::move(body)}));
    }
    body_ = b;
    // 设置 body_ 的父节点为当前 For 对象
    set_parent(body_, this);
  }

  // 带循环选项的 For 循环构造函数，初始化循环变量 var_、起始表达式 start_、终止表达式 stop_、循环体 body_ 和循环选项 loop_options_
  For(VarPtr var,
      ExprPtr start,
      ExprPtr stop,
      StmtPtr body,
      LoopOptions loop_options)
      : var_(var),
        start_(start),
        stop_(stop),
        loop_options_(std::move(loop_options)) {
    // 检查参数的有效性
    if (!var) {
      throw malformed_input("invalid Var in For loop");
    } else if (!start) {
      throw malformed_input("invalid Start in For loop");
    } else if (!stop) {
      throw malformed_input("invalid Stop in For loop");
    } else if (!body || body->get_parent()) {
      throw malformed_input("invalid Body in For loop");
    }

    // 将 body 转换为 BlockPtr 类型
    BlockPtr b = to<Block>(body);
    if (!b) {
      // 如果 body 不是 BlockPtr 类型，则创建一个只包含 body 的 BlockPtr
      b = alloc<Block>(std::vector<StmtPtr>({std::move(body)}));
    }
    body_ = b;
    // 设置 body_ 的父节点为当前 For 对象
    set_parent(body_, this);
  }

  // 设置 GPU 块索引
  void set_gpu_block_index(int block_index) {
    loop_options_.set_gpu_block_index(block_index);
  }

  // 设置 GPU 线程索引
  void set_gpu_thread_index(int thread_index) {
    loop_options_.set_gpu_thread_index(thread_index);
  }

  // 设置并行执行标志
  void set_parallel() {
    loop_options_.set_parallel();
  }

  // 返回是否并行执行
  bool is_parallel() const {
    return loop_options_.is_parallel();
  }

  // 设置缓冲映射
  void set_buffer_map(const std::unordered_map<std::string, BufPtr>& map) {
    loop_options_.set_buffer_mapping(map);
  }

  // 克隆并替换循环体
  ForPtr cloneWithNewBody(StmtPtr body) const {
    return alloc<For>(var_, start_, stop_, body, loop_options_);
  }

  // 移除循环体并返回
  BlockPtr removeBody() {
    auto res = body_;
    set_parent(res, nullptr);
    body_ = nullptr;
    return res;
  }

  // 设置循环体
  void set_body(StmtPtr body) {
    // 将 body 转换为 BlockPtr 类型
    BlockPtr b = to<Block>(body);
    if (!b) {
      // 如果 body 不是 BlockPtr 类型，则创建一个只包含 body 的 BlockPtr
      b = alloc<Block>(std::vector<StmtPtr>({std::move(body)}));
    }
    body_ = b;
    // 设置 body_ 的父节点为当前 For 对象
    set_parent(body_, this);
  }

  // 设置起始表达式
  void set_start(ExprPtr start) {
    start_ = std::move(start);
  }

  // 设置终止表达式
  void set_stop(ExprPtr stop) {
  // 将参数 stop 移动到成员变量 stop_ 中
  stop_ = std::move(stop);
}

void set_var(VarPtr var) {
  // 将 var 参数移动到成员变量 var_ 中
  var_ = std::move(var);
}

private:
VarPtr var_;         // 用于存储变量的智能指针
ExprPtr start_;      // 循环开始表达式的智能指针
ExprPtr stop_;       // 循环结束表达式的智能指针
BlockPtr body_;      // 循环体的智能指针
LoopOptions loop_options_;  // 循环选项结构体
};

// 表示具有原子加法功能的后端特定的IR节点。
// 这种节点只会在GPU后端的内部显示。
// TODO: 将其移到内部IR中。
// TODO: 使IR节点可扩展。
class TORCH_API AtomicAdd : public StmtNode<AtomicAdd> {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数，初始化 AtomicAdd 对象。
  AtomicAdd(BufPtr buf, std::vector<ExprPtr> indices, ExprPtr value)
      : buf_(std::move(buf)),
        indices_(std::move(indices)),
        value_(std::move(value)) {}

  // 返回基础句柄（base_handle）的变量指针。
  VarPtr base_handle() const {
    return buf_->base_handle();
  }

  // 返回缓冲区指针。
  BufPtr buf() const {
    return buf_;
  }

  // 返回平坦索引表达式指针。
  ExprPtr flat_index() const {
    TORCH_CHECK(indices_.size() == 1, "Indices haven't been flattened.");
    return indices_[0];
  }

  // 返回值表达式指针。
  ExprPtr value() const {
    return value_;
  }

  // 返回索引表达式的向量。
  const std::vector<ExprPtr>& indices() const {
    return indices_;
  }

  // 设置缓冲区。
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  // 设置索引向量。
  void set_indices(std::vector<ExprPtr> indices) {
    indices_ = std::move(indices);
  }

  // 设置值表达式。
  void set_value(ExprPtr value) {
    value_ = std::move(value);
  }

 private:
  BufPtr buf_;                   // 缓冲区指针
  std::vector<ExprPtr> indices_; // 索引表达式向量
  ExprPtr value_;                // 值表达式指针
};

// 表示同步线程的IR节点。
class TORCH_API SyncThreads : public StmtNode<SyncThreads> {
 public:
  SyncThreads() = default;
};

/*
 * ExternalCall 语句表示对外部函数的调用，该函数将计算输出缓冲区的内容。
 * ExternalCall 语句包括：
 *   1) 输出缓冲区 - 将被调用初始化的缓冲区
 *   2) 外部函数名称 - 从NNC函数注册表中查找实际要调用的函数的键
 *   3) 缓冲区参数 - 函数使用的输入缓冲区
 *   4) 非缓冲区参数 - 传递给函数的标量参数
 *
 * 示例：
 *   A = nnc_conv2d(buf_args={Input, Weight, Bias}, args={1})
 * 这里 'A' 是输出缓冲区，"nnc_conv2d" 是函数名称，缓冲区参数是 'Input'、'Weight' 和 'Bias'，
 * 还有一个非缓冲区参数 - 1。
 *
 * 标量参数的语义完全由外部函数的实现定义。
 */
class TORCH_API ExternalCall : public StmtNode<ExternalCall> {
 public:
  // 创建 ExternalCall 对象的静态工厂方法。
  static ExternalCallPtr make(
      BufHandle buf,
      const std::string& func_name,
      const std::vector<BufHandle>& buf_args,
      const std::vector<ExprHandle>& args);

  // 返回缓冲区指针。
  BufPtr buf() const {
    return buf_;
  }

  // 返回函数名称。
  std::string func_name() const {
    return func_name_;
  }

  // 返回缓冲区参数的向量。
  std::vector<BufPtr> buf_args() const {
    return buf_args_;
  }

  // 返回参数表达式的向量。
  std::vector<ExprPtr> args() const {
    return args_;
  }

  // 设置缓冲区。
  void set_buf(BufPtr buf) {
    buf_ = std::move(buf);
  }

  // 设置缓冲区参数。
  void set_buf_args(std::vector<BufPtr> buf_args) {
    buf_args_ = std::move(buf_args);
  }

  // 设置参数表达式。
  void set_args(std::vector<ExprPtr> args) {
    args_ = std::move(args);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ExternalCall(
      BufPtr buf,
      std::string func_name,
      std::vector<BufPtr> buf_args,
      std::vector<ExprPtr> args)
      : buf_(std::move(buf)),  // 将传入的 buf 参数移动赋值给类的 buf_ 成员变量
        func_name_(std::move(func_name)),  // 将传入的 func_name 参数移动赋值给类的 func_name_ 成员变量
        buf_args_(std::move(buf_args)),  // 将传入的 buf_args 参数移动赋值给类的 buf_args_ 成员变量
        args_(std::move(args)) {}  // 将传入的 args 参数移动赋值给类的 args_ 成员变量

 private:
  BufPtr buf_;  // 用于保存 buf 参数的指针
  std::string func_name_;  // 用于保存 func_name 参数的字符串
  std::vector<BufPtr> buf_args_;  // 用于保存 buf_args 参数的向量，存储了 BufPtr 类型的指针
  std::vector<ExprPtr> args_;  // 用于保存 args 参数的向量，存储了 ExprPtr 类型的指针
};

// 定义一个名为 ExternalCallWithAlloc 的类，继承自 StmtNode<ExternalCallWithAlloc>
class TORCH_API ExternalCallWithAlloc : public StmtNode<ExternalCallWithAlloc> {
 public:
  // 静态方法，用于创建 ExternalCallWithAlloc 对象
  static ExternalCallWithAllocPtr make(
      const std::string& func_name,  // 函数名
      const std::vector<BufHandle>& buf_out_args,  // 输出缓冲区参数
      const std::vector<BufHandle>& buf_args,  // 缓冲区参数
      const std::vector<ExprHandle>& args);  // 表达式参数

  // 返回输出缓冲区参数的向量
  std::vector<BufPtr> buf_out_args() const {
    return buf_out_args_;
  }

  // 返回函数名
  std::string func_name() const {
    return func_name_;
  }

  // 返回缓冲区参数的向量
  std::vector<BufPtr> buf_args() const {
    return buf_args_;
  }

  // 返回表达式参数的向量
  std::vector<ExprPtr> args() const {
    return args_;
  }

  // 设置输出缓冲区参数
  void set_buf_out_args(std::vector<BufPtr> buf_out_args) {
    buf_out_args_ = std::move(buf_out_args);
  }

  // 设置缓冲区参数
  void set_buf_args(std::vector<BufPtr> buf_args) {
    buf_args_ = std::move(buf_args);
  }

  // 设置表达式参数
  void set_args(std::vector<ExprPtr> args) {
    args_ = std::move(args);
  }

  // 构造函数，初始化 ExternalCallWithAlloc 对象
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ExternalCallWithAlloc(
      std::string func_name,
      std::vector<BufPtr> buf_out_args,
      std::vector<BufPtr> buf_args,
      std::vector<ExprPtr> args)
      : func_name_(std::move(func_name)),
        buf_out_args_(std::move(buf_out_args)),
        buf_args_(std::move(buf_args)),
        args_(std::move(args)) {}

 private:
  std::string func_name_;  // 函数名
  std::vector<BufPtr> buf_out_args_;  // 输出缓冲区参数的向量
  std::vector<BufPtr> buf_args_;  // 缓冲区参数的向量
  std::vector<ExprPtr> args_;  // 表达式参数的向量
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```