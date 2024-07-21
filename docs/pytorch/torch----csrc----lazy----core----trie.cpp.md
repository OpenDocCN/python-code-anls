# `.\pytorch\torch\csrc\lazy\core\trie.cpp`

```py
namespace torch {
namespace lazy {
namespace {

void TraverseTrie(TrieNode* node, std::stringstream& ss) {
  // 如果节点为空，则直接返回，不进行遍历操作
  if (!node) {
    return;
  }
  // 如果节点包含 IR 节点，则将节点的唯一标识、操作符名称和命中计数器信息添加到 stringstream 中
  if (node->ir_node) {
    ss << node->unique_id << "[label=\"" << node->ir_node->op().ToString()
       << ", " << node->hit_counter << " hits\"]\n";
  }
  // 遍历当前节点的所有后继节点，将其连接关系添加到 stringstream 中，并递归遍历后继节点
  for (auto& successor : node->successors) {
    ss << node->unique_id << " -> " << successor->unique_id << "\n";
    TraverseTrie(successor.get(), ss);
  }
}

} // namespace

TrieCache* TrieCache::Get() {
  // 使用线程局部存储的方式获取 TrieCache 单例对象
  static thread_local TrieCache* trie = new TrieCache();
  return trie;
}

TrieCache::TrieCache()
    : root_(std::make_shared<TrieNode>()), current_(root_.get()) {}

TrieNode* TrieCache::Current() const {
  // 返回当前 TrieCache 对象的当前节点指针
  return current_;
}

void TrieCache::SetCurrent(
    std::list<std::shared_ptr<TrieNode>>::iterator& iter) {
  auto& successors = current_->successors;
  // 在迭代器 `iter` 被销毁前更新 `current_` 指针
  current_ = (*iter).get();

  // 将当前节点插入其父节点后继列表的最前面
  if (iter != successors.begin()) {
    successors.push_front(std::move(*iter));
    successors.erase(iter);
  }
}

void TrieCache::ResetCurrent() {
  // 将 `current_` 指针重置为根节点
  current_ = root_.get();
}

void TrieCache::Insert(NodePtr ir_node) {
  // 检查当前节点是否为空，如果不为空且存在后继节点，则增加 TrieForked 计数器
  TORCH_CHECK(current_);
  if (!current_->successors.empty()) {
    TORCH_LAZY_COUNTER("TrieForked", 1);
  }
  // 创建一个包含 IR 节点的新节点，并将其添加到当前节点的后继列表的最前面
  auto new_node = std::make_shared<TrieNode>(std::move(ir_node));
  current_->successors.push_front(std::move(new_node));
  // 更新 `current_` 指针指向新插入的节点
  current_ = current_->successors.front().get();
}

void TrieCache::Clear() {
  // 重置 `current_` 指针为根节点，并清空根节点的后继列表
  ResetCurrent();
  // 在根节点级别清空后继列表应该足够，因为所有节点都作为 shared_ptr 创建
  root_->successors.clear();
}

void TrieCache::DumpToDotFile(const std::string& file_name) {
  // 创建一个 stringstream 用于存储 DOT 格式的图形描述
  std::stringstream ss;
  ss << "digraph G {\n";
  // 调用 TraverseTrie 函数遍历整个 Trie，并将遍历结果存储到 stringstream 中
  TraverseTrie(root_.get(), ss);
  ss << "}\n";

  // 将 stringstream 中的图形描述写入到指定文件中
  std::ofstream graph_file(file_name);
  graph_file << ss.str();
}

} // namespace lazy
} // namespace torch
```