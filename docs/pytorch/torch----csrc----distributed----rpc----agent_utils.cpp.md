# `.\pytorch\torch\csrc\distributed\rpc\agent_utils.cpp`

```
// 引入 fmt 库中的 format.h 和 Torch 库中的 agent_utils.h 头文件
#include <fmt/format.h>
#include <torch/csrc/distributed/rpc/agent_utils.h>

// 定义命名空间 torch::distributed::rpc
namespace torch {
namespace distributed {
namespace rpc {

// 收集各个 worker 的名称与 ID 的映射关系
std::unordered_map<std::string, worker_id_t> collectNames(
    ::c10d::PrefixStore store,   // 使用 ::c10d::PrefixStore 类型的 store 作为前缀存储
    const worker_id_t selfId,    // 当前 worker 的 ID
    const std::string& selfName, // 当前 worker 的名称
    const int worldSize) {       // 总共的 worker 数量
  // 将 selfName 转换成 uint8_t 向量
  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());
  // 将当前 worker 的 ID 和名称存入 store 中
  store.set(std::to_string(selfId), selfNameVector);

  // 创建映射关系的 unordered_map
  std::unordered_map<std::string, worker_id_t> nameToId;
  nameToId.reserve(worldSize);
  nameToId.emplace(selfName, selfId); // 将当前 worker 的名称和 ID 存入映射中

  // 遍历所有 worker
  for (worker_id_t workerId = 0; workerId < worldSize; ++workerId) {
    if (workerId == selfId) {
      continue; // 跳过当前 worker
    }
    // 从 store 中获取指定 workerId 的名称信息
    std::vector<uint8_t> workerNameVector = store.get(std::to_string(workerId));
    // 将获取的名称信息转换为字符串
    std::string workerName(
        (char*)workerNameVector.data(), workerNameVector.size());

    // 检查 worker 名称是否唯一
    TORCH_CHECK(
        nameToId.find(workerName) == nameToId.end(),
        "RPC worker name ",
        workerName,
        " is not unique. Workers ",
        nameToId.find(workerName)->second,
        " and ",
        workerId,
        " share the same name.");

    // 将 worker 的名称和 ID 存入映射中
    nameToId.emplace(workerName, workerId);
  }
  return nameToId; // 返回名称到 ID 的映射关系
}

// 分割字符串函数，根据指定的分隔符进行字符串分割
static std::vector<std::string> splitString(
    const std::string& s,    // 要分割的字符串
    const std::string& delim) {  // 分割符
  std::vector<std::string> tokens; // 存储分割后的子串
  size_t start = 0; // 起始位置
  size_t end;       // 结束位置

  // 迭代查找每个分割符
  while ((end = s.find(delim, start)) != std::string::npos) {
    // 将分割后的子串存入 tokens 中
    tokens.emplace_back(s.substr(start, end - start));
    start = end + delim.length(); // 更新起始位置
  }
  tokens.emplace_back(s.substr(start)); // 存入最后一个子串
  return tokens; // 返回分割后的子串列表
}

// 定义全局常量，表示所有 worker 信息的键
const std::string allWorkerInfosKey = "_ALL_WORKER_INFOS";

// 收集当前 worker 名称与 ID 的映射关系
std::unordered_map<std::string, worker_id_t> collectCurrentNames(
    ::c10d::PrefixStore store,   // 使用 ::c10d::PrefixStore 类型的 store 作为前缀存储
    const worker_id_t selfId,    // 当前 worker 的 ID
    const std::string& selfName) { // 当前 worker 的名称
  // 将 selfName 转换成 uint8_t 向量
  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());

  // 检查 ID 是否已存在，并设置 {ID : NAME}
  std::vector<uint8_t> resultVector = store.compareSet(
      std::to_string(selfId), std::vector<uint8_t>(), selfNameVector);
  TORCH_CHECK(
      resultVector == selfNameVector,
      "RPC worker id ",
      selfId,
      " is not unique. Worker ",
      resultVector,
      " and already has ID and ",
      selfNameVector,
      " cannot be added.");

  // 将当前 worker 的 ID 和名称存入 store 中
  store.set(std::to_string(selfId), selfNameVector);

  // 创建映射关系的 unordered_map
  std::unordered_map<std::string, worker_id_t> nameToId;
  nameToId.emplace(selfName, selfId); // 将当前 worker 的名称和 ID 存入映射中

  // 检查 store 中是否有 worker 名称列表
  bool worker_names_available =
      store.check(std::vector<std::string>{allWorkerInfosKey});
  std::string allWorkerInfos;
  if (worker_names_available) {
    // 获取当前 worker 名称列表
    std::vector<uint8_t> allWorkerInfosKeyVector = store.get(allWorkerInfosKey);
    allWorkerInfos.assign(
        (char*)allWorkerInfosKeyVector.data(), allWorkerInfosKeyVector.size());
  }
    // 将 allWorkerInfosKeyVector 中的数据转换为 std::string
    allWorkerInfos = std::string(
        (char*)allWorkerInfosKeyVector.data(), allWorkerInfosKeyVector.size());
    
    // workerInfos 是以逗号分隔的字符串，以逗号结尾（例如 "Name1-Rank1,Name2-Rank2,Name3-Rank2,"）解析工作者列表。
    if (!allWorkerInfos.empty()) {
      // 将 allWorkerInfos 中最后一个逗号之前的部分提取出来，并按逗号分割成 workerInfoString 列表进行处理
      for (const std::string& workerInfoString : splitString(
               allWorkerInfos.substr(0, allWorkerInfos.size() - 1), ",")) {
        // 将 workerInfoString 按照 '-' 分割，得到工作者名和工作者 ID
        auto workerInfoVec = splitString(workerInfoString, "-");
        std::string workerName = workerInfoVec.at(0);
        int workerId = std::stoi(workerInfoVec.at(1));

        // 检查工作者名是否已经存在于 nameToId 映射中，确保工作者名是唯一的
        TORCH_CHECK(
            nameToId.find(workerName) == nameToId.end(),
            "RPC worker name ",
            workerName,
            " is not unique. Workers ",
            nameToId.find(workerName)->second,
            " and ",
            workerId,
            " share the same name.");

        // 将工作者名和工作者 ID 添加到 nameToId 映射中
        nameToId.emplace(workerName, workerId);
      }
    }
  }
  // 将自身的名称和 ID 添加到工作者列表中
  allWorkerInfos = fmt::format("{}{}-{},", allWorkerInfos, selfName, selfId);
  
  // 将 allWorkerInfos 转换为 uint8_t 类型的向量 allWorkerInfosVector
  std::vector<uint8_t> allWorkerInfosVector(
      (uint8_t*)allWorkerInfos.c_str(),
      (uint8_t*)allWorkerInfos.c_str() + allWorkerInfos.length());
  
  // 使用 store 将 allWorkerInfosVector 存储到 allWorkerInfosKey 键下
  store.set(allWorkerInfosKey, allWorkerInfosVector);

  // 返回 nameToId 映射，其中包含所有工作者的名称和对应的 ID
  return nameToId;
} // namespace rpc
} // namespace distributed
} // namespace torch
```