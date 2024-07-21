# `.\pytorch\c10\core\Event.h`

```
/**
 * @brief 声明一个不可复制、不可移动、非线程安全的事件类
 *
 * 此类实现了一种与后端无关的可移动事件，不支持复制或移动操作。
 * 事件的设计遵循 CUDA 和 HIP 事件的模式。这些事件可以被流记录和等待，
 * 并且可以被重新记录，每次重新记录相当于创建事件的新版本。
 * 例如，如果在 CPU 时间中，流 X 被要求记录事件 E，
 * 流 Y 在事件 E 上等待，然后流 X 又被要求重新记录事件 E，
 * 那么流 Y 将等待流 X 完成第一次记录而不是第二次，因为它等待的是事件 E 的第一个版本而不是第二个。
 * 查询事件只返回其最新版本的状态。
 *
 * 与此类似的后端无关事件由本类及 impl::InlineEvent 实现。
 * 除了这些事件外，还有一些后端特定的事件，例如 ATen 的 CUDAEvent。
 * 每个类都有其特定的用途。
 *
 * 当后端在编译时未知或不可用时，应使用此 Event 类。
 * 这些通用事件基于 DeviceGuardImpls，类似于 DeviceGuard 和 InlineDeviceGuard。
 * "DeviceGuardImpls" 这个名称不再完全准确，因为这些类实现了通用后端接口的后端特定逻辑。
 *
 * 请参阅 DeviceGuardImplInterface.h 查看所有支持的标志。
 */
struct Event final {
  // Constructors
  /**
   * @brief 构造函数，初始化事件对象
   * @param _device_type 设备类型
   * @param _flag 事件标志，默认为 EventFlag::PYTORCH_DEFAULT
   */
  Event(
      const DeviceType _device_type,
      const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
      : impl_{_device_type, _flag} {}

  // Copy constructor and copy assignment operator (deleted)

  // Move constructor and move assignment operator
  /**
   * @brief 移动构造函数
   */
  Event(Event&&) noexcept = default;
  /**
   * @brief 移动赋值运算符
   */
  Event& operator=(Event&&) noexcept = default;

  // Destructor
  /**
   * @brief 析构函数，默认析构
   */
  ~Event() = default;

  // Getters
  /**
   * @brief 获取事件关联的设备
   * @return 事件关联的设备对象
   */
  Device device() const noexcept {
    return Device(device_type(), device_index());
  }
  /**
   * @brief 获取事件的设备类型
   * @return 设备类型
   */
  DeviceType device_type() const noexcept {
    return impl_.device_type();
  }
  /**
   * @brief 获取事件的设备索引
   * @return 设备索引
   */
  DeviceIndex device_index() const noexcept {
    return impl_.device_index();
  }
  /**
   * @brief 获取事件的标志
   * @return 事件的标志
   */
  EventFlag flag() const noexcept {
    return impl_.flag();
  }
  /**
   * @brief 检查事件是否被标记为记录
   * @return 如果事件被标记为记录则返回 true，否则返回 false
   */
  bool was_marked_for_recording() const noexcept {

    /**
     * @brief 检查事件是否被标记为记录
     * @return 如果事件被标记为记录则返回 true，否则返回 false
     */
    return impl_.was_marked_for_recording();
  }

private:
  impl::InlineEvent impl_;  ///< 实际事件的实现对象
};
    // 返回当前事件是否标记为记录
    return impl_.was_marked_for_recording();
  }

  /**
   * Calls record() if and only if record() has never been called for this
   * event. Note: because Event is not thread-safe recordOnce() may call
   * record() multiple times if called from multiple threads.
   */
  void recordOnce(const Stream& stream) {
    // 调用record()，仅当该事件从未被记录过时调用。注意：由于Event不是线程安全的，如果从多个线程调用recordOnce()，可能会多次调用record()。
    impl_.recordOnce(stream);
  }

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  void record(const Stream& stream) {
    // 增加事件的版本号，并将带有此版本号的作业加入流的工作队列中。
    // 当流处理该作业时，通知所有等待或受阻于该版本事件的流继续，并标记该版本为已记录。
    impl_.record(stream);
  }

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  void block(const Stream& stream) const {
    // 如果事件未被安排记录，则什么也不做。
    // 如果事件先前已被加入记录队列，则插入一个命令到流的工作队列中，以等待此调用时存在的事件版本。
    // 当流到达此命令时，它将停止处理额外的命令，直到该事件版本被标记为已记录。
    impl_.block(stream);
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool query() const {
    // 如果事件从未被安排记录，并且当前版本被标记为已记录，则返回true；否则返回false。
    return impl_.query();
  }

  // 返回此事件与给定事件之间的经过时间。
  double elapsedTime(const Event& event) const {
    return impl_.elapsedTime(event.impl_);
  }

  // 返回事件的唯一标识符。
  void* eventId() const {
    return impl_.eventId();
  }

  // 同步事件。
  void synchronize() const {
    return impl_.synchronize();
  }

 private:
  impl::InlineEvent<impl::VirtualGuardImpl> impl_;
};

} // namespace c10
```