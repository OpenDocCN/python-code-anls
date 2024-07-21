# `.\pytorch\torch\nn\parallel\parallel_apply.py`

```
    i: int,
    module: Module,
    input: Any,
    kwargs: Dict[str, Any],
    device: Optional[Union[int, torch.device]] = None,
    stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # 设置是否启用梯度计算
        torch.set_grad_enabled(grad_enabled)
        # 如果设备未指定，则尝试从输入中获取一个变量
        if device is None:
            t = get_a_var(input)
            # 如果没有找到变量，则记录异常信息并返回
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where=f"in replica {i}, no device was provided and no tensor input was found; "
                        "device cannot be resolved"
                    )
                return
            # 获取找到的变量所在的设备
            device = t.get_device()
        # 如果流未指定，则使用当前设备的 CUDA 流
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            # 使用指定的设备和流，以及自动类型转换（如果启用），执行模块计算
            with torch.cuda.device(device), torch.cuda.stream(stream), autocast(
                enabled=autocast_enabled
            ):
                # 避免对 `input` 进行意外的切片操作（如果它是一个张量）
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                # 调用模块，传递输入和关键字参数
                output = module(*input, **kwargs)
            # 记录计算结果到线程安全的结果列表中
            with lock:
                results[i] = output
        except Exception:
            # 捕获异常并记录异常信息到结果列表中
            with lock:
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )

    # 如果模块数量大于1，则启动多线程执行
    if len(modules) > 1:
        # 创建多个线程，每个线程执行一个模块的计算任务
        threads = [
            threading.Thread(
                target=_worker, args=(i, module, input, kwargs, device, stream)
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams)
            )
        ]
        # 启动所有线程
        for thread in threads:
            thread.start()
        # 等待所有线程执行完毕
        for thread in threads:
            thread.join()
    else:
        # 如果只有一个模块，则直接在当前线程执行
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    # 从结果列表中提取每个输入对应的输出结果
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        # 如果输出是异常包装对象，则重新引发异常
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        # 将输出添加到输出列表中
        outputs.append(output)
    # 返回所有模块计算的输出结果列表
    return outputs
```