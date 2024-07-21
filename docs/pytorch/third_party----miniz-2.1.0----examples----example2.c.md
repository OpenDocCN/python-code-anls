# `.\pytorch\third_party\miniz-2.1.0\examples\example2.c`

```py
    sprintf(archive_filename, "%u.txt", i);
    // 根据当前循环的索引值构造文件名，格式为 "%u.txt"
    sprintf(data, "%u %s %u", (N - 1) - i, s_pTest_str, i);
    // 根据当前循环的索引值，以及预设的测试字符串 s_pTest_str，构造文件内容字符串

    // 向压缩存档中添加新文件。注意这是一个原地操作，如果操作失败，你的存档可能会损坏（其中央目录可能不完整）。但可以使用 zip -F 或 -FF 命令来恢复。因此在使用这个操作时要小心。
    // 一个更健壮的方法是将文件读入内存，执行操作，然后将新的存档写入临时文件，最后删除/重命名文件。
    // 向指定的 ZIP 文件添加内存中的数据，并根据需要压缩
    status = mz_zip_add_mem_to_archive_file_in_place(s_Test_archive_filename, archive_filename, data, strlen(data) + 1, s_pComment, (uint16)strlen(s_pComment), MZ_BEST_COMPRESSION);
    // 检查添加操作是否成功，若失败则打印错误消息并退出程序
    if (!status)
    {
      printf("mz_zip_add_mem_to_archive_file_in_place failed!\n");
      return EXIT_FAILURE;
    }
  }

  // 添加一个用于测试的目录条目
  status = mz_zip_add_mem_to_archive_file_in_place(s_Test_archive_filename, "directory/", NULL, 0, "no comment", (uint16)strlen("no comment"), MZ_BEST_COMPRESSION);
  // 检查添加操作是否成功，若失败则打印错误消息并退出程序
  if (!status)
  {
    printf("mz_zip_add_mem_to_archive_file_in_place failed!\n");
    return EXIT_FAILURE;
  }

  // 初始化一个 zip 归档结构体实例并打开指定的 ZIP 文件
  memset(&zip_archive, 0, sizeof(zip_archive));
  status = mz_zip_reader_init_file(&zip_archive, s_Test_archive_filename, 0);
  // 检查初始化操作是否成功，若失败则打印错误消息并退出程序
  if (!status)
  {
    printf("mz_zip_reader_init_file() failed!\n");
    return EXIT_FAILURE;
  }

  // 获取并打印归档中每个文件的信息
  for (i = 0; i < (int)mz_zip_reader_get_num_files(&zip_archive); i++)
  {
    mz_zip_archive_file_stat file_stat;
    // 获取指定索引处文件的统计信息
    if (!mz_zip_reader_file_stat(&zip_archive, i, &file_stat))
    {
       printf("mz_zip_reader_file_stat() failed!\n");
       // 若获取文件信息失败，则结束 ZIP 读取器的使用并退出程序
       mz_zip_reader_end(&zip_archive);
       return EXIT_FAILURE;
    }

    // 打印文件名、注释、未压缩大小、压缩后大小及其是否为目录的信息
    printf("Filename: \"%s\", Comment: \"%s\", Uncompressed size: %u, Compressed size: %u, Is Dir: %u\n", file_stat.m_filename, file_stat.m_comment, (uint)file_stat.m_uncomp_size, (uint)file_stat.m_comp_size, mz_zip_reader_is_file_a_directory(&zip_archive, i));

    // 检查特定文件是否为目录
    if (!strcmp(file_stat.m_filename, "directory/"))
    {
      if (!mz_zip_reader_is_file_a_directory(&zip_archive, i))
      {
        printf("mz_zip_reader_is_file_a_directory() didn't return the expected results!\n");
        // 若检查目录失败，则结束 ZIP 读取器的使用并退出程序
        mz_zip_reader_end(&zip_archive);
        return EXIT_FAILURE;
      }
    }
  }

  // 关闭 ZIP 归档，释放其占用的资源
  mz_zip_reader_end(&zip_archive);

  // 现在验证压缩数据的正确性
  for (sort_iter = 0; sort_iter < 2; sort_iter++)
  {
    memset(&zip_archive, 0, sizeof(zip_archive));
    // 初始化 ZIP 读取器，并根据需要选择是否排序中央目录
    status = mz_zip_reader_init_file(&zip_archive, s_Test_archive_filename, sort_iter ? MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY : 0);
    // 检查初始化操作是否成功，若失败则打印错误消息并退出程序
    if (!status)
    {
      printf("mz_zip_reader_init_file() failed!\n");
      return EXIT_FAILURE;
    }

    for (i = 0; i < N; i++)
    {
      // 格式化生成归档文件名，格式为 "%u.txt"，其中 %u 是当前循环索引 i
      sprintf(archive_filename, "%u.txt", i);
    
      // 格式化生成数据字符串，格式为 "%u %s %u"，其中 (N - 1) - i 是计算得到的值，s_pTest_str 是字符串指针，i 是当前循环索引
      sprintf(data, "%u %s %u", (N - 1) - i, s_pTest_str, i);
    
      // 尝试将归档文件提取到堆上
      p = mz_zip_reader_extract_file_to_heap(&zip_archive, archive_filename, &uncomp_size, 0);
    
      // 如果提取失败，输出错误信息并退出
      if (!p)
      {
        printf("mz_zip_reader_extract_file_to_heap() failed!\n");
        mz_zip_reader_end(&zip_archive);
        return EXIT_FAILURE;
      }
    
      // 确保提取成功，并且提取的数据长度与期望长度相符，且数据内容正确
      if ((uncomp_size != (strlen(data) + 1)) || (memcmp(p, data, strlen(data))))
      {
        printf("mz_zip_reader_extract_file_to_heap() failed to extract the proper data\n");
        mz_free(p);
        mz_zip_reader_end(&zip_archive);
        return EXIT_FAILURE;
      }
    
      // 提取成功的情况下输出提取文件的信息和数据内容
      printf("Successfully extracted file \"%s\", size %u\n", archive_filename, (uint)uncomp_size);
      printf("File data: \"%s\"\n", (const char *)p);
    
      // 释放堆上分配的内存
      mz_free(p);
    }
    
    // 关闭 ZIP 归档，释放任何使用的资源
    mz_zip_reader_end(&zip_archive);
    
    // 循环结束，输出成功信息并返回成功状态
    printf("Success.\n");
    return EXIT_SUCCESS;
}



# 这行代码是一个单独的右大括号 '}'，用于结束一个代码块或语句的范围。
```