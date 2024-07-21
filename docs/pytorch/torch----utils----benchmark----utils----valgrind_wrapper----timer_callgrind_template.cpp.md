# `.\pytorch\torch\utils\benchmark\utils\valgrind_wrapper\timer_callgrind_template.cpp`

```py
/*
   C++ template for Timer.collect_callgrind

   This template will be consumed by `cpp_jit.py`, and will replace:
       `GLOBAL_SETUP_TEMPLATE_LOCATION`,
       `SETUP_TEMPLATE_LOCATION`
         and
       `STMT_TEMPLATE_LOCATION`
   sections with user provided statements.
*/

#include <c10/util/irange.h>
#include <callgrind.h>
#include <torch/torch.h>

#include <string>

// Global setup. (e.g. #includes)
// GLOBAL_SETUP_TEMPLATE_LOCATION

#if defined(NVALGRIND)
static_assert(false);
#endif

int main(int argc, char* argv[]) {
  // This file should only be called inside of `Timer`, so we can adopt a
  // very simple and rigid argument parsing scheme.

  // Check if the correct number of arguments are provided
  TORCH_CHECK(argc == 9);

  // Check if the first argument is "--number" and parse its value
  TORCH_CHECK(std::string(argv[1]) == "--number");
  auto number = std::stoi(argv[2]);

  // Check if the third argument is "--number-warmup" or "--number_warmup" and parse its value
  TORCH_CHECK(
      std::string(argv[3]) == "--number-warmup" ||
      std::string(argv[3]) == "--number_warmup");
  auto number_warmup = std::stoi(argv[4]);

  // Check if the fifth argument is "--repeats" and parse its value
  TORCH_CHECK(std::string(argv[5]) == "--repeats");
  auto repeats = std::stoi(argv[6]);

  // Check if the seventh argument is "--number-threads" or "--number_threads" and parse its value
  TORCH_CHECK(
      std::string(argv[7]) == "--number-threads" ||
      std::string(argv[7]) == "--number_threads");
  auto number_threads = std::stoi(argv[8]);
  
  // Set the number of threads for Torch
  torch::set_num_threads(number_threads);

  // Setup
  // SETUP_TEMPLATE_LOCATION

  // Warmup loop
  for (const auto i : c10::irange(number_warmup)) {
    (void)i;
    // STMT_TEMPLATE_LOCATION
  }

  // Main loop
  for (const auto repeat : c10::irange(repeats)) {
    (void)repeat;
    // Start collecting callgrind profiling data
    CALLGRIND_TOGGLE_COLLECT;

    // Loop over the main computation loop
    for (const auto i : c10::irange(number)) {
      (void)i;
      // STMT_TEMPLATE_LOCATION
    }

    // Stop collecting callgrind profiling data
    CALLGRIND_TOGGLE_COLLECT;
    // Dump callgrind statistics
    CALLGRIND_DUMP_STATS;
  }
}
```