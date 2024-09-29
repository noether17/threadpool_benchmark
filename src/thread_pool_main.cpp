#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "ThreadPool.hpp"

int main() {
  auto thread_pool = ThreadPool(4);

  auto v = std::vector<int>(31);
  thread_pool.call_parallel_kernel([&v](std::size_t i) { v[i] = i; }, v.size());
  std::this_thread::sleep_for(std::chrono::seconds(1));

  std::cout << "v:";
  for (auto const& i : v) {
    std::cout << " " << i;
  }
  std::cout << '\n';

  return 0;
}
