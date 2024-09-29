#include <iostream>
#include <vector>

#include "ThreadPool.hpp"

int main() {
  auto thread_pool = ThreadPool(4);

  auto v = std::vector<int>(500);
  thread_pool.call_parallel_kernel(
      [](std::size_t i, std::vector<int>& v) { v[i] = i; }, v.size(), v);

  std::cout << "v:";
  for (auto const& i : v) {
    std::cout << " " << i;
  }
  std::cout << '\n';

  return 0;
}
