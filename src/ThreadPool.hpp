#pragma once

#include <functional>
#include <latch>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t n_threads)
      : m_tasks(n_threads), m_tasks_ready(n_threads) {
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      m_threads.emplace_back([this, thread_id] {
        while (true) {
          auto task = std::function<void()>{};
          for (auto trial = 0; !m_stop and not(m_tasks_ready[thread_id]);
               ++trial) {
            if (trial == 8) {
              trial = 0;
              using namespace std::chrono_literals;
              std::this_thread::yield();
            }
          }
          if (m_stop) {
            break;
          }

          std::swap(task, m_tasks[thread_id]);
          m_tasks_ready[thread_id] = false;
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    m_stop = true;

    for (auto& t : m_threads) {
      t.join();
    }
  }

  template <typename ParallelKernel, typename... Args>
  void call_parallel_kernel(ParallelKernel kernel, std::size_t total_items,
                            Args&&... args) {
    auto n_threads = m_threads.size();
    auto items_per_thread = (total_items + n_threads - 1) / n_threads;

    //// prevent false sharing, assuming that all containers being modified are
    //// aligned on cache lines.
    // auto constexpr cache_line_size = 64;
    // items_per_thread =
    //     ((items_per_thread + cache_line_size - 1) / cache_line_size) *
    //     cache_line_size;

    auto latch = std::latch{static_cast<std::ptrdiff_t>(n_threads)};
    for (std::size_t thread_id = 0; thread_id < n_threads; ++thread_id) {
      auto thread_begin = thread_id * items_per_thread;
      auto thread_end =
          std::min((thread_id + 1) * items_per_thread, total_items);
      m_tasks[thread_id] = [&latch, thread_begin, thread_end, kernel,
                            &args...] {
        for (auto i = thread_begin; i < thread_end; ++i) {
          kernel(i, args...);
        }
        latch.count_down();
      };
      m_tasks_ready[thread_id] = true;
    }

    latch.wait();
  }

 private:
  std::vector<std::jthread> m_threads{};
  std::vector<std::function<void()>> m_tasks{};
  std::vector<std::atomic_bool> m_tasks_ready{};
  std::atomic<bool> m_stop{};
};
