#pragma once

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <new>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t n_threads) {
    for (std::size_t i = 0; i < n_threads; ++i) {
      m_threads.emplace_back([this]() {
        while (true) {
          auto task = std::function<void()>{};
          {
            auto lock = std::unique_lock{m_mx};
            m_cv.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });

            if (m_stop and m_tasks.empty()) {
              break;
            }

            task = std::move(m_tasks.front());
            m_tasks.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      auto lock = std::unique_lock{m_mx};
      m_stop = true;
    }

    m_cv.notify_all();

    for (auto& t : m_threads) {
      t.join();
    }
  }

  template <typename ParallelKernel, typename... Args>
  void call_parallel_kernel(ParallelKernel kernel, std::size_t total_items,
                            Args... args) {
    auto n_threads = m_threads.size();
    auto items_per_thread = (total_items + n_threads - 1) / n_threads;
    items_per_thread =
        ((items_per_thread + std::hardware_destructive_interference_size - 1) /
         std::hardware_destructive_interference_size) *
        std::hardware_destructive_interference_size;
    for (std::size_t thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
      auto thread_begin = thread_idx * items_per_thread;
      auto thread_end =
          std::min((thread_idx + 1) * items_per_thread, total_items);
      std::cout << "\nthread_idx: " << thread_idx
                << "\nthread_begin: " << thread_begin
                << "\nthread_end: " << thread_end << '\n';
      {
        auto lock = std::unique_lock{m_mx};
        m_tasks.emplace([thread_begin, thread_end, &kernel, &args...]() {
          for (auto i = thread_begin; i < thread_end; ++i) {
            kernel(i, args...);
          }
        });
      }

      m_cv.notify_one();
    }
  }

 private:
  std::vector<std::jthread> m_threads{};
  std::queue<std::function<void()>> m_tasks{};
  std::mutex m_mx{};
  std::condition_variable m_cv{};
  bool m_stop{};
};
