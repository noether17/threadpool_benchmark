#pragma once

#include <barrier>
#include <span>
#include <thread>
#include <vector>

#include "physics.hpp"

template <typename AccFunc>
void thread_euler_loop(std::span<Vector3d> pos, std::span<Vector3d> vel,
                       std::span<Vector3d> acc, std::size_t offset, double t0,
                       double tf, double dt, AccFunc acc_func,
                       std::barrier<>& barrier) {
  auto t = t0;
  while (t < tf) {
    barrier.arrive_and_wait();
    acc_func(pos, offset, acc);

    barrier.arrive_and_wait();
    auto thread_size = std::ssize(vel);
    for (auto i = 0; i < thread_size; ++i) {
      pos[i + offset].x += vel[i].x * dt;
      pos[i + offset].y += vel[i].y * dt;
      pos[i + offset].z += vel[i].z * dt;
      vel[i].x += acc[i].x * dt;
      vel[i].y += acc[i].y * dt;
      vel[i].z += acc[i].z * dt;
    }

    t += dt;
  }
}

template <typename AccFunc>
void threaded_euler(std::span<Vector3d> pos, std::span<Vector3d> vel, double t0,
                    double tf, double dt, AccFunc acc_func,
                    std::size_t n_threads) {
  auto system_size = std::ssize(pos);
  auto thread_size = system_size / n_threads;
  auto acc = std::vector<Vector3d>(pos.size());
  auto barrier = std::barrier{static_cast<int>(n_threads)};
  auto threads = std::vector<std::jthread>{};
  for (std::size_t i = 0; i < n_threads; ++i) {
    auto offset = i * thread_size;
    auto thread_vel_portion = vel.subspan(offset, thread_size);
    auto thread_acc_portion =
        std::span<Vector3d>{acc}.subspan(offset, thread_size);
    auto thread_function = [=, &barrier]() {
      thread_euler_loop(pos, thread_vel_portion, thread_acc_portion, offset, t0,
                        tf, dt, acc_func, barrier);
    };
    threads.emplace_back(thread_function);
  }
}
