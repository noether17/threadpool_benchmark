#pragma once

#include <span>

#include "ThreadPool.hpp"
#include "physics.hpp"

template <typename AccFunc>
void threadpool_euler(std::span<Vector3d> pos, std::span<Vector3d> vel,
                      double t0, double tf, double dt, AccFunc acc_func,
                      int n_threads) {
  auto tp = ThreadPool{static_cast<std::size_t>(n_threads)};
  auto acc = std::vector<Vector3d>(pos.size());
  for (auto t = t0; t < tf; t += dt) {
    tp.call_parallel_kernel(acc_func, pos.size(), pos, acc);

    tp.call_parallel_kernel(
        [](std::size_t i, double dt, std::span<Vector3d> pos,
           std::span<Vector3d> vel, std::span<Vector3d const> acc) {
          pos[i].x += vel[i].x * dt;
          pos[i].y += vel[i].y * dt;
          pos[i].z += vel[i].z * dt;
          vel[i].x += acc[i].x * dt;
          vel[i].y += acc[i].y * dt;
          vel[i].z += acc[i].z * dt;
        },
        pos.size(), dt, pos, vel, acc);
  }
}
