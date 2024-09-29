#pragma once

#include <span>
#include <vector>

#include "physics.hpp"

template <typename AccFunc>
void single_threaded_euler(std::span<Vector3d> pos, std::span<Vector3d> vel,
                           double t0, double tf, double dt, AccFunc acc_func,
                           int) {
  auto t = t0;
  auto acc = std::vector<Vector3d>(pos.size());
  while (t < tf) {
    acc_func(pos, acc);
    for (std::size_t i = 0; i < pos.size(); ++i) {
      pos[i].x += vel[i].x * dt;
      pos[i].y += vel[i].y * dt;
      pos[i].z += vel[i].z * dt;
      vel[i].x += acc[i].x * dt;
      vel[i].y += acc[i].y * dt;
      vel[i].z += acc[i].z * dt;
    }

    t += dt;
  }
}
