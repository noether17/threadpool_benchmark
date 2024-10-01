#pragma once

#include <execution>
#include <span>
#include <vector>

#include "physics.hpp"

template <typename AccFunc>
void parallel_algo_euler(std::span<Vector3d> pos, std::span<Vector3d> vel,
                         double t0, double tf, double dt, AccFunc, int) {
  auto acc = std::vector<Vector3d>(pos.size());
  for (auto t = t0; t < tf; t += dt) {
    std::for_each(
        std::execution::par_unseq, acc.begin(), acc.end(),
        [&pos, acc_data = acc.data()](Vector3d& acc_i) {
          auto i = static_cast<std::size_t>(&acc_i - acc_data);
          acc_i = Vector3d{};
          for (std::size_t j = 0; j < pos.size(); ++j) {
            if (i == j) {
              continue;
            }
            auto disp = Vector3d{pos[j].x - pos[i].x, pos[j].y - pos[i].y,
                                 pos[j].z - pos[i].z};
            auto dist_sq = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
            auto denominator = std::sqrt(dist_sq) * (dist_sq + softening_sq);
            acc_i.x += disp.x / denominator;
            acc_i.y += disp.y / denominator;
            acc_i.z += disp.z / denominator;
          }
        });

    std::for_each(std::execution::par_unseq, pos.begin(), pos.end(),
                  [dt, &vel, pos_data = pos.data()](Vector3d& pos_i) {
                    auto i = static_cast<std::size_t>(&pos_i - pos_data);
                    pos_i.x += vel[i].x * dt;
                    pos_i.y += vel[i].y * dt;
                    pos_i.z += vel[i].z * dt;
                  });

    std::for_each(std::execution::par_unseq, vel.begin(), vel.end(),
                  [dt, &acc, vel_data = vel.data()](Vector3d& vel_i) {
                    auto i = static_cast<std::size_t>(&vel_i - vel_data);
                    vel_i.x += acc[i].x * dt;
                    vel_i.y += acc[i].y * dt;
                    vel_i.z += acc[i].z * dt;
                  });
  }
}
