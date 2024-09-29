#include "physics.hpp"

void threaded_gravity(std::span<Vector3d const> pos, int offset,
                      std::span<Vector3d> acc) {
  for (auto i = offset; i < std::ssize(acc) + offset; ++i) {
    auto& current_acc = acc[i - offset];
    current_acc = Vector3d{};
    for (auto j = 0; j < std::ssize(pos); ++j) {
      if (i == j) {
        continue;
      }
      auto disp = Vector3d{pos[j].x - pos[i].x, pos[j].y - pos[i].y,
                           pos[j].z - pos[i].z};
      auto dist_sq = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
      auto denominator = std::sqrt(dist_sq) * (dist_sq + softening_sq);
      current_acc.x += disp.x / denominator;
      current_acc.y += disp.y / denominator;
      current_acc.z += disp.z / denominator;
    }
  }
}

void threadpool_gravity(std::size_t i, std::span<Vector3d const> pos,
                        std::span<Vector3d> acc) {
  auto& current_acc = acc[i];
  current_acc = Vector3d{};
  for (std::size_t j = 0; j < pos.size(); ++j) {
    if (i == j) {
      continue;
    }
    auto disp =
        Vector3d{pos[j].x - pos[i].x, pos[j].y - pos[i].y, pos[j].z - pos[i].z};
    auto dist_sq = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
    auto denominator = std::sqrt(dist_sq) * (dist_sq + softening_sq);
    current_acc.x += disp.x / denominator;
    current_acc.y += disp.y / denominator;
    current_acc.z += disp.z / denominator;
  }
}
