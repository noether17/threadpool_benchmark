#include "physics.hpp"

void single_threaded_gravity(std::span<Vector3d> pos, std::span<Vector3d> acc) {
  for (std::size_t i = 0; i < pos.size(); ++i) {
    acc[i] = Vector3d{};
    for (std::size_t j = 0; j < pos.size(); ++j) {
      if (i == j) {
        continue;
      }
      auto disp = Vector3d{pos[j].x - pos[i].x, pos[j].y - pos[i].y,
                           pos[j].z - pos[i].z};
      auto dist_sq = disp.x * disp.x + disp.y * disp.y + disp.z * disp.z;
      auto denominator = std::sqrt(dist_sq) * (dist_sq + softening_sq);
      acc[i].x += disp.x / denominator;
      acc[i].y += disp.y / denominator;
      acc[i].z += disp.z / denominator;
    }
  }
}

void threaded_gravity(std::span<Vector3d const> pos, std::size_t offset,
                      std::span<Vector3d> acc) {
  for (auto i = offset; i < acc.size() + offset; ++i) {
    auto& current_acc = acc[i - offset];
    current_acc = Vector3d{};
    for (std::size_t j = 0; j < pos.size(); ++j) {
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

void atomic_threadpool_gravity(std::size_t i,
                               std::span<AtomicVector3d const> pos,
                               std::span<AtomicVector3d> acc) {
  auto& current_acc = acc[i];
  current_acc.x = 0.0;
  current_acc.y = 0.0;
  current_acc.z = 0.0;
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
