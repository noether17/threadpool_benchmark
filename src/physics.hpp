#pragma once

#include <cmath>
#include <span>

struct Vector3d {
  double x{};
  double y{};
  double z{};
};

auto constexpr G = 6.67e-11;          // gravitational constant
auto constexpr L = 1.0;               // box width
auto constexpr epsilon = 1.0e-3 * L;  // softening parameter
auto constexpr softening_sq = epsilon * epsilon;

// dynamical timescale based on system parameters
auto constexpr characteristic_time(int N, double L) {
  return std::sqrt(L * L * L / (G * N));
}

void threaded_gravity(std::span<Vector3d const> pos, int offset,
                      std::span<Vector3d> acc);
