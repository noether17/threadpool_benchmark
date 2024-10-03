#pragma once

#include <atomic>
#include <cmath>
#include <span>

struct Vector3d {
  double x{};
  double y{};
  double z{};
};

// Assumption is that there will be a latch separating operations which read an
// array of AtomicVector3d from operations which write to an array of
// AtomicVector3d. Given this assumption, it is not necessary for write
// operations to be performed as transactions on the full object. It is only
// necessary that write operations to each component do not interfere with one
// another.
struct AtomicVector3d {
  std::atomic<double> x{};
  std::atomic<double> y{};
  std::atomic<double> z{};
};

auto constexpr G = 6.67e-11;          // gravitational constant
auto constexpr L = 1.0;               // box width
auto constexpr epsilon = 1.0e-3 * L;  // softening parameter
auto constexpr softening_sq = epsilon * epsilon;

// dynamical timescale based on system parameters
auto constexpr characteristic_time(int N, double L) {
  return std::sqrt(L * L * L / (G * N));
}

void single_threaded_gravity(std::span<Vector3d> pos, std::span<Vector3d> acc);

void threaded_gravity(std::span<Vector3d const> pos, std::size_t offset,
                      std::span<Vector3d> acc);

void threadpool_gravity(std::size_t i, std::span<Vector3d const> pos,
                        std::span<Vector3d> acc);

void atomic_threadpool_gravity(std::size_t i,
                               std::span<AtomicVector3d const> pos,
                               std::span<AtomicVector3d> acc);
