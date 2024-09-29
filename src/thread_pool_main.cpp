#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "ThreadPool.hpp"
#include "physics.hpp"

template <typename AccFunc>
void thread_pool_euler(std::span<Vector3d> pos, std::span<Vector3d> vel,
                       double t0, double tf, double dt, AccFunc,
                       int n_threads) {
  auto tp = ThreadPool{static_cast<std::size_t>(n_threads)};
  auto acc = std::vector<Vector3d>(pos.size());
  for (auto t = t0; t < tf; t += dt) {
    tp.call_parallel_kernel(
        [](std::size_t i, std::span<Vector3d const> pos,
           std::span<Vector3d> acc) {
          auto& current_acc = acc[i];
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
        },
        pos.size(), pos, acc);

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

int main(int argc, char* argv[]) {
  // default test parameters
  auto n_threads = 8;
  auto N = 512;  // number of particles (should be multiple of 64 * n_threads).
  if (argc > 1) {
    N = std::stoi(argv[1]);
  }
  if (argc > 2) {
    n_threads = std::stoi(argv[2]);
  }

  // initialize state
  auto pos = std::vector<Vector3d>(N);
  auto gen = std::mt19937{0};  // seeded run.
  auto dist = std::uniform_real_distribution<double>(0.0, L);
  for (auto& p : pos) {
    p = Vector3d{dist(gen), dist(gen), dist(gen)};
  }
  auto vel = std::vector<Vector3d>(N);

  // simulate
  auto t0 = 0.0;
  auto tf = characteristic_time(N, L);
  auto dt = 1.0e-3 * (tf - t0);
  auto start = std::chrono::high_resolution_clock::now();
  thread_pool_euler(pos, vel, t0, tf, dt, nullptr, n_threads);
  auto duration = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << "Execution time: " << duration.count() << "ms\n";

  // output
  auto filename = std::stringstream{};
  filename << "threadpool_n_body_output_" << N << "_particles_" << n_threads
           << "_threads.txt";
  auto output_file = std::ofstream{filename.str()};
  for (auto const& p : pos) {
    output_file << '{' << p.x << ',' << p.y << ',' << p.z << '}';
  }
  output_file << '\n';

  return 0;
}
