#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "physics.hpp"
#include "threaded_euler.hpp"

int main(int argc, char* argv[]) {
  // default test parameters
  auto n_threads = 8;
  auto N = 128;  // number of particles (must be multiple of n_threads).
  if (argc > 1) {
    N = std::stoi(argv[1]);  // TODO: Handle input error.
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
  threaded_euler(pos, vel, t0, tf, dt, threaded_gravity, n_threads);
  auto duration = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << "Execution time: " << duration.count() << "ms\n";

  // output
  auto filename = std::stringstream{};
  filename << "n_body_output_" << N << "_particles_" << n_threads
           << "_threads.txt";
  auto output_file = std::ofstream{filename.str()};
  for (auto& p : pos) {
    output_file << '{' << p.x << ',' << p.y << ',' << p.z << '}';
  }
  output_file << '\n';

  return 0;
}
