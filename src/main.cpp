#include <barrier>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <thread>
#include <vector>

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

void threaded_euler(std::span<Vector3d> pos, std::span<Vector3d> vel, double t0,
                    double tf, double dt, int n_threads) {
  auto system_size = std::ssize(pos);
  auto thread_size = system_size / n_threads;
  auto acc = std::vector<Vector3d>(pos.size());
  auto barrier = std::barrier{n_threads};
  auto threads = std::vector<std::jthread>{};
  for (auto i = 0; i < n_threads; ++i) {
    auto offset = i * thread_size;
    auto thread_vel_portion = vel.subspan(offset, thread_size);
    auto thread_acc_portion =
        std::span<Vector3d>{acc}.subspan(offset, thread_size);
    auto thread_function = [=, &barrier]() {
      thread_euler_loop(pos, thread_vel_portion, thread_acc_portion, offset, t0,
                        tf, dt, threaded_gravity, barrier);
    };
    threads.emplace_back(thread_function);
  }
}

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
  threaded_euler(pos, vel, t0, tf, dt, n_threads);
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
