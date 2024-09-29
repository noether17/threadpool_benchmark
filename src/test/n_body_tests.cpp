#include <gtest/gtest.h>

#include <fstream>
#include <random>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "physics.hpp"
#include "threaded_euler.hpp"
#include "threadpool_euler.hpp"

auto initialize_state(std::size_t N) {
  auto pos = std::vector<Vector3d>(N);
  auto gen = std::mt19937{0};  // seeded run.
  auto dist = std::uniform_real_distribution<double>(0.0, L);
  for (auto& p : pos) {
    p = Vector3d{dist(gen), dist(gen), dist(gen)};
  }
  auto vel = std::vector<Vector3d>(N);
  return std::make_tuple(pos, vel);
}

template <typename AccFunc>
struct Simulator {
  using Solver = void(std::span<Vector3d>, std::span<Vector3d>, double, double,
                      double, std::function<AccFunc>, std::size_t);

  std::function<Solver> solver;
  std::function<AccFunc> acc_func;
};

auto const threaded_sim = Simulator<decltype(threaded_gravity)>(
    threaded_euler<std::function<decltype(threaded_gravity)>>,
    threaded_gravity);
auto const threadpool_sim = Simulator<decltype(threadpool_gravity)>{
    threadpool_euler<std::function<decltype(threadpool_gravity)>>,
    threadpool_gravity};

class NBodyTest : public testing::Test {
 protected:
  static inline std::unordered_map<std::size_t, decltype(initialize_state(0))>
      precomputed_states{};
  static constexpr int n_steps = 1000;

  template <typename SimulatorType>
  auto static run_simulation(SimulatorType const& sim, std::size_t N,
                             std::size_t n_threads) {
    auto t0 = 0.0;
    auto tf = characteristic_time(N, L);
    auto dt = (tf - t0) / n_steps;

    if (!precomputed_states.count(N)) {
      precomputed_states[N] = initialize_state(N);
    }
    auto [pos, vel] = precomputed_states[N];

    sim.solver(pos, vel, t0, tf, dt, sim.acc_func, n_threads);

    return std::make_tuple(std::move(pos), std::move(vel));
  }
};

TEST_F(NBodyTest, Threaded1024P8T) {
  auto constexpr N = 1024;
  auto constexpr n_threads = 8;
  auto [pos, vel] = run_simulation(threaded_sim, N, n_threads);

  auto filename = std::stringstream{};
  filename << "test_threaded_n_body_output_" << N << "_particles_" << n_threads
           << "_threads.txt";
  auto output_file = std::ofstream{filename.str()};
  for (auto const& p : pos) {
    output_file << '{' << p.x << ',' << p.y << ',' << p.z << '}';
  }
  output_file << '\n';
}
