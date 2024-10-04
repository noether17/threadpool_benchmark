#include <gtest/gtest.h>

#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "atomic_threadpool_euler.hpp"
#include "parallel_algo_euler.hpp"
#include "physics.hpp"
#include "single_threaded_euler.hpp"
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

auto const single_threaded_sim = Simulator<decltype(single_threaded_gravity)>{
    single_threaded_euler<std::function<decltype(single_threaded_gravity)>>,
    single_threaded_gravity};
auto const parallel_algo_sim =
    Simulator<void()>{parallel_algo_euler<std::function<void()>>, [] {}};
auto const threaded_sim = Simulator<decltype(threaded_gravity)>{
    threaded_euler<std::function<decltype(threaded_gravity)>>,
    threaded_gravity};
auto const threadpool_sim = Simulator<decltype(threadpool_gravity)>{
    threadpool_euler<std::function<decltype(threadpool_gravity)>>,
    threadpool_gravity};

template <typename AccFunc>
void atomic_threadpool_euler_wrapper(std::span<Vector3d> pos,
                                     std::span<Vector3d> vel, double t0,
                                     double tf, double dt, AccFunc acc_func,
                                     int n_threads) {
  auto pos_atomic = std::vector<AtomicVector3d>(pos.size());
  memcpy(reinterpret_cast<void*>(pos_atomic.data()), pos.data(),
         pos.size() * sizeof(Vector3d));
  auto vel_atomic = std::vector<AtomicVector3d>(vel.size());
  memcpy(reinterpret_cast<void*>(vel_atomic.data()), vel.data(),
         vel.size() * sizeof(Vector3d));

  atomic_threadpool_euler(pos_atomic, vel_atomic, t0, tf, dt, acc_func,
                          n_threads);

  memcpy(pos.data(), reinterpret_cast<void*>(pos_atomic.data()),
         pos.size() * sizeof(Vector3d));
  memcpy(vel.data(), reinterpret_cast<void*>(vel_atomic.data()),
         vel.size() * sizeof(Vector3d));
}

auto const atomic_threadpool_sim =
    Simulator<decltype(atomic_threadpool_gravity)>{
        atomic_threadpool_euler_wrapper<
            std::function<decltype(atomic_threadpool_gravity)>>,
        atomic_threadpool_gravity};

class NBodyTest : public testing::Test {
 protected:
  static inline std::unordered_map<std::size_t, decltype(initialize_state(0))>
      precomputed_initial_states{};
  static inline std::unordered_map<std::size_t, decltype(initialize_state(0))>
      precomputed_final_states{};
  static constexpr int n_steps = 100;

  template <typename SimulatorType>
  auto static run_simulation(SimulatorType const& sim, std::size_t N,
                             std::size_t n_threads) {
    auto t0 = 0.0;
    auto tf = characteristic_time(N, L);
    auto dt = (tf - t0) / n_steps;

    if (!precomputed_initial_states.count(N)) {
      precomputed_initial_states[N] = initialize_state(N);
    }
    auto [pos, vel] = precomputed_initial_states[N];
    sim.solver(pos, vel, t0, tf, dt, sim.acc_func, n_threads);

    return std::make_tuple(std::move(pos), std::move(vel));
  }

  auto static compare_states_to_reference(std::span<Vector3d const> pos,
                                          std::span<Vector3d> const vel) {
    auto N = pos.size();
    if (!precomputed_final_states.count(N)) {
      precomputed_final_states[N] = run_simulation(single_threaded_sim, N, 1);
    }
    auto const& [ref_pos, ref_vel] = precomputed_final_states[N];
    for (std::size_t i = 0; i < N; ++i) {
      EXPECT_DOUBLE_EQ(ref_pos[i].x, pos[i].x);
      EXPECT_DOUBLE_EQ(ref_pos[i].y, pos[i].y);
      EXPECT_DOUBLE_EQ(ref_pos[i].z, pos[i].z);
      EXPECT_DOUBLE_EQ(ref_vel[i].x, vel[i].x);
      EXPECT_DOUBLE_EQ(ref_vel[i].y, vel[i].y);
      EXPECT_DOUBLE_EQ(ref_vel[i].z, vel[i].z);
    }
  }
};

// ParallelAlgo tests
TEST_F(NBodyTest, ParallelAlgo2P8T) {
  auto constexpr N = 2;

  auto [pos, vel] = run_simulation(parallel_algo_sim, N, 0);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, ParallelAlgo8P8T) {
  auto constexpr N = 8;

  auto [pos, vel] = run_simulation(parallel_algo_sim, N, 0);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, ParallelAlgo64P8T) {
  auto constexpr N = 64;

  auto [pos, vel] = run_simulation(parallel_algo_sim, N, 0);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, ParallelAlgo1024P8T) {
  auto constexpr N = 1024;

  auto [pos, vel] = run_simulation(parallel_algo_sim, N, 0);

  compare_states_to_reference(pos, vel);
}

// Threaded tests
// TEST_F(NBodyTest, Threaded2P8T) {
//  auto constexpr N = 2;  // N not a multiple of n_threads violates contract.
//  auto constexpr n_threads = 8;
//
//  auto [pos, vel] = run_simulation(threaded_sim, N, n_threads);
//
//  compare_states_to_reference(pos, vel);
// }

TEST_F(NBodyTest, Threaded8P8T) {
  auto constexpr N = 8;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threaded_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threaded64P8T) {
  auto constexpr N = 64;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threaded_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threaded1024P8T) {
  auto constexpr N = 1024;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threaded_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

// Threadpool tests
TEST_F(NBodyTest, Threadpool2P4T) {
  auto constexpr N = 2;
  auto constexpr n_threads = 4;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool2P8T) {
  auto constexpr N = 2;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool2P16T) {
  auto constexpr N = 2;
  auto constexpr n_threads = 16;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool16P4T) {
  auto constexpr N = 16;
  auto constexpr n_threads = 4;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool16P8T) {
  auto constexpr N = 16;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool16P16T) {
  auto constexpr N = 16;
  auto constexpr n_threads = 16;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool128P4T) {
  auto constexpr N = 128;
  auto constexpr n_threads = 4;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool128P8T) {
  auto constexpr N = 128;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool128P16T) {
  auto constexpr N = 128;
  auto constexpr n_threads = 16;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool1024P4T) {
  auto constexpr N = 1024;
  auto constexpr n_threads = 4;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool1024P8T) {
  auto constexpr N = 1024;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, Threadpool1024P16T) {
  auto constexpr N = 1024;
  auto constexpr n_threads = 16;

  auto [pos, vel] = run_simulation(threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

// Atomic Threadpool tests
TEST_F(NBodyTest, AtomicThreadpool2P8T) {
  auto constexpr N = 2;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(atomic_threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, AtomicThreadpool8P8T) {
  auto constexpr N = 8;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(atomic_threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, AtomicThreadpool64P8T) {
  auto constexpr N = 64;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(atomic_threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}

TEST_F(NBodyTest, AtomicThreadpool1024P8T) {
  auto constexpr N = 1024;
  auto constexpr n_threads = 8;

  auto [pos, vel] = run_simulation(atomic_threadpool_sim, N, n_threads);

  compare_states_to_reference(pos, vel);
}
