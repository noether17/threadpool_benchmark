#include <benchmark/benchmark.h>

#include <chrono>
#include <random>
#include <tuple>
#include <vector>

#include "physics.hpp"
#include "single_threaded_euler.hpp"
#include "threaded_euler.hpp"
#include "threadpool_euler.hpp"

auto initialize_state(std::size_t N) {
  auto pos = std::vector<Vector3d>(N);
  auto gen = std::mt19937{0};
  auto dist = std::uniform_real_distribution<double>(0.0, L);
  for (auto& p : pos) {
    p = Vector3d{dist(gen), dist(gen), dist(gen)};
  }
  auto vel = std::vector<Vector3d>(N);
  return std::make_tuple(pos, vel);
}

void static BM_NBodySingleThreaded(benchmark::State& state) {
  auto N = state.range(0);
  // auto n_threads = state.range(1); // n_threads not used for single threaded.
  auto n_steps = state.range(2);

  auto t0 = 0.0;
  auto tf = characteristic_time(N, L);
  auto dt = (tf - t0) / n_steps;
  auto [pos, vel] = initialize_state(N);

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    single_threaded_euler(pos, vel, t0, tf, dt, single_threaded_gravity, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    benchmark::DoNotOptimize(pos.data());
    benchmark::DoNotOptimize(vel.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(n_steps * state.iterations());
}

void static BM_NBodyThreaded(benchmark::State& state) {
  auto N = state.range(0);
  auto n_threads = state.range(1);
  auto n_steps = state.range(2);

  auto t0 = 0.0;
  auto tf = characteristic_time(N, L);
  auto dt = (tf - t0) / n_steps;
  auto [pos, vel] = initialize_state(N);

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    threaded_euler(pos, vel, t0, tf, dt, threaded_gravity, n_threads);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    benchmark::DoNotOptimize(pos.data());
    benchmark::DoNotOptimize(vel.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(n_steps * state.iterations());
}

void static BM_NBodyThreadpool(benchmark::State& state) {
  auto N = state.range(0);
  auto n_threads = state.range(1);
  auto n_steps = state.range(2);

  auto t0 = 0.0;
  auto tf = characteristic_time(N, L);
  auto dt = (tf - t0) / n_steps;
  auto [pos, vel] = initialize_state(N);

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    threadpool_euler(pos, vel, t0, tf, dt, threadpool_gravity, n_threads);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
    benchmark::DoNotOptimize(pos.data());
    benchmark::DoNotOptimize(vel.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(n_steps * state.iterations());
}

// 16 Particles
// Single-Threaded
BENCHMARK(BM_NBodySingleThreaded)
    ->Args({16, 1, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 4 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({16, 4, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({16, 4, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 8 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({16, 8, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({16, 8, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 16 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({16, 16, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({16, 16, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 128 Particles
// Single-Threaded
BENCHMARK(BM_NBodySingleThreaded)
    ->Args({128, 1, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 4 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({128, 4, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({128, 4, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 8 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({128, 8, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({128, 8, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 16 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({128, 16, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({128, 16, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 1024 Particles
// Single-Threaded
BENCHMARK(BM_NBodySingleThreaded)
    ->Args({1024, 1, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 4 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({1024, 4, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({1024, 4, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 8 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({1024, 8, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({1024, 8, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 16 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({1024, 16, 10'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({1024, 16, 10'000})
    ->Iterations(1)
    ->UseManualTime();

// 8192 Particles
// (Reduce n_steps so that single threaded can finish in under an hour.)
// Single-Threaded
BENCHMARK(BM_NBodySingleThreaded)
    ->Args({8192, 1, 1'000})
    ->Iterations(1)
    ->UseManualTime();

// 4 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({8192, 4, 1'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({8192, 4, 1'000})
    ->Iterations(1)
    ->UseManualTime();

// 8 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({8192, 8, 1'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({8192, 8, 1'000})
    ->Iterations(1)
    ->UseManualTime();

// 16 Threads
BENCHMARK(BM_NBodyThreaded)
    ->Args({8192, 16, 1'000})
    ->Iterations(1)
    ->UseManualTime();
BENCHMARK(BM_NBodyThreadpool)
    ->Args({8192, 16, 1'000})
    ->Iterations(1)
    ->UseManualTime();
