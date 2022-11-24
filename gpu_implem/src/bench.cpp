#include "render.hpp"
#include <vector>
#include <benchmark/benchmark.h>

constexpr int kRGBASize = 4;
constexpr int width = 4800;
constexpr int height = 3200;
constexpr int niteration = 1000;

void BM_Rendering_cpu(benchmark::State& st)
{
  int stride = width * kRGBASize;
  std::vector<char> data(height * stride);

  for (auto _ : st)
    render_cpu(data.data(), width, height, stride, niteration);

  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu(benchmark::State& st)
{
  int stride = width * kRGBASize;
  std::vector<char> data(height * stride);

  for (auto _ : st)
    render(data.data(), width, height, stride, niteration);

  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
