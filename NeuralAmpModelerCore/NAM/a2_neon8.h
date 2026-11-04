#pragma once

#include <arm_neon.h>

// Cortex-A17: accumulate the eight output channels of several adjacent frames
// without packing an Eigen matrix for every tap of every 48-frame audio block.
namespace nam::wavenet::a2_fast::neon8
{
template <int Frames>
struct Tile
{
  float32x4_t low[Frames];
  float32x4_t high[Frames];
};

template <int Lane, int Frames>
inline void add_column(Tile<Frames>& sum, const float32x4_t (&input)[Frames], const float* column)
{
  const float32x4_t low = vld1q_f32(column);
  const float32x4_t high = vld1q_f32(column + 4);
  for (int f = 0; f < Frames; ++f)
  {
    const float32x2_t pair = Lane < 2 ? vget_low_f32(input[f]) : vget_high_f32(input[f]);
    sum.low[f] = vmlaq_lane_f32(sum.low[f], low, pair, Lane % 2);
    sum.high[f] = vmlaq_lane_f32(sum.high[f], high, pair, Lane % 2);
  }
}

template <int Frames>
inline void multiply_add(Tile<Frames>& sum, const float* input, const float* weights)
{
  for (int group = 0; group < 8; group += 4)
  {
    float32x4_t values[Frames];
    for (int f = 0; f < Frames; ++f)
      values[f] = vld1q_f32(input + 8 * f + group);
    const float* columns = weights + 8 * group;
    add_column<0>(sum, values, columns);
    add_column<1>(sum, values, columns + 8);
    add_column<2>(sum, values, columns + 16);
    add_column<3>(sum, values, columns + 24);
  }
}

template <int Frames>
inline Tile<Frames> zero()
{
  Tile<Frames> sum;
  for (int f = 0; f < Frames; ++f)
    sum.low[f] = sum.high[f] = vdupq_n_f32(0.0f);
  return sum;
}
} // namespace nam::wavenet::a2_fast::neon8
