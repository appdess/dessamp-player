#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace dessamp
{
// One producer (audio callback), one consumer (editor idle). The callback
// only averages incoming samples and occasionally publishes a fixed frame.
// Note detection and display formatting never run on the audio thread.
class GuitarTuner
{
public:
  struct Reading { int note = -1; double hz = 0.0, cents = 0.0; };
  static constexpr size_t kSize = 1536;
  struct Frame { std::array<float, kSize> signal{}; double rate = 0; unsigned session = 0; };

  void Enable(bool enable)
  {
    const auto id = mSession.load();
    if (bool(id & 1U) != enable) mSession.store(id + 1U);
  }

  template<class Sample>
  void Push(Sample* const* inputs, int channels, int count, double rate)
  {
    const auto session = mSession.load(std::memory_order_acquire);
    if (!(session & 1U) || !inputs || channels <= 0 || count <= 0 || !std::isfinite(rate) || rate < 8000 || rate > 768000) return;
    if (session != mWrite.session || rate != mSourceRate)
    {
      mWrite.session = session;
      mSourceRate = rate;
      mStride = std::max(1, int(std::lround(rate / 16000.0)));
      mWrite.rate = rate / mStride;
      mIndex = mCount = 0;
      mSum = 0;
    }
    for (int i = 0; i < count; ++i)
    {
      double mono = 0;
      for (int c = 0; c < channels; ++c)
        if (inputs[c] && std::isfinite(inputs[c][i])) mono += inputs[c][i];
      mSum += std::clamp(mono, -4.0, 4.0);
      if (++mCount < mStride) continue;
      mWrite.signal[mIndex++] = float(mSum / mCount);
      mCount = 0;
      mSum = 0;
      if (mIndex != kSize) continue;
      if (!mReady.load(std::memory_order_acquire))
      {
        mShared = mWrite;
        mReady.store(true, std::memory_order_release);
      }
      mIndex = 0;
    }
  }

  bool Read(Reading& result)
  {
    if (!mReady.load(std::memory_order_acquire)) return false;
    const Frame frame = mShared;
    mReady.store(false, std::memory_order_release);
    if (frame.session != mSession.load(std::memory_order_acquire)) return false;
    result = Detect(frame);
    return true;
  }

  // First sufficiently periodic minimum of the normalized difference (YIN),
  // refined between samples. A4 = 440 Hz; A0 through E6 includes drop tunings.
  static Reading Detect(const Frame& f)
  {
    if (f.rate < 6000 || f.rate > 24000) return {};
    double sum = 0, squares = 0;
    for (float x : f.signal) { sum += x; squares += double(x) * x; }
    if (squares / kSize - (sum/kSize)*(sum/kSize) < 1.0e-6) return {};
    const int hi = std::min(int(kSize / 2) - 1, int(std::ceil(f.rate / 27.0)) + 1);
    const int lo = std::max(2, int(std::floor(f.rate / 1400.0)));
    const int span = int(kSize) - hi;
    std::array<double, kSize / 2> diff{};
    double total = 0;
    diff[0] = 1;
    for (int lag = 1; lag <= hi; ++lag)
    {
      double dist = 0;
      for (int i = 0; i < span; ++i)
      {
        const double d = f.signal[i] - f.signal[i + lag];
        dist += d * d;
      }
      total += dist;
      diff[lag] = total > 0 ? lag * dist / total : 1;
    }
    for (int lag = lo; lag < hi; ++lag)
    {
      if (diff[lag] >= (lag < 12 ? 0.55 : 0.14) || diff[lag] > diff[lag-1] || diff[lag] >= diff[lag+1]) continue;
      // Refine the *waveform* alignment, not the normalized valley's shape:
      // strong upper harmonics otherwise shift a high string several cents.
      auto error = [&](double period) {
        const int k = int(period);
        const double u = period-k;
        double energy = 0;
        for (int i=0; i<span-3; ++i)
        {
          const auto* p = &f.signal[i+k];
          const double y = p[0] + 0.5*u*((p[1]-p[-1]) +
            u*((2*p[-1]-5*p[0]+4*p[1]-p[2])+u*(3*(p[0]-p[1])+p[2]-p[-1])));
          const double d = f.signal[i] - y;
          energy += d*d;
        }
        return energy;
      };
      double left = lag-1.0, right = lag+1.0;
      for (int i=0; i<13; ++i)
      {
        const double a = (2*left+right)/3, b = (left+2*right)/3;
        if (error(a) < error(b)) right=b; else left=a;
      }
      const double period = (left+right)/2;
      const double variance = squares/kSize - (sum/kSize)*(sum/kSize);
      // For a short period the integer lags are coarse. Accept a candidate
      // only if the interpolated waveform actually repeats.
      if (error(period) > 0.20 * variance * (span-3)) continue;
      const double hz = f.rate / period;
      if (hz < 27.0 || hz > 1400.0) return {};
      const double midi = 69 + 12 * std::log2(hz / 440.0);
      const int note = int(std::lround(midi));
      return {note, hz, (midi - note) * 100.0};
    }
    return {};
  }

private:
  static_assert(std::atomic<bool>::is_always_lock_free && std::atomic<unsigned>::is_always_lock_free);
  std::atomic<unsigned> mSession{0};
  std::atomic<bool> mReady{false};
  Frame mWrite, mShared;
  size_t mIndex = 0;
  int mStride = 1, mCount = 0;
  double mSum = 0, mSourceRate = 0;
};
} // namespace dessamp
