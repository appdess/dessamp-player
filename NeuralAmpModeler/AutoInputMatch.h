#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace dessamp {
// A finite measurement, not an AGC. Only Process touches accumulation; UI
// requests and results cross the audio boundary through atomic publications.
class AutoInputMatch
{
public:
  enum class State { Idle, Requested, Measuring, Complete, Cancelled };
  bool Start()
  {
    auto state = mState.load(std::memory_order_acquire);
    if (state == State::Requested || state == State::Measuring) return false;
    mProgress.store(0, std::memory_order_relaxed);
    mState.store(State::Requested, std::memory_order_release);
    return true;
  }
  void Cancel() { mState.store(State::Cancelled, std::memory_order_release); }
  State GetState() const { return mState.load(std::memory_order_acquire); }
  int Progress() const { return mProgress.load(std::memory_order_relaxed); }
  double Peak() const { return mResultPeak.load(std::memory_order_relaxed); }
  bool Clipped() const { return mResultClipped.load(std::memory_order_relaxed); }

  template <typename Sample>
  void Process(Sample** input, int frames, int channels, double rate, bool averageChannels)
  {
    auto state = mState.load(std::memory_order_acquire);
    if (state == State::Requested)
    {
      mCount = 0;
      mPeak = 0;
      mClipped = false;
      mRate = rate;
      mTotal = static_cast<std::uint64_t>(std::max(1.0, rate * 5.0));
      if (!mState.compare_exchange_strong(state, State::Measuring, std::memory_order_acq_rel)) return;
    }
    else if (state != State::Measuring) return;
    if (rate != mRate) { Cancel(); return; }
    for (int s = 0; s < frames && mCount < mTotal; ++s, ++mCount)
    {
      double mono = 0;
      for (int c = 0; input != nullptr && c < channels; ++c)
      {
        const double value = input[c][s];
        if (!std::isfinite(value) || std::abs(value) >= 0.999) mClipped = true;
        if (std::isfinite(value)) mono += value;
      }
      if (averageChannels && channels > 0) mono /= channels;
      mPeak = std::max(mPeak, std::abs(mono));
    }
    mProgress.store(static_cast<int>(100 * mCount / mTotal), std::memory_order_relaxed);
    if (mCount == mTotal)
    {
      mResultPeak.store(mPeak, std::memory_order_relaxed);
      mResultClipped.store(mClipped, std::memory_order_relaxed);
      auto measuring = State::Measuring;
      mState.compare_exchange_strong(measuring, State::Complete, std::memory_order_release);
    }
  }

  static double TrimDB(double peak, double calibrationDB)
  {
    return -12.0 - 20.0 * std::log10(std::max(peak, 1e-12)) - calibrationDB;
  }

private:
  std::atomic<State> mState{State::Idle};
  std::atomic<int> mProgress{0};
  std::atomic<double> mResultPeak{0};
  std::atomic<bool> mResultClipped{false};
  std::uint64_t mCount = 0, mTotal = 1;
  double mPeak = 0, mRate = 0;
  bool mClipped = false;
};
}
