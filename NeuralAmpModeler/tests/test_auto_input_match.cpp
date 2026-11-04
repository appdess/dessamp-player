#include "../AutoInputMatch.h"
#include <cassert>
#include <iostream>
#include <limits>

int main()
{
  using Matcher = dessamp::AutoInputMatch;
  Matcher match;
  double left[32], right[32];
  double* input[] = {left, right};
  auto fill = [&](double l, double r) {
    std::fill_n(left, 32, l); std::fill_n(right, 32, r);
  };
  auto run = [&]() {
    for (int i = 0; i < 16; ++i) match.Process(input, 32, 2, 100.0, true);
    assert(match.GetState() == Matcher::State::Complete);
  };
  assert(match.Start());
  assert(!match.Start());
  fill(0.1, 0.1); run();
  assert(std::abs(match.Peak() - 0.1) < 1e-12);
  assert(match.Progress() == 100 && !match.Clipped());
  assert(std::abs(Matcher::TrimDB(match.Peak(), 0) - 8.0) < 1e-12);
  assert(std::abs(Matcher::TrimDB(match.Peak(), 16.54) + 8.54) < 1e-12);
  fill(0.8, 0.8);
  match.Process(input, 32, 2, 100, true);
  assert(match.Peak() == 0.1); // finished; never follows subsequent dynamics
  match.Start(); fill(1.0, -1.0); run();
  assert(match.Clipped()); // detect channel clipping even when mono cancels
  match.Start(); fill(0, 0); run(); assert(match.Peak() == 0);
  match.Start(); fill(std::numeric_limits<double>::quiet_NaN(), 0); run();
  assert(match.Clipped());
  match.Start(); fill(0.1, 0); match.Process(input, 32, 2, 100, true);
  match.Process(input, 32, 2, 200, true);
  assert(match.GetState() == Matcher::State::Cancelled);
  match.Start(); match.Cancel(); match.Process(input, 32, 2, 100, true);
  assert(match.GetState() == Matcher::State::Cancelled);
  match.Start(); fill(0.2, 0.2);
  for (int i = 0; i < 16; ++i) match.Process(input, 32, 2, 100, false);
  assert(std::abs(match.Peak() - 0.4) < 1e-12); // standalone mono sum
  std::cout << "Auto Input: finite window, stereo clipping, silence, nonfinite, cancel, reset, fixed trim passed\n";
}
