#include "../GuitarTuner.h"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using Tuner = dessamp::GuitarTuner;
static Tuner::Reading pluck(Tuner& tuner, int midi, double cents, double rate)
{
  const double hz = 440 * std::exp2((midi - 69 + cents / 100) / 12);
  std::vector<double> right(int(rate*0.135)), left(right.size(), 0);
  for (size_t i=0; i<right.size(); ++i)
  {
    const double t=i/rate, p=6.283185307179586*hz*t;
    right[i]=0.015+std::exp(-2*t)*(0.08*std::sin(p)+0.14*std::sin(2*p)+0.04*std::sin(3*p));
  }
  for (size_t i=0; i<left.size(); i+=127)
  {
    double* inputs[]{left.data()+i,right.data()+i};
    tuner.Push(inputs,2,int(std::min<size_t>(127,left.size()-i)),rate);
  }
  Tuner::Reading r;
  assert(tuner.Read(r));
  return r;
}

int main()
{
  for (const double rate : {44100.,48000.,96000.,192000.})
    for (int note : {21,23,26,28,35,38,40,45,50,55,59,64,76,88})
      for (double cents : {-23.,0.,19.})
      {
        Tuner t; t.Enable(true);
        const auto r=pluck(t,note,cents,rate);
        if (r.note!=note || std::abs(r.cents-cents)>3.0)
        { std::cerr<<rate<<" note "<<note<<" at "<<cents<<" cents gave "<<r.note<<" at "<<r.cents<<"\n"; return 1; }
      }
  Tuner::Frame frame; frame.rate=8000.;
  assert(Tuner::Detect(frame).note==-1);
  frame.signal.fill(0.5f); assert(Tuner::Detect(frame).note==-1);
  std::mt19937 gen(104);std::uniform_real_distribution<float> random(-.2f,.2f);
  for(auto& s:frame.signal)s=random(gen);
  assert(Tuner::Detect(frame).note==-1);
  Tuner t; std::array<float,4800> data{}; float* inputs[]{data.data()};
  Tuner::Reading r;
  t.Push(inputs,1,int(data.size()),48000.); assert(!t.Read(r));
  t.Enable(true); t.Push(inputs,1,int(data.size()),48000.);
  t.Enable(false); t.Enable(true); assert(!t.Read(r)); // discard the previous panel session
  assert(pluck(t,40,12.,48000.).note==40);
  std::cout<<"168 detuned and centered string checks passed, plus silence, noise, right input and reopen\n";
}
