#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define AUDIO_BUFFER_SIZE 64

double inputBuffer[AUDIO_BUFFER_SIZE];
double outputBuffer[AUDIO_BUFFER_SIZE];

int main(int argc, char* argv[])
{
  if (argc > 1)
  {
    const char* modelPath = argv[1];
    std::vector<double> parameters;
    bool qualityProvided = false;
    double quality = 1.0;
    try
    {
      for (int index = 2; index < argc; ++index)
      {
        const std::string argument(argv[index]);
        if (argument == "--param")
        {
          if (++index >= argc)
            throw std::invalid_argument("--param requires a value");
          parameters.push_back(std::stod(argv[index]));
          if (!std::isfinite(parameters.back()))
            throw std::invalid_argument("--param must be finite");
        }
        else if (argument == "--quality")
        {
          if (++index >= argc)
            throw std::invalid_argument("--quality requires a value");
          quality = std::stod(argv[index]);
          if (!std::isfinite(quality) || quality < 0.0 || quality > 1.0)
            throw std::invalid_argument("--quality must be in [0, 1]");
          qualityProvided = true;
        }
        else
          throw std::invalid_argument("Unknown argument: " + argument);
      }
    }
    catch (const std::exception& error)
    {
      std::cerr << error.what() << "\n";
      std::cerr << "Usage: benchmodel <model_path> [--param VALUE ...] [--quality 0..1]\n";
      return 1;
    }

    std::cout << "Loading model " << modelPath << "\n";

    // Turn on fast tanh approximation
    nam::activations::Activation::enable_fast_tanh();

    std::unique_ptr<nam::DSP> model;

    model.reset();
    model = nam::get_dsp(std::filesystem::path(modelPath));

    if (model == nullptr)
    {
      std::cerr << "Failed to load model\n";

      exit(1);
    }

    if (parameters.empty() && model->NumGlobalParams() > 0)
      parameters.assign(static_cast<std::size_t>(model->NumGlobalParams()), 0.5);
    if (model->NumGlobalParams() != static_cast<int>(parameters.size()))
    {
      std::cerr << "Model expects " << model->NumGlobalParams() << " parameters; received "
                << parameters.size() << "\n";
      return 1;
    }
    model->SetDefaultParams(parameters.empty() ? nullptr : parameters.data(),
                            static_cast<int>(parameters.size()));
    auto* slimmable = dynamic_cast<nam::SlimmableModel*>(model.get());
    if (qualityProvided && slimmable == nullptr)
    {
      std::cerr << "--quality is only valid for a slimmable model\n";
      return 1;
    }
    if (slimmable != nullptr)
      slimmable->SetSlimmableSize(quality);

    size_t bufferSize = AUDIO_BUFFER_SIZE;
    model->Reset(model->GetExpectedSampleRate(), bufferSize);
    size_t numBuffers = (48000 / bufferSize) * 2;

    // Allocate multi-channel buffers
    const int in_channels = model->NumInputChannels();
    const int out_channels = model->NumOutputChannels();

    std::vector<std::vector<double>> inputBuffers(in_channels);
    std::vector<std::vector<double>> outputBuffers(out_channels);
    std::vector<double*> inputPtrs(in_channels);
    std::vector<double*> outputPtrs(out_channels);

    for (int ch = 0; ch < in_channels; ch++)
    {
      inputBuffers[ch].resize(AUDIO_BUFFER_SIZE, 0.0);
      inputPtrs[ch] = inputBuffers[ch].data();
    }
    for (int ch = 0; ch < out_channels; ch++)
    {
      outputBuffers[ch].resize(AUDIO_BUFFER_SIZE, 0.0);
      outputPtrs[ch] = outputBuffers[ch].data();
    }

    std::cout << "Running benchmark\n";
    auto t1 = high_resolution_clock::now();
    for (size_t i = 0; i < numBuffers; i++)
    {
      if (parameters.empty())
        model->process(inputPtrs.data(), outputPtrs.data(), AUDIO_BUFFER_SIZE);
      else
        model->process(inputPtrs.data(), outputPtrs.data(), AUDIO_BUFFER_SIZE,
                       parameters.data(), static_cast<int>(parameters.size()));
    }
    auto t2 = high_resolution_clock::now();
    std::cout << "Finished\n";


    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "quality=" << quality << " global_params=" << parameters.size()
              << " estimated_ops_per_sample=" << model->EstimatedOperationsPerSample()
              << " realtime_factor=" << (2000.0 / ms_double.count()) << "x\n";
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
  }
  else
  {
    std::cerr << "Usage: benchmodel <model_path> [--param VALUE ...] [--quality 0..1]\n";
  }

  exit(0);
}
