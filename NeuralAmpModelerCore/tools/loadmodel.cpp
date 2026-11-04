#include <cstdlib>
#include <exception>
#include <filesystem>
#include <cstdio>
#include <stdexcept>
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::fprintf(stderr, "Usage: loadmodel <model_path>\n");
    return EXIT_FAILURE;
  }

  const char* modelPath = argv[1];
  std::fprintf(stderr, "Loading model [%s]\n", modelPath);

  try
  {
    auto model = nam::get_dsp(std::filesystem::path(modelPath));
    if (model == nullptr)
      throw std::runtime_error("model factory returned no model");
    std::fprintf(stderr, "Model loaded successfully\n");
  }
  catch (const std::exception& error)
  {
    std::fprintf(stderr, "Failed to load model: %s\n", error.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
