#define main DessAmpIntegrationTestMain
#include "test_au_player.mm"
#undef main
#include <fstream>
#include "factory_test_profile.h"

int main(int argc, char** argv)
{
  try
  {
    if (argc != 4)
      throw std::runtime_error("usage: test_au_factory_defaults exact-binary profile.json original-IR.wav");
    std::ifstream profile(argv[2]);
    const auto document = nlohmann::json::parse(profile);
    const auto& entry = document.at("amplifiers").at(0);
    const auto expected = FactoryTestParameters<kParameterCount>(entry);
    for (const auto& amp : document.at("amplifiers"))
      if (amp.at("parameters").at(2) != -85 || amp.at("parameters").at(9) != 1)
        throw std::runtime_error("Factory gate must start on at -85 dB for every amp");
    if (expected[11] != 0) throw std::runtime_error("FIRE factory drive must start off");
    const auto resources = std::filesystem::canonical(argv[1]).parent_path().parent_path() / "Resources";
    const auto model = (resources / "PrivateValidation/Amplifiers/amp-01.dess").string();
    const auto cabinet = (resources / "IRs" / entry.value("cabinet", std::string("Desfire.wav"))).string();
    void* handle = nullptr;
    const auto component = RegisterComponentBinary(argv[1], handle);
    auto unit = CreateUnit(component);
    VerifyHostParameters(unit, expected);
    const auto audio = RenderRealtime(unit, 48000);
    VerifyHostParameters(unit, expected);
    const auto saved = CopyState(unit);
    VerifySerializedState(saved, model, cabinet, expected);
    if (RootMeanSquare(audio) <= 1e-8)
      throw std::runtime_error("Factory default is silent");
    AudioComponentInstanceDispose(unit);
    const auto recalled = RenderState(component, model, cabinet, expected);
    const auto original = RenderState(component, model, argv[3], expected);
    const auto recallDifference = DifferenceRMS(audio, recalled);
    const auto cabinetDifference = DifferenceRMS(recalled, original);
    std::cout << "Factory startup vs recall RMS difference: " << recallDifference << '\n'
              << "Packaged cabinet vs source IR RMS difference: " << cabinetDifference << '\n';
    if (recallDifference > 1e-9 || cabinetDifference > 1e-9)
      throw std::runtime_error("Factory preset changed the saved tone");
    // A restored session must override factory voicing, including a dry cabinet.
    auto changed = expected;
    changed[0] = -3.0; changed[2] = -61.0; changed[13] = 0.0; changed[21] = 12.0;
    unit = CreateUnit(component);
    const auto state = MakeStateDictionary(model, "", changed);
    ApplyState(unit, state);
    RenderRealtime(unit, 48000);
    VerifyHostParameters(unit, changed);
    CFRelease(state); CFRelease(saved);
    AudioComponentInstanceDispose(unit);
    std::cout << "Factory startup, complete rig state, cabinet identity, and session precedence PASS\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
