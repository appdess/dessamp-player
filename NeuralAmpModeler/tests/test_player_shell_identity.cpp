#include "../PlayerShellIdentity.h"

#include <cassert>
#include <iostream>

using dessamp::player::HeadShellIdentity;
using dessamp::player::ResolveHeadShellIdentity;

int main()
{
  // The trained DESS REDLINE package keeps its explicit shell even though its
  // model exposes a learned Morph control.
  assert(ResolveHeadShellIdentity("dual-red-modern-v1", true, "REDLINE01")
         == HeadShellIdentity::Redline);

  // Genuine Forge packages and the existing generic morph fallback retain
  // their prior Forge behavior.
  assert(ResolveHeadShellIdentity("forge-streamlined-v1", true, "FORGE001")
         == HeadShellIdentity::Forge);
  assert(ResolveHeadShellIdentity("studio-black-v1", true, "MORPH001")
         == HeadShellIdentity::Forge);
  assert(ResolveHeadShellIdentity("forge-streamlined-v1", false, "FORGE001")
         == HeadShellIdentity::Forge);

  assert(ResolveHeadShellIdentity("fire-chrome-v1", false, "FIRE0001")
         == HeadShellIdentity::Fire);
  assert(ResolveHeadShellIdentity("studio-black-v1", false, "DESSFIRE")
         == HeadShellIdentity::Fire);
  assert(ResolveHeadShellIdentity("dual-red-modern-v1", false, "DESSFIRE")
         == HeadShellIdentity::Redline);
  assert(ResolveHeadShellIdentity("studio-black-v1", false, "OTHER001")
         == HeadShellIdentity::Generic);

  std::cout << "DESS AMP player shell identity tests passed\n";
  return 0;
}
