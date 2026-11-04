#pragma once

#include <string_view>

namespace dessamp::player
{
enum class HeadShellIdentity
{
  Generic,
  Forge,
  Fire,
  Redline,
};

constexpr HeadShellIdentity ResolveHeadShellIdentity(const std::string_view requestedStyle,
                                                      const bool morphEnabled,
                                                      const std::string_view artifactCode) noexcept
{
  // A package's explicit Redline presentation is authoritative even when its
  // model contains a learned Morph control. Other morph-enabled packages keep
  // the Forge identity used by the existing conditioned-morph product.
  if (requestedStyle == "dual-red-modern-v1")
    return HeadShellIdentity::Redline;
  if (morphEnabled || requestedStyle == "forge-streamlined-v1")
    return HeadShellIdentity::Forge;
  if (requestedStyle == "fire-chrome-v1" || artifactCode == "DESSFIRE")
    return HeadShellIdentity::Fire;
  return HeadShellIdentity::Generic;
}
} // namespace dessamp::player
