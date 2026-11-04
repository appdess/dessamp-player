#pragma once

#include <optional>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath> // std::round
#include <filesystem>
#include <functional>
#include <iomanip>
#include <sstream> // std::stringstream
#include <string>
#include <array>
#include <utility>
#include <vector>
#include "IControls.h"
#include "DessAmpResponsiveLayout.h"
#include "DessPresetFile.h"

using namespace iplug;
using namespace igraphics;

namespace DessAmpUI
{
inline const IColor Window{255, 9, 11, 13};
inline const IColor Surface{255, 25, 28, 32};
inline const IColor SurfaceRaised{255, 31, 34, 38};
inline const IColor Hairline{255, 52, 56, 61};
inline const IColor Accent{255, 232, 48, 54};
inline const IColor AccentHot{255, 255, 44, 50};
inline const IColor MorphBrit{255, 255, 166, 52};
inline const IColor TextPrimary{255, 242, 241, 236};
inline const IColor TextSecondary{255, 169, 173, 178};
inline const IColor Disabled{255, 100, 105, 112};
inline const IColor Metal{255, 198, 200, 199};
inline const IColor Destructive{255, 211, 74, 74};

inline IText Text(float size, const IColor& color = TextPrimary, EAlign align = EAlign::Center,
                  const char* font = nullptr)
{
  const bool operationalLabel = std::abs(size - 12.f) < 0.01f || std::abs(size - 18.f) < 0.01f;
  return IText(size, color, font == nullptr ? (operationalLabel ? "Roboto-Semibold" : "Roboto-Regular") : font, align,
               EVAlign::Middle);
}

inline void DrawGlass(IGraphics& g, const IRECT& r, const float radius, const bool strong = false)
{
  g.FillRoundRect(COLOR_BLACK.WithOpacity(strong ? 0.38f : 0.25f), r.GetTranslated(0.f, 5.f), radius);
  g.PathRoundRect(r, radius);
  g.PathFill(IPattern::CreateLinearGradient(
    r, EDirection::Vertical,
    {{IColor(strong ? 226 : 198, 38, 42, 48), 0.f}, {IColor(strong ? 214 : 180, 15, 18, 22), 1.f}}));
  g.DrawRoundRect(COLOR_WHITE.WithOpacity(0.20f), r, radius, nullptr, 1.f);
  g.DrawLine(COLOR_WHITE.WithOpacity(0.17f), r.L + radius, r.T + 1.f, r.R - radius, r.T + 1.f, nullptr, 1.f);
}
} // namespace DessAmpUI

// The complete product frame is vector-rendered. Generated imagery is limited
// to the physical amp shell and knob face so layout, state and accessibility
// never depend on baked text.
class DessAmpBackdropControl : public IControl
{
public:
  explicit DessAmpBackdropControl(const IRECT& bounds)
  : IControl(bounds)
  {
    mIgnoreMouse = true;
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    g.FillRect(Window, mRECT);
    g.PathRect(mRECT);
    g.PathFill(IPattern::CreateLinearGradient(
      mRECT, EDirection::Vertical, {{IColor(255, 29, 32, 37), 0.f}, {IColor(255, 7, 9, 12), 1.f}}));
  }
};

class DessAmpGlassChromeControl : public IControl
{
public:
  explicit DessAmpGlassChromeControl(const IRECT& bounds)
  : IControl(bounds)
  {
    mIgnoreMouse = true;
  }

  void SetPage(DessAmpPage page) { mPage = page; SetDirty(false); }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const ResponsiveLayout layout(mRECT);
    if (!layout.phone)
    {
      DrawGlass(g, layout.header, 24.f, true);
      DrawGlass(g, layout.utility, 24.f, true);
      return;
    }
    g.DrawLine(Hairline, layout.header.L, layout.header.B, layout.header.R, layout.header.B);
    if (mPage == kRigPage)
    {
      g.FillRoundRect(Surface, layout.utility, 9.f);
      g.DrawText(Text(12.f, TextSecondary, EAlign::Near), "RIG LEVELS  /  SHARED BY ALL AMPS", layout.levels.GetFromTop(18.f));
      g.DrawLine(Hairline, layout.levels.L, layout.levels.B, layout.levels.R, layout.levels.B);
    }
    if (layout.portraitPhone && mPage == kAmpPage)
    {
      g.FillRoundRect(Surface, layout.controls, 9.f);
      g.DrawText(Text(12.f, TextSecondary, EAlign::Near), "AMP CONTROLS",
                 layout.controls.GetHPadded(-14.f).GetFromTop(27.f));
    }
  }
private:
  DessAmpPage mPage = kAmpPage;
};

// Header treatment shared by standalone and plug-in editors. The grille keeps
// its own smaller DESS badge. This mark is decorative and never captures input.
class DessAmpWordmark : public IControl, public IBitmapBase
{
public:
  DessAmpWordmark(const IRECT& bounds, const IBitmap& mark) : IControl(bounds), IBitmapBase(mark)
  { mIgnoreMouse = true; }
  void OnRescale() override { mBitmap = GetUI()->GetScaledBitmap(mBitmap); }
  void Draw(IGraphics& g) override
  {
    if (!mBitmap.IsValid()) return;
    const float width = std::min({mRECT.W(), mRECT.H() * 0.60f * 480.f / 62.f, 208.f});
    const float height = width * 62.f / 480.f;
    g.DrawFittedBitmap(mBitmap, mRECT.GetFromLeft(width).GetCentredInside(width, height));
  }
};

class DessAmpHeadControl : public IControl, public IBitmapBase
{
public:
  enum class ShellStyle
  {
    Studio,
    DualRed,
    ForgeStreamlined,
    FireChrome,
  };

  DessAmpHeadControl(const IRECT& bounds, const IBitmap& studioShell, const IBitmap& dualRedShell,
                     const IBitmap& forgeShell, const IBitmap& fireShell, const IBitmap& badge)
  : IControl(bounds)
  , IBitmapBase(studioShell)
  , mStudioShell(studioShell)
  , mDualRedShell(dualRedShell)
  , mForgeShell(forgeShell)
  , mFireShell(fireShell)
  , mBadge(badge)
  {
    mIgnoreMouse = true;
  }

  void SetShellStyle(const char* style)
  {
    ShellStyle nextStyle = ShellStyle::Studio;
    if (style != nullptr)
    {
      const std::string token(style);
      if (token == "dual-red-modern-v1")
        nextStyle = ShellStyle::DualRed;
      else if (token == "forge-streamlined-v1")
        nextStyle = ShellStyle::ForgeStreamlined;
      else if (token == "fire-chrome-v1")
        nextStyle = ShellStyle::FireChrome;
    }
    if (mShellStyle == nextStyle)
      return;
    mShellStyle = nextStyle;
    ApplyShell();
    SetDirty(false);
  }

  void SetMorphPresentation(
    const bool enabled, const double value, const char* fromLabel = "GER", const char* toLabel = "BRIT")
  {
    const double nextValue = std::clamp(value, 0.0, 1.0);
    const std::string nextFrom = fromLabel == nullptr ? "GER" : fromLabel;
    const std::string nextTo = toLabel == nullptr ? "BRIT" : toLabel;
    if (mMorphEnabled == enabled && std::abs(mMorphValue - nextValue) < 1.0e-9 && mMorphFrom == nextFrom
        && mMorphTo == nextTo)
      return;
    mMorphEnabled = enabled;
    mMorphValue = nextValue;
    mMorphFrom = nextFrom;
    mMorphTo = nextTo;
    SetDirty(false);
  }

  void SetMorphValue(const double value)
  {
    const double nextValue = std::clamp(value, 0.0, 1.0);
    if (!mMorphEnabled || std::abs(mMorphValue - nextValue) < 1.0e-9)
      return;
    mMorphValue = nextValue;
    SetDirty(false);
  }

  void SetSubtitle(const char* subtitle)
  {
    const std::string next = subtitle == nullptr ? "" : subtitle;
    if (mSubtitle == next)
      return;
    mSubtitle = next;
    SetDirty(false);
  }

  void OnRescale() override
  {
    mStudioShell = GetUI()->GetScaledBitmap(mStudioShell);
    mDualRedShell = GetUI()->GetScaledBitmap(mDualRedShell);
    mForgeShell = GetUI()->GetScaledBitmap(mForgeShell);
    mFireShell = GetUI()->GetScaledBitmap(mFireShell);
    mBadge = GetUI()->GetScaledBitmap(mBadge);
    ApplyShell();
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const IRECT& art = mRECT;
    g.FillRoundRect(COLOR_BLACK.WithOpacity(0.48f), art.GetTranslated(0.f, 6.f), 18.f);
    g.DrawFittedBitmap(mBitmap, art);
    const float morph = static_cast<float>(std::clamp(mMorphValue, 0.0, 1.0));

    if (mMorphEnabled)
    {
      // The shell bitmap already contains a finished grille. Paint the light
      // spill, then lay the same shell back over it so the metal, valve edges,
      // and shadows remain in the foreground. This reads as illumination from
      // behind the grille instead of a coloured film over the amplifier.
      // Use the normalized knob position directly. The equal linear slopes
      // make every drag increment visibly change both banks instead of
      // lingering near an endpoint and then appearing to switch at once.
      // A low idle floor keeps the opposite bank physically present.
      const float germanStrength = 0.12f + 0.88f * (1.f - morph);
      const float britishStrength = 0.12f + 0.88f * morph;
      const float glowY = art.T + art.H() * 0.405f;
      const float bankRadius = art.H() * 0.145f;
      const auto drawTubeBank = [&g, glowY, bankRadius](const float centerX, const IColor& color,
                                                        const float strength) {
        // A soft bank spill plus two tighter vertical sources follows the two
        // photographed glass valves on each side of the Forge shell.
        g.PathEllipse(centerX, glowY, bankRadius * 1.10f, bankRadius * 0.72f);
        g.PathFill(IPattern::CreateRadialGradient(
          centerX, glowY, bankRadius,
          {{color.WithOpacity(strength * 0.50f), 0.f}, {color.WithOpacity(strength * 0.22f), 0.48f},
           {COLOR_TRANSPARENT, 1.f}}));
        for (const float offset : {-0.75f, 0.75f})
        {
          const float tubeX = centerX + offset * bankRadius;
          g.PathEllipse(tubeX, glowY, bankRadius * 0.30f, bankRadius * 0.58f);
          g.PathFill(IPattern::CreateRadialGradient(
            tubeX, glowY, bankRadius * 0.42f,
            {{COLOR_WHITE.WithOpacity(strength * 0.18f), 0.f}, {color.WithOpacity(strength * 0.96f), 0.32f},
             {color.WithOpacity(strength * 0.30f), 0.66f}, {COLOR_TRANSPARENT, 1.f}}));
        }
      };
      drawTubeBank(art.L + art.W() * 0.255f, AccentHot, germanStrength);
      drawTubeBank(art.L + art.W() * 0.745f, MorphBrit, britishStrength);

      // Re-establish the photographed grille as the front plane. Keeping a
      // little of the light layer visible gives the black apertures depth
      // without tinting the leather, badge, or control panel.
      const IBlend foregroundBlend(EBlend::Default, 0.54f);
      g.DrawFittedBitmap(mBitmap, art, &foregroundBlend);
    }

    if (mShellStyle != ShellStyle::Studio)
    {
      // Dedicated product shells place their identity on the authored centre
      // metalwork, without adding a second bolted overlay.  REDLINE keeps its
      // subtitle on the solid dark centre while the side grille and valve
      // banks remain visible.
      const float badgeWidth = art.W() * 0.225f;
      const float badgeHeight = badgeWidth / 3.48f;
      const float badgeLeft = art.MW() - badgeWidth * 0.5f;
      const bool phoneSurface = art.H() < 600.f && art.W() / std::max(1.f, art.H()) > 1.75f;
      // Keep the badge and subtitle inside the grille on shallow heads.
      const float badgeTopFraction = phoneSurface ? 0.41f : 0.315f;
      const float badgeLift = phoneSurface ? std::min(badgeHeight, art.H() * 0.075f) : badgeHeight;
      const float badgeTop = art.T + art.H() * badgeTopFraction - badgeLift;
      const auto badgeBounds = IRECT(
        badgeLeft, badgeTop, badgeLeft + badgeWidth, badgeTop + badgeHeight);
      const IBlend badgeBlend(EBlend::Default, 0.94f);
      g.DrawFittedBitmap(mBadge, badgeBounds, &badgeBlend);

      if (!mSubtitle.empty())
      {
        const IColor titleAccent =
          mShellStyle == ShellStyle::ForgeStreamlined && mMorphEnabled
            ? IColor::LinearInterpolateBetween(AccentHot, MorphBrit, morph)
            : AccentHot;
        const auto nameBounds = IRECT(
          art.L + art.W() * 0.365f, badgeBounds.B + art.H() * 0.004f,
          art.R - art.W() * 0.365f, badgeBounds.B + art.H() * 0.078f);
        g.DrawText(Text(std::max(16.f, art.H() * 0.041f), titleAccent.WithOpacity(0.86f),
                        EAlign::Center, "Michroma-Regular"),
                   mSubtitle.c_str(), nameBounds);
      }
      return;
    }

    // Generic user-loaded heads retain the recessed neutral faceplate. The
    // dedicated DESS product shells use the direct-on-shell branch above.
    const float plateWidth = art.W() * 0.37f;
    const float plateHeight = art.H() * 0.28f;
    const auto plateBounds = IRECT(
      art.MW() - plateWidth * 0.5f, art.T + art.H() * 0.275f,
      art.MW() + plateWidth * 0.5f, art.T + art.H() * 0.275f + plateHeight);
    g.FillRoundRect(COLOR_BLACK.WithOpacity(0.72f), plateBounds.GetTranslated(0.f, 5.f), 8.f);
    g.PathRoundRect(plateBounds, 8.f);
    g.PathFill(IPattern::CreateLinearGradient(
      plateBounds, EDirection::Vertical,
      {{IColor(248, 45, 47, 49), 0.f}, {IColor(252, 13, 15, 17), 0.48f}, {IColor(255, 4, 5, 6), 1.f}}));
    g.DrawRoundRect(Metal.WithOpacity(0.58f), plateBounds, 8.f, nullptr, 1.4f);
    g.DrawRoundRect(COLOR_BLACK.WithOpacity(0.88f), plateBounds.GetPadded(-4.f), 6.f, nullptr, 1.f);
    g.DrawLine(COLOR_WHITE.WithOpacity(0.13f), plateBounds.L + 12.f, plateBounds.T + 2.f,
               plateBounds.R - 12.f, plateBounds.T + 2.f, nullptr, 1.f);
    for (int line = 0; line < 5; ++line)
    {
      const float y = plateBounds.T + 12.f + static_cast<float>(line) * (plateBounds.H() - 24.f) / 4.f;
      g.DrawLine(COLOR_WHITE.WithOpacity(line % 2 == 0 ? 0.025f : 0.014f), plateBounds.L + 10.f, y,
                 plateBounds.R - 10.f, y, nullptr, 1.f);
    }
    for (const auto& screw : {std::pair<float, float>{plateBounds.L + 11.f, plateBounds.T + 11.f},
                              std::pair<float, float>{plateBounds.R - 11.f, plateBounds.T + 11.f},
                              std::pair<float, float>{plateBounds.L + 11.f, plateBounds.B - 11.f},
                              std::pair<float, float>{plateBounds.R - 11.f, plateBounds.B - 11.f}})
    {
      g.FillCircle(IColor(255, 36, 38, 40), screw.first, screw.second, 3.6f);
      g.DrawCircle(Metal.WithOpacity(0.58f), screw.first, screw.second, 3.6f, nullptr, 1.f);
      g.FillCircle(COLOR_WHITE.WithOpacity(0.22f), screw.first - 0.8f, screw.second - 0.8f, 0.9f);
    }

    const float badgeWidth = plateBounds.W() * 0.72f;
    const float badgeHeight = badgeWidth / 3.48f;
    const float badgeLeft = plateBounds.MW() - (badgeWidth * 0.5f);
    const auto badgeBounds = IRECT(
      badgeLeft, plateBounds.T + plateBounds.H() * 0.075f, badgeLeft + badgeWidth,
      plateBounds.T + plateBounds.H() * 0.075f + badgeHeight);
    const IBlend badgeBlend(EBlend::Default, 0.96f);
    g.DrawFittedBitmap(mBadge, badgeBounds, &badgeBlend);
    if (!mSubtitle.empty())
    {
      const auto forgeBounds = IRECT(plateBounds.L + plateBounds.W() * 0.12f,
                                     plateBounds.T + plateBounds.H() * 0.64f,
                                     plateBounds.R - plateBounds.W() * 0.12f,
                                     plateBounds.B - plateBounds.H() * 0.055f);
      const float lineY = forgeBounds.MH();
      const IColor titleAccent =
        mMorphEnabled ? IColor::LinearInterpolateBetween(AccentHot, MorphBrit, morph) : AccentHot;
      g.DrawLine(Metal.WithOpacity(0.72f), forgeBounds.L, lineY, forgeBounds.L + forgeBounds.W() * 0.12f, lineY,
                 nullptr, 1.4f);
      g.DrawLine(Metal.WithOpacity(0.72f), forgeBounds.R - forgeBounds.W() * 0.12f, lineY, forgeBounds.R, lineY,
                 nullptr, 1.4f);
      g.DrawText(Text(std::max(28.f, art.H() * 0.07f), titleAccent, EAlign::Center, "Michroma-Regular"),
                 mSubtitle.c_str(), forgeBounds);
      g.DrawLine(titleAccent.WithOpacity(0.52f), plateBounds.L + plateBounds.W() * 0.22f, plateBounds.B - 2.f,
                 plateBounds.R - plateBounds.W() * 0.22f, plateBounds.B - 2.f, nullptr, 1.2f);
    }
  }

private:
  void ApplyShell()
  {
    switch (mShellStyle)
    {
      case ShellStyle::DualRed:
        mBitmap = mDualRedShell;
        break;
      case ShellStyle::ForgeStreamlined:
        mBitmap = mForgeShell;
        break;
      case ShellStyle::FireChrome:
        mBitmap = mFireShell;
        break;
      case ShellStyle::Studio:
      default:
        mBitmap = mStudioShell;
        break;
    }
  }

  IBitmap mStudioShell;
  IBitmap mDualRedShell;
  IBitmap mForgeShell;
  IBitmap mFireShell;
  IBitmap mBadge;
  ShellStyle mShellStyle = ShellStyle::Studio;
  bool mMorphEnabled = false;
  double mMorphValue = 0.5;
  std::string mMorphFrom{"GER"};
  std::string mMorphTo{"BRIT"};
  std::string mSubtitle;
};

class DessAmpKnobControl : public IVKnobControl, public IBitmapBase
{
public:
  DessAmpKnobControl(const IRECT& bounds, int paramIdx, const char* label, const IBitmap& bitmap, float diameter = 76.f)
  : IVKnobControl(
      bounds, paramIdx, label, DEFAULT_STYLE.WithShowLabel(false).WithShowValue(false).WithDrawFrame(false), true)
  , IBitmapBase(bitmap)
  , mDisplayLabel(label == nullptr ? "" : label)
  , mDiameter(diameter)
  {
    SetGearing(2.0);
    mMinimum = 0.0;
    mMaximum = 10.0;
    mDefaultNormalized = 0.5;
  }

  void OnRescale() override { mBitmap = GetUI()->GetScaledBitmap(mBitmap); }

  bool IsHit(float x, float y) const override { return mRECT.Contains(x, y); }

  IRECT GetKnobDragBounds() override { return IRECT(mRECT.L, mRECT.T, mRECT.R, mRECT.T + 120.f); }

  void SetPresentation(const char* label, double minimum, double maximum, double defaultValue, const char* unit,
                       double dialMinimumNormalized = 0.0, double dialMaximumNormalized = 1.0)
  {
    mDisplayLabel = label == nullptr ? "" : label;
    mMinimum = minimum;
    mMaximum = maximum;
    mUnit = unit == nullptr ? "" : unit;
    if (std::isfinite(minimum) && std::isfinite(maximum) && maximum > minimum)
      mDefaultNormalized = std::clamp((defaultValue - minimum) / (maximum - minimum), 0.0, 1.0);
    else
      mDefaultNormalized = 0.5;
    mDialMinimumNormalized = std::clamp(dialMinimumNormalized, 0.0, 1.0);
    mDialMaximumNormalized = std::clamp(dialMaximumNormalized, mDialMinimumNormalized, 1.0);
    mPlaceholder = false;
    mLeftEndpoint.clear();
    mRightEndpoint.clear();
    SetDirty(false);
  }

  void SetEndpointLabels(const char* left, const char* right)
  {
    mLeftEndpoint = left == nullptr ? "" : left;
    mRightEndpoint = right == nullptr ? "" : right;
    SetDirty(false);
  }

  void SetPlaceholderPresentation(const char* label, double minimum, double maximum, double defaultValue,
                                  const char* unit)
  {
    SetPresentation(label, minimum, maximum, defaultValue, unit);
    mPlaceholder = true;
    SetDirty(false);
  }

  void OnMouseDblClick(float x, float y, const IMouseMod& mod) override
  {
    if (IsDisabled())
      return;
    if (ValueBounds().Contains(x, y))
    {
      PromptUserInput(ValueBounds());
      return;
    }
    SetValue(mDefaultNormalized);
    SetDirty(true);
  }

  bool OnKeyDown(float x, float y, const IKeyPress& key) override
  {
    if (IsDisabled())
      return false;

    if (key.VK == kVK_RETURN || key.VK == (kVK_RETURN | 0x8000))
    {
      PromptUserInput(ValueBounds());
      return true;
    }

    double next = GetValue();
    const double step = key.S ? 0.001 : 0.01;
    switch (key.VK)
    {
      case kVK_LEFT:
      case kVK_DOWN: next -= step; break;
      case kVK_RIGHT:
      case kVK_UP: next += step; break;
      case kVK_HOME: next = 0.0; break;
      case kVK_END: next = 1.0; break;
      default: return false;
    }
    SetValue(std::clamp(next, 0.0, 1.0));
    SetDirty(true);
    return true;
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const float opacity = IsDisabled() ? 0.42f : 1.f;
    const IBlend localBlend(EBlend::Default, opacity);
    const auto knobRect = KnobBounds();
    const float diameter = knobRect.W();
    const float cx = knobRect.MW();
    const float cy = knobRect.MH();
    const float radius = diameter * 0.5f + 4.f;
    const double visualValue = mPlaceholder ? mDefaultNormalized : GetValue();
    const double dialValue = mDialMinimumNormalized + visualValue * (mDialMaximumNormalized - mDialMinimumNormalized);
    const float activeStart = mAngle1 + static_cast<float>(mDialMinimumNormalized) * (mAngle2 - mAngle1);
    const float angle = mAngle1 + static_cast<float>(dialValue) * (mAngle2 - mAngle1);

    g.DrawArc(Hairline.WithOpacity(opacity), cx, cy, radius, mAngle1, mAngle2, nullptr, 2.f);
    if (!IsDisabled())
      g.DrawArc((mMouseDown ? AccentHot : Accent).WithOpacity(opacity), cx, cy, radius, activeStart, angle, nullptr,
                mMouseDown ? 3.f : 2.f);
    g.DrawFittedBitmap(mBitmap, knobRect, &localBlend);
    if (mMouseIsOver && !IsDisabled())
      g.FillEllipse(COLOR_WHITE.WithOpacity(0.055f), knobRect.GetPadded(-5.f));
    g.DrawRadialLine((IsDisabled() ? Disabled : AccentHot).WithOpacity(opacity), cx, cy, angle, diameter * 0.12f,
                     diameter * 0.34f, nullptr, std::max(2.f, diameter * 0.035f));
    g.FillCircle((IsDisabled() ? Disabled : AccentHot).WithOpacity(opacity), cx, cy, std::max(1.5f, diameter * 0.025f));

    const auto labelBounds = IRECT(mRECT.L - 4.f, knobRect.T - 26.f, mRECT.R + 4.f, knobRect.T);
    // Keep the readout visually attached to its control. The prior baseline
    // left an unnecessary gap below the one-knob REDLINE layout.
    const auto valueBounds = ValueBounds();
    g.DrawText(Text(13.f, IsDisabled() ? Disabled : TextPrimary, EAlign::Center, "Roboto-Semibold"),
               mDisplayLabel.c_str(), labelBounds);
    if (!mLeftEndpoint.empty() || !mRightEndpoint.empty())
    {
      const auto endpoints = IRECT(mRECT.L + 1.f, valueBounds.T, mRECT.R - 1.f, valueBounds.B);
      const float sideWidth = endpoints.W() * 0.30f;
      const auto left = IRECT(endpoints.L, endpoints.T, endpoints.L + sideWidth, endpoints.B);
      const auto center = IRECT(endpoints.L + sideWidth, endpoints.T, endpoints.R - sideWidth, endpoints.B);
      const auto right = IRECT(endpoints.R - sideWidth, endpoints.T, endpoints.R, endpoints.B);
      g.DrawText(Text(12.f, AccentHot, EAlign::Near, "Roboto-Semibold"), mLeftEndpoint.c_str(), left);
      g.DrawText(Text(13.f, TextSecondary), DisplayValue().c_str(), center);
      g.DrawText(Text(12.f, MorphBrit, EAlign::Far, "Roboto-Semibold"), mRightEndpoint.c_str(), right);
    }
    else
      g.DrawText(Text(14.f, IsDisabled() ? Disabled : TextSecondary), DisplayValue().c_str(), valueBounds);
  }

private:
  IRECT KnobBounds() const
  {
    const float diameter = std::max(1.f, std::min({mDiameter, mRECT.W() - 8.f, mRECT.H() - 38.f}));
    return IRECT(0.f, 0.f, diameter, diameter)
      .GetTranslated(mRECT.MW() - diameter * 0.5f, mRECT.T + 18.f);
  }

  IRECT ValueBounds() const
  {
    const auto knobRect = KnobBounds();
    return IRECT(mRECT.L - 4.f, knobRect.B - 8.f, mRECT.R + 4.f, knobRect.B + 18.f);
  }

  std::string DisplayValue() const
  {
    const double normalized = std::clamp(mPlaceholder ? mDefaultNormalized : GetValue(), 0.0, 1.0);
    const double value = mMinimum + normalized * (mMaximum - mMinimum);
    std::ostringstream stream;
    const double span = std::abs(mMaximum - mMinimum);
    const bool morphReadout = mUnit == "%" && (!mLeftEndpoint.empty() || !mRightEndpoint.empty());
    const bool useInteger = !morphReadout && (mUnit == "%" || span >= 100.0);
    stream << std::fixed << std::setprecision(useInteger ? 0 : 1) << value;
    if (!mUnit.empty())
      stream << (mUnit == "%" ? "" : " ") << mUnit;
    return stream.str();
  }

  std::string mDisplayLabel;
  std::string mUnit;
  double mMinimum = 0.0;
  double mMaximum = 1.0;
  double mDefaultNormalized = 0.5;
  double mDialMinimumNormalized = 0.0;
  double mDialMaximumNormalized = 1.0;
  float mDiameter = 76.f;
  bool mPlaceholder = false;
  std::string mLeftEndpoint;
  std::string mRightEndpoint;
};

class DessAmpMeterControl : public IVPeakAvgMeterControl<>
{
public:
  explicit DessAmpMeterControl(const IRECT& bounds)
  : IVPeakAvgMeterControl<>(bounds, "", DEFAULT_STYLE.WithShowValue(false).WithDrawFrame(false).WithWidgetFrac(0.9),
                            EDirection::Vertical, {}, 0, -70.f, -0.01f, {})
  {
    SetPeakSize(1.f);
  }

  void DrawBackground(IGraphics& g, const IRECT& r) override
  {
    g.FillRoundRect(DessAmpUI::Window, r, 2.f);
    g.DrawRoundRect(DessAmpUI::Hairline, r, 2.f, nullptr, 1.f);
  }

  void DrawTrackHandle(IGraphics& g, const IRECT& r, int chIdx, bool aboveBaseValue) override
  {
    if (r.H() > 1.f)
      g.FillRect(DessAmpUI::Accent.WithOpacity(0.82f), r);
  }

  void DrawPeak(IGraphics& g, const IRECT& r, int chIdx, bool aboveBaseValue) override
  {
    g.FillRect(DessAmpUI::Destructive, r);
  }
};

class DessAmpToggleControl : public IVToggleControl
{
public:
  DessAmpToggleControl(const IRECT& bounds, int paramIdx, const char* label, bool large = false)
  : IVToggleControl(bounds, paramIdx, "", DEFAULT_STYLE, "OFF", "ON")
  , mLabel(label == nullptr ? "" : label)
  , mLarge(large)
  {
  }

  void SetPresentation(const char* label, const bool defaultOn)
  {
    mLabel = label == nullptr ? "" : label;
    mDefaultValue = defaultOn ? 1.0 : 0.0;
    SetDirty(false);
  }

  void OnMouseDblClick(float x, float y, const IMouseMod& mod) override
  {
    if (IsDisabled())
      return;
    SetValue(mDefaultValue);
    SetDirty(true);
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const bool on = GetValue() > 0.5;
    const float opacity = IsDisabled() ? 0.42f : 1.f;
    const auto labelBounds = mLarge ? mRECT.GetFromTop(32.f) : mRECT.GetFromTop(20.f);
    std::string displayLabel = mLabel;
    if (!mLarge && displayLabel.size() > 8U)
      displayLabel = displayLabel.substr(0U, 7U) + ".";
    const float labelSize = mLarge ? 14.f : (displayLabel.size() > 7U ? 10.f : 12.f);
    g.DrawText(Text(labelSize, IsDisabled() ? Disabled : TextPrimary), displayLabel.c_str(), labelBounds);
    const float trackW = mLarge ? 76.f : 42.f;
    const float trackH = mLarge ? 28.f : 20.f;
    const auto track = IRECT(0.f, 0.f, trackW, trackH)
                         .GetTranslated(mRECT.MW() - trackW * 0.5f, mRECT.B - trackH - (mLarge ? 18.f : 8.f));
    g.FillRoundRect((on ? Accent.WithOpacity(0.35f) : Window).WithOpacity(opacity), track, trackH * 0.5f);
    g.DrawRoundRect((on ? Accent : Hairline).WithOpacity(opacity), track, trackH * 0.5f, nullptr, 1.f);
    const float thumbRadius = trackH * 0.5f - 3.f;
    const float thumbX = on ? track.R - trackH * 0.5f : track.L + trackH * 0.5f;
    g.FillCircle((on ? AccentHot : Disabled).WithOpacity(opacity), thumbX, track.MH(), thumbRadius);
    if (on)
      g.FillCircle(Accent.WithOpacity(0.16f * opacity), thumbX, track.MH(), thumbRadius + 7.f);
    if (mLarge)
      g.DrawText(Text(12.f, on ? TextPrimary : TextSecondary), on ? "ON" : "BYPASS",
                 IRECT(mRECT.L, track.B + 2.f, mRECT.R, mRECT.B));
  }

private:
  std::string mLabel;
  bool mLarge = false;
  double mDefaultValue = 0.0;
};

// Selects between separately hash-verified packages that belong to one
// product family. This is deliberately not another host parameter: the exact
// selected package path is already serialized in plugin state, and changing
// voice must travel through the normal off-audio-thread package loader.
class DessAmpVariantSelectorControl : public IControl
{
public:
  using VariantAction = std::function<void(const std::string&)>;

  DessAmpVariantSelectorControl(const IRECT& bounds, VariantAction action)
  : IControl(bounds)
  , mAction(std::move(action))
  {
    SetTooltip("Choose a verified product voice.");
  }

  void SetVariants(const char* leftLabel, const char* rightLabel, const char* selectedVariant,
                   const bool enabled)
  {
    mLeftLabel = leftLabel == nullptr ? "US" : leftLabel;
    mRightLabel = rightLabel == nullptr ? "BRIT" : rightLabel;
    mSelectedVariant = selectedVariant == nullptr ? "" : selectedVariant;
    SetDisabled(!enabled);
    SetTooltip(enabled ? "Switch between the verified Redline US and Brit packages."
                       : "Both verified product voices are required.");
    SetDirty(false);
  }

  void OnMouseDown(float x, float y, const IMouseMod& mod) override
  {
    if (IsDisabled())
      return;
    const std::string requested = x < mRECT.MW() ? "us" : "brit";
    if (requested == mSelectedVariant || !mAction)
      return;
    SetDisabled(true);
    SetDirty(false);
    mAction(requested);
  }

  void OnMouseOver(float x, float y, const IMouseMod& mod) override
  {
    GetUI()->SetMouseCursor(IsDisabled() ? ECursor::ARROW : ECursor::HAND);
    IControl::OnMouseOver(x, y, mod);
  }

  void OnMouseOut() override
  {
    GetUI()->SetMouseCursor();
    IControl::OnMouseOut();
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const float opacity = IsDisabled() ? 0.42f : 1.f;
    g.DrawText(Text(12.f, TextPrimary.WithOpacity(opacity)), "VOICE", mRECT.GetFromTop(22.f));
    const float trackWidth = std::min(126.f, mRECT.W() - 8.f);
    const float trackHeight = 32.f;
    const auto track = IRECT(0.f, 0.f, trackWidth, trackHeight)
                         .GetTranslated(mRECT.MW() - trackWidth * 0.5f, mRECT.T + 34.f);
    const auto left = track.GetFromLeft(track.W() * 0.5f);
    const auto right = track.GetFromRight(track.W() * 0.5f);
    g.FillRoundRect(Window.WithOpacity(opacity), track, 6.f);
    g.DrawRoundRect(Hairline.WithOpacity(opacity), track, 6.f, nullptr, 1.f);
    const bool britSelected = mSelectedVariant == "brit";
    const auto selected = britSelected ? right : left;
    g.FillRoundRect(Accent.WithOpacity(0.32f * opacity), selected.GetPadded(-2.f), 4.f);
    g.DrawRoundRect(AccentHot.WithOpacity(0.8f * opacity), selected.GetPadded(-2.f), 4.f, nullptr, 1.f);
    g.DrawText(Text(12.f, britSelected ? TextSecondary : TextPrimary), mLeftLabel.c_str(), left);
    g.DrawText(Text(12.f, britSelected ? TextPrimary : TextSecondary), mRightLabel.c_str(), right);
  }

private:
  VariantAction mAction;
  std::string mLeftLabel{"US"};
  std::string mRightLabel{"BRIT"};
  std::string mSelectedVariant{"us"};
};

class DessAmpPowerControl : public IVToggleControl
{
public:
  DessAmpPowerControl(const IRECT& bounds, int paramIdx)
  : IVToggleControl(bounds, paramIdx, "", DEFAULT_STYLE, "BYPASS", "ON")
  {
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const bool on = GetValue() > 0.5;
    const IColor color = on ? AccentHot : TextSecondary;
    if (mMouseIsOver)
      g.FillRoundRect(COLOR_WHITE.WithOpacity(0.055f), mRECT, 6.f);
    const float cx = mRECT.MW(), cy = mRECT.MH() + 1.f;
    g.DrawArc(color, cx, cy + 2.f, 11.f, 38.f, 322.f, nullptr, 2.2f);
    g.DrawLine(color, cx, cy - 13.f, cx, cy + 1.f, nullptr, 2.2f);
    if (on)
      g.FillCircle(Accent.WithOpacity(0.12f), cx, cy, 18.f);
  }
};

class DessAmpTabsControl : public IControl
{
public:
  DessAmpTabsControl(const IRECT& bounds, IActionFunction action, DessAmpPage page)
  : IControl(bounds, action), mPage(page) {}
  DessAmpPage Selected() const { return mPage; }
  void Configure(DessAmpPage page, bool rigTab)
  {
    mPage = page;
    mTabCount = rigTab ? 4 : 3;
    SetDirty(false);
  }
  void OnMouseDown(float x, float, const IMouseMod&) override
  {
    mPage = static_cast<DessAmpPage>(std::clamp(static_cast<int>(mTabCount * (x - mRECT.L) / mRECT.W()), 0, mTabCount - 1));
    SetDirty(true);
  }
  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const char* names[] = {"PRE", "AMP", "CAB", "RIG"};
    for (int i = 0; i < mTabCount; ++i)
    {
      const auto tab = mRECT.SubRectHorizontal(mTabCount, i);
      g.DrawText(Text(14.f, i == mPage ? TextPrimary : TextSecondary), names[i], tab);
      if (i == mPage)
        g.FillRect(Accent, IRECT(tab.L + 8.f, tab.B - 2.f, tab.R - 8.f, tab.B));
    }
  }
private:
  int mTabCount = 3;
  DessAmpPage mPage;
};

class DessAmpFactoryStrip : public IControl
{
public:
  DessAmpFactoryStrip(const IRECT& bounds, const std::vector<std::string>& names,
                      std::function<void(int)> select)
  : IControl(bounds), mNames(names), mSelect(std::move(select)) {}
  void Select(int index) { if (index != mCurrent) { mCurrent = index; SetDirty(false); } }
  void OnMouseDown(float x, float y, const IMouseMod&) override
  {
    const int count = static_cast<int>(mNames.size());
    for (int i = 0; i < count; ++i)
      if (mRECT.SubRectHorizontal(count, i).Contains(x, y)) { mSelect(i); return; }
  }
  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const bool large = mRECT.H() >= 50.f;
    const int count = static_cast<int>(mNames.size());
    for (int i = 0; i < count; ++i)
    {
      const auto cell = mRECT.SubRectHorizontal(count, i);
      const auto button = cell.GetPadded(-3.f);
      if (i == mCurrent || large)
        g.FillRoundRect(i == mCurrent ? SurfaceRaised : Surface.WithOpacity(0.64f), button, large ? 7.f : 4.f);
      g.DrawText(Text(large ? 15.f : 13.f, i == mCurrent ? TextPrimary : TextSecondary,
                      EAlign::Center, large ? "Roboto-Semibold" : "Roboto-Regular"), mNames[i].c_str(), cell);
      if (i == mCurrent)
        g.FillRect(Accent, IRECT(cell.L + 8.f, cell.B - 3.f, cell.R - 8.f, cell.B - 1.f));
    }
  }
private:
  std::vector<std::string> mNames;
  std::function<void(int)> mSelect;
  int mCurrent = -1;
};

class DessAmpPedalBank : public IControl
{
public:
  DessAmpPedalBank(const IRECT& bounds, std::array<IBitmap, 4> shells, bool available)
  : IControl(bounds, {kBoostModel, kBoostActive}), mShells(std::move(shells)), mAvailable(available)
  { SetTooltip(available ? "Select a drive. Click the active drive to bypass." : "Drive bank unavailable"); }

  IRECT LevelBounds(int index) const
  {
    const auto face = Face(index);
    if (index == kVerdantDriveIndex)
      return KnobRow(face).SubRectHorizontal(2, 0);
    const float halfWidth = std::min(44.f, face.W() * 0.35f);
    return IRECT(face.MW()-halfWidth, face.T+face.H()*0.08f, face.MW()+halfWidth, face.T+face.H()*0.42f);
  }
  IRECT ToneBounds() const { return KnobRow(Face(kVerdantDriveIndex)).SubRectHorizontal(2, 1); }

  void SetHasTone(bool hasTone)
  {
    mHasTone = hasTone;
    SetDirty(false);
  }

  void SetDirty(bool triggerAction = true, int valIdx = kNoValIdx) override
  {
    IControl::SetDirty(triggerAction, valIdx);
    if (!GetUI()) return;
    for (int i = 0; i < 4; ++i)
      if (auto* level = GetUI()->GetControlWithTag(kCtrlTagPlayerDriveKnob + i))
        level->SetDisabled(!mAvailable);
    if (auto* tone = GetUI()->GetControlWithTag(kCtrlTagVerdantTone))
      tone->SetDisabled(!mHasTone || !mAvailable);
  }
  void OnMouseDown(float x, float y, const IMouseMod&) override
  {
    if (!mAvailable) return;
    int index = 0;
    while (index < 4 && !Face(index).Contains(x, y)) ++index;
    if (index == 4) return;
    const bool wasOn = index == std::lround(3 * GetValue(0)) && GetValue(1) >= 0.5;
    SetValueFromUserInput(index / 3.0, 0);
    SetValueFromUserInput(wasOn ? 0.0 : 1.0, 1);
  }
  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const char* names[] = {"OXIDE 08", "AMBER 01", "VERDANT 09", "NOCTURNE"};
    for (int i = 0; i < 4; ++i)
    {
      const auto face = Face(i);
      g.DrawFittedBitmap(mShells[i], face);
      const bool on = mAvailable && i == std::lround(3 * GetValue(0)) && GetValue(1) >= 0.5;
      g.DrawText(Text(16.f, i == 1 || i == 2 ? Window : TextPrimary), names[i], face);
      const float cx = face.MW(), cy = face.T + face.H() * 0.74f;
      const float r = std::min(16.f, face.W() * 0.09f);
      g.FillCircle(on ? AccentHot : Disabled, cx, cy - 1.65f * r, r * 0.26f);
      g.FillCircle(Window, cx, cy, r + 2.f);
      g.FillCircle(Metal, cx, cy, r);
      g.DrawCircle(TextSecondary, cx, cy, r * 0.8f);
    }
    if (!mAvailable)
      g.DrawText(Text(14.f, TextSecondary), "Drive bank unavailable", mRECT.GetFromTop(26.f));
  }
  void OnRescale() override { for (auto& shell : mShells) shell = GetUI()->GetScaledBitmap(shell); }
private:
  static IRECT KnobRow(const IRECT& face)
  {
    const float halfWidth = face.W() * 0.36f;
    return IRECT(face.MW() - halfWidth, face.T + face.H() * 0.08f,
                 face.MW() + halfWidth, face.T + face.H() * 0.42f);
  }
  bool mHasTone = false;
  IRECT Face(int index) const
  {
    const auto cell = mRECT.W() < 600.f && mRECT.H() > mRECT.W()
                        ? mRECT.SubRectVertical(2, index / 2).SubRectHorizontal(2, index % 2)
                        : mRECT.SubRectHorizontal(4, index);
    const float height = std::min(cell.H() * 0.90f, cell.W() * 1.35f);
    const float width = height * 0.62f;
    return IRECT(cell.MW()-width/2, cell.MH()-height/2, cell.MW()+width/2, cell.MH()+height/2);
  }
  std::array<IBitmap, 4> mShells;
  bool mAvailable;
};

class DessAmpReservedToggleControl : public IControl
{
public:
  DessAmpReservedToggleControl(const IRECT& bounds, const char* label, bool channel = false)
  : IControl(bounds)
  , mLabel(label == nullptr ? "" : label)
  , mChannel(channel)
  {
    SetTooltip("Not available in this model.");
  }

  void SetUnavailable()
  {
    mHasFixedState = false;
    mFixedOn = false;
    SetTooltip("Not available in this model.");
    SetDirty(false);
  }

  void SetFixedState(const char* label, const bool on)
  {
    mLabel = label == nullptr ? "" : label;
    mHasFixedState = true;
    mFixedOn = on;
    SetTooltip((mLabel + (on ? ": fixed ON for this amplifier." : ": fixed OFF for this amplifier.")).c_str());
    SetDirty(false);
  }

  void OnMouseOver(float x, float y, const IMouseMod& mod) override
  {
    GetUI()->SetMouseCursor(ECursor::ARROW);
    IControl::OnMouseOver(x, y, mod);
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const float opacity = mHasFixedState ? 0.62f : 0.42f;
    g.DrawText(Text(12.f, mFixedOn ? AccentHot : Disabled), mLabel.c_str(), mRECT.GetFromTop(20.f));
    const auto bezel = IRECT(0.f, 0.f, 26.f, 36.f).GetTranslated(mRECT.MW() - 13.f, mRECT.T + 27.f);
    g.FillRoundRect(COLOR_BLACK.WithOpacity(0.5f), bezel, 6.f);
    g.DrawRoundRect(Metal.WithOpacity(opacity), bezel, 6.f, nullptr, 1.f);
    const float leverX = mRECT.MW();
    const float leverDirection = mFixedOn ? 5.f : -5.f;
    g.DrawLine(Metal.WithOpacity(opacity), leverX, bezel.T + 9.f,
               leverX + (mChannel ? -leverDirection : leverDirection), bezel.B - 8.f, nullptr, 4.f);
    if (mFixedOn)
      g.FillCircle(Accent.WithOpacity(0.18f), mRECT.MW(), mRECT.B - 5.f, 8.f);
    g.FillCircle((mFixedOn ? AccentHot : Disabled).WithOpacity(opacity), mRECT.MW(), mRECT.B - 5.f, 3.f);
  }

private:
  std::string mLabel;
  bool mChannel = false;
  bool mHasFixedState = false;
  bool mFixedOn = false;
};

class DessAmpModelSelectorControl : public IControl
{
public:
  using PresetPathAction = std::function<void(const char*)>;
  using PresetSaveAction = std::function<void(const char*, bool)>;
  using PresetNavigationAction = std::function<void(int)>;
  DessAmpModelSelectorControl(const IRECT& bounds, PresetPathAction loadPreset, PresetSaveAction savePreset,
                             PresetNavigationAction navigatePreset)
  : IControl(bounds), mLoadPreset(std::move(loadPreset)), mSavePreset(std::move(savePreset)),
    mNavigatePreset(std::move(navigatePreset)) { SetTextEntryLength(60); }
  void SetQuick(bool quick) { mQuick = quick; }
  void ChoosePreset()
  {
    if (!IsReady() || mPresetDirectory.empty()) return;
    mBank.Clear();
    mBank.SetRootTitle((mName + " presets").c_str());
    mBankFiles = dessamp::preset::Enumerate(mPresetDirectory);
    mBankScope = mPresetDirectory;
    mBankHasDefault = mHasDefault;
    if (mBankHasDefault)
      mBank.AddItem("Model Default", -1, mPresetPath.empty() ? IPopupMenu::Item::kChecked : 0);
    for (const auto& file : mBankFiles)
      mBank.AddItem(dessamp::preset::Label(file).c_str(), -1, file == mPresetPath ? IPopupMenu::Item::kChecked : 0);
    if (!mBankHasDefault && mBankFiles.empty())
      mBank.AddItem("No saved presets yet", -1, IPopupMenu::Item::kDisabled);
    GetUI()->CreatePopupMenu(*this, mBank, mQuick ? QuickPicker() : PresetFieldHitBounds(), kNoValIdx);
  }
  void OnPopupMenuSelection(IPopupMenu* menu, int) override
  {
    if (!menu || !mLoadPreset || mBankScope != mPresetDirectory) return;
    const int index = menu->GetChosenItemIdx();
    if (mBankHasDefault && index == 0) { mLoadPreset(""); return; }
    const int i = index - (mBankHasDefault ? 1 : 0);
    if (i >= 0 && i < static_cast<int>(mBankFiles.size()))
      mLoadPreset(mBankFiles[i].u8string().c_str());
  }
  void ImportPreset()
  {
    WDL_String fileName;
    WDL_String path(mPresetDirectory.c_str());
    GetUI()->PromptForFile(
      fileName, path, EFileAction::Open, "desspreset", [this](const WDL_String& selected, const WDL_String&) {
        if (selected.GetLength() && mLoadPreset)
          mLoadPreset(selected.Get());
      });
  }

  void SavePreset()
  {
    if (!mPresetPath.empty())
    {
      if (mSavePreset)
        mSavePreset(mPresetPath.c_str(), false);
      return;
    }

    SavePresetAs();
  }

  void SavePresetAs()
  {
    if (!IsReady() || mPresetDirectory.empty()) return;
    mNamingScope = mPresetDirectory;
    // Returning an empty field lets the user type a name without first
    // deleting a suggested filename. Cancel never writes a file.
    GetUI()->CreateTextEntry(*this, DessAmpUI::Text(15.f), mQuick ? QuickPicker() : PresetFieldHitBounds(),
                             "", kNoValIdx);
  }
  void OnTextEntryCompletion(const char* text, int) override
  {
    if (mNamingScope != mPresetDirectory || !mSavePreset || !text) return;
    mSavePreset(text, true);
  }
  void SetPresetState(const char* label, const char* path, const char* directory, bool hasDefault,
                      const bool modified)
  {
    mPresetLabel = (label == nullptr || label[0] == '\0') ? "Model Default" : label;
    mPresetPath = path == nullptr ? "" : path;
    mPresetDirectory = directory == nullptr ? "" : directory;
    mHasDefault = hasDefault;
    mPresetModified = modified;
    SetDirty(false);
  }

  void SetReady(const char* name, int controls, const char* runtime)
  {
    mState = State::Ready;
    mName = (name == nullptr || name[0] == '\0') ? "Conditioned amplifier" : name;
    if (mName.size() > 48U)
      mName = mName.substr(0U, 45U) + "...";
    std::ostringstream summary;
    summary << controls << (controls == 1 ? " control" : " controls");
    if (runtime != nullptr && runtime[0] != '\0')
      summary << "  \xC2\xB7  " << runtime;
    mSummary = summary.str();
    SetDirty(false);
  }

  void SetLoading()
  {
    mState = State::Loading;
    mName = "Loading amplifier";
    mSummary.clear();
    SetDirty(false);
  }

  void SetEmpty()
  {
    mState = State::Empty;
    mName = "No amplifier loaded";
    mSummary = "Select a head or open a .dess with CUSTOM.";
    SetDirty(false);
  }

  void SetError()
  {
    mState = State::Error;
    mName = "This amplifier could not be loaded.";
    mSummary = "Check the amplifier package.";
    SetDirty(false);
  }

  void SetMissing()
  {
    mState = State::Missing;
    mName = "Amplifier file not found";
    mSummary = "Restore the original package to its saved location.";
    SetDirty(false);
  }

  bool IsReady() const { return mState == State::Ready; }

  void OnMsgFromDelegate(int msgTag, int dataSize, const void* pData) override
  {
    if (msgTag == kMsgTagLoadedPlayerManifest && pData != nullptr && dataSize > 0)
    {
      std::string path(static_cast<const char*>(pData), static_cast<std::size_t>(dataSize));
      while (!path.empty() && path.back() == '\0')
        path.pop_back();
      mPath = path;
      if (!mPath.empty())
        SetTooltip(mPath.c_str());
    }
    else if (msgTag == kMsgTagLoadFailed)
    {
      SetError();
    }
  }

  void OnMouseDown(float x, float y, const IMouseMod& mod) override
  {
    if (mQuick)
    {
      if (QuickSave().Contains(x,y)) SavePreset();
      else if (QuickNew().Contains(x,y)) SavePresetAs();
      else ChoosePreset();
      return;
    }
    if (PresetPreviousBounds().Contains(x, y))
    {
      if (mNavigatePreset) mNavigatePreset(-1);
    }
    else if (PresetFieldHitBounds().Contains(x, y))
      ChoosePreset();
    else if (PresetNextBounds().Contains(x, y))
    {
      if (mNavigatePreset) mNavigatePreset(1);
    }
    else if (PresetSaveHitBounds().Contains(x, y))
      mod.S ? SavePresetAs() : SavePreset();
    else if (PresetSaveAsHitBounds().Contains(x, y))
      SavePresetAs();
  }

  void OnMouseOver(float x, float y, const IMouseMod& mod) override
  {
    const bool interactive = PresetPreviousBounds().Contains(x, y) || PresetFieldHitBounds().Contains(x, y)
                             || PresetNextBounds().Contains(x, y) || PresetSaveHitBounds().Contains(x, y)
                             || PresetSaveAsHitBounds().Contains(x, y);
    if (PresetPreviousBounds().Contains(x, y))
      SetTooltip("Previous Preset");
    else if (PresetFieldHitBounds().Contains(x, y))
      SetTooltip(mPresetModified ? "Preset has unsaved changes." : "Choose Preset");
    else if (PresetNextBounds().Contains(x, y))
      SetTooltip("Next Preset");
    else if (PresetSaveHitBounds().Contains(x, y))
      SetTooltip(mod.S || mPresetPath.empty() ? "Save As New Preset\xE2\x80\xA6"
                                              : "Save Current Preset");
    else if (PresetSaveAsHitBounds().Contains(x, y))
      SetTooltip("Save As New Preset\xE2\x80\xA6");
    GetUI()->SetMouseCursor(interactive ? ECursor::HAND : ECursor::ARROW);
    IControl::OnMouseOver(x, y, mod);
  }

  void OnMouseOut() override
  {
    SetTooltip(mPath.empty() ? "Factory amplifier" : mPath.c_str());
    GetUI()->SetMouseCursor();
    IControl::OnMouseOut();
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    if (mQuick)
    {
      const auto picker = QuickPicker();
      const auto save = QuickSave(), add = QuickNew();
      const auto button = [&](const IRECT& r, const char* label, bool highlight)
      {
        const auto frame = r.GetPadded(-2.f);
        g.FillRoundRect(highlight ? SurfaceRaised : Surface, frame, 5.f);
        g.DrawRoundRect(highlight ? Accent : Hairline, frame, 5.f);
        g.DrawText(Text(13.f, IsReady() ? TextPrimary : Disabled), label, frame);
      };
      button(save, "SAVE", mPresetModified);
      button(add, "NEW", false);
      const auto field = picker.GetPadded(-2.f);
      g.FillRoundRect(Surface, field, 5.f);
      g.DrawRoundRect(Hairline, field, 5.f);
      g.DrawText(Text(13.f, TextPrimary, EAlign::Near), mPresetLabel.c_str(),
                 field.GetReducedFromLeft(9.f).GetReducedFromRight(26.f));
      const float x = field.R - 12.f, y = field.MH();
      g.DrawLine(TextSecondary, x-3.f,y-1.5f,x,y+1.5f);
      g.DrawLine(TextSecondary, x,y+1.5f,x+3.f,y-1.5f);
      if (mPresetModified) g.FillCircle(Accent, field.R-23.f,y,2.f);
      return;
    }
    const auto row1 = AmplifierRow();
    const auto row2 = PresetRow();
    const bool compact = IsCompact();
    const bool phoneCompact = IsPhoneCompact();
    const float labelWidth = phoneCompact ? 0.f : compact ? 72.f : 96.f;
    const float stateX = row1.L + labelWidth + 5.f;
    const float contentX = stateX + 14.f;
    const float nameLeft = phoneCompact ? row1.L + 4.f : contentX;
    const float nameRight = row1.R - 8.f;
    if (!phoneCompact)
      g.DrawText(
        Text(12.f, TextSecondary, EAlign::Near), "AMPLIFIER", IRECT(row1.L, row1.T, row1.L + labelWidth, row1.B));
    const IColor stateColor = mState == State::Ready
                                ? Accent
                                : (mState == State::Error || mState == State::Missing) ? Destructive : Disabled;
    if (!phoneCompact)
      g.FillCircle(stateColor, stateX, row1.MH(), 4.f);
    if (compact)
      g.DrawText(Text(phoneCompact ? 13.f : 12.f, TextPrimary, EAlign::Near), mName.c_str(),
                 IRECT(nameLeft, row1.T, nameRight, row1.B));
    else
    {
      g.DrawText(Text(14.f, TextPrimary, EAlign::Near), mName.c_str(), IRECT(nameLeft, row1.T, nameRight, row1.B - 8.f));
      g.DrawText(
        Text(12.f, TextSecondary, EAlign::Near), mSummary.c_str(), IRECT(nameLeft, row1.T + 20.f, nameRight, row1.B));
    }

    g.DrawLine(Hairline, row2.L, row2.T, row2.R, row2.T, nullptr, 1.f);
    if (!phoneCompact)
      g.DrawText(Text(12.f, TextSecondary, EAlign::Near), "PRESET",
                 IRECT(row2.L, row2.T, row2.L + (compact ? 52.f : 96.f), row2.B));
    g.DrawText(Text(18.f, TextPrimary), "‹", PresetPreviousBounds());
    const auto presetField = PresetFieldVisualBounds();
    g.FillRoundRect(Window.WithOpacity(0.72f), presetField, 4.f);
    g.DrawRoundRect(Hairline, presetField, 4.f, nullptr, 1.f);
    auto presetTextBounds = presetField.GetHPadded(phoneCompact ? -8.f : -12.f);
    if (mPresetModified)
      presetTextBounds.R -= 16.f;
    g.DrawText(Text(14.f, TextSecondary, EAlign::Near), mPresetLabel.c_str(), presetTextBounds);
    if (mPresetModified)
      g.FillCircle(AccentHot, presetField.R - 10.f, presetField.MH(), 3.5f);
    g.DrawText(Text(18.f, TextPrimary), "›", PresetNextBounds());
    const auto saveBounds = PresetSaveVisualBounds();
    const auto saveAsBounds = PresetSaveAsVisualBounds();
    g.FillRoundRect(Window.WithOpacity(0.72f), saveBounds, 4.f);
    g.DrawRoundRect(TextSecondary.WithOpacity(0.62f), saveBounds, 4.f, nullptr, 1.f);
    g.FillRoundRect(Window.WithOpacity(0.72f), saveAsBounds, 4.f);
    g.DrawRoundRect(TextSecondary.WithOpacity(0.62f), saveAsBounds, 4.f, nullptr, 1.f);
    g.DrawText(Text(phoneCompact ? 10.f : 11.f, TextPrimary), "SAVE", saveBounds);
    g.DrawText(Text(phoneCompact ? 9.f : 11.f, TextPrimary), "NEW", saveAsBounds);
  }

private:
  enum class State
  {
    Empty,
    Loading,
    Ready,
    Error,
    Missing
  };

  IRECT AmplifierRow() const { return IRECT(mRECT.L, mRECT.T, mRECT.R, mRECT.T + mRECT.H() * 0.5f); }
  IRECT PresetRow() const { return IRECT(mRECT.L, mRECT.T + mRECT.H() * 0.5f, mRECT.R, mRECT.B); }
  bool IsCompact() const { return mRECT.W() < 500.f; }
  bool IsPhoneCompact() const { return mRECT.W() < 375.f && mRECT.H() <= 125.f; }
  IRECT PresetPreviousBounds() const
  {
    const auto row = PresetRow();
    return IsPhoneCompact() ? IRECT(row.L, row.T, row.L + 44.f, row.B)
           : IsCompact() ? IRECT(row.L + 50.f, row.T, row.L + 94.f, row.B)
                       : IRECT(row.L + 94.f, row.T, row.L + 138.f, row.B);
  }
  IRECT PresetFieldHitBounds() const
  {
    const auto row = PresetRow();
    return IsPhoneCompact() ? IRECT(row.L + 46.f, row.T, row.R - 150.f, row.B)
           : IsCompact() ? IRECT(row.L + 96.f, row.T, row.R - 142.f, row.B)
                       : IRECT(row.L + 144.f, row.T, row.R - 190.f, row.B);
  }
  IRECT PresetFieldVisualBounds() const { return PresetFieldHitBounds().GetVPadded(-6.f); }
  IRECT PresetNextBounds() const
  {
    const auto row = PresetRow();
    const auto field = PresetFieldHitBounds();
    return IRECT(field.R + 2.f, row.T, field.R + 46.f, row.B);
  }
  IRECT PresetSaveAreaBounds() const
  {
    const auto row = PresetRow();
    // Keep both actions independently tappable. The 100-point phone area
    // yields 54- and 44-point hit widths after the two-point separator.
    return IRECT(row.R - (IsPhoneCompact() ? 100.f : IsCompact() ? 92.f : 142.f), row.T, row.R, row.B);
  }
  IRECT PresetSaveHitBounds() const
  {
    const auto area = PresetSaveAreaBounds();
    const float split = area.L + area.W() * (IsPhoneCompact() ? 0.55f : 0.5f);
    return IRECT(area.L, area.T, split - 1.f, area.B);
  }
  IRECT PresetSaveAsHitBounds() const
  {
    const auto area = PresetSaveAreaBounds();
    const float split = area.L + area.W() * (IsPhoneCompact() ? 0.55f : 0.5f);
    return IRECT(split + 1.f, area.T, area.R, area.B);
  }
  IRECT PresetSaveVisualBounds() const { return PresetSaveHitBounds().GetVPadded(-6.f); }
  IRECT PresetSaveAsVisualBounds() const { return PresetSaveAsHitBounds().GetVPadded(-6.f); }

  PresetPathAction mLoadPreset;
  IRECT QuickNew() const { return mRECT.GetFromRight(44.f); }
  IRECT QuickSave() const { return QuickNew().GetTranslated(-48.f, 0.f).GetHPadded(2.f); }
  IRECT QuickPicker() const { return mRECT.GetReducedFromRight(94.f); }
  IPopupMenu mBank;
  std::vector<std::filesystem::path> mBankFiles;
  std::string mBankScope, mNamingScope;
  bool mQuick = false, mHasDefault = false, mBankHasDefault = false;
  PresetSaveAction mSavePreset;
  PresetNavigationAction mNavigatePreset;
  State mState = State::Empty;
  std::string mName{"No amplifier loaded"};
  std::string mSummary{"Select a head or open a .dess with CUSTOM."};
  std::string mPath;
  std::string mPresetLabel{"Model Default"};
  std::string mPresetPath;
  std::string mPresetDirectory;
  bool mPresetModified = false;
};

class DessAmpCabSelectorControl : public IControl
{
public:
  using IRChoice = std::pair<std::string, std::string>;
  using Images = std::vector<std::pair<std::string, IBitmap>>;
  using IRPathAction = std::function<void(const char*)>;

  DessAmpCabSelectorControl(const IRECT& bounds, std::vector<IRChoice> choices, Images images,
                            IRPathAction loadIR, IActionFunction toggle)
  : IControl(bounds, kNoParameter, std::move(toggle)), mChoices(std::move(choices)),
    mImages(std::move(images)), mLoadIR(std::move(loadIR))
  {
    SetTooltip("Select a cabinet or load a mono WAV");
  }

  void ChooseFile()
  {
    WDL_String fileName, path;
    GetUI()->PromptForFile(fileName, path, EFileAction::Open, "wav",
      [this](const WDL_String& selected, const WDL_String&) {
        if (selected.GetLength())
          Load(selected.Get());
      });
  }

  void OnMsgFromDelegate(int msgTag, int size, const void* data) override
  {
    if (msgTag == kMsgTagLoadedIR && data && size > 0)
    {
      std::string path(static_cast<const char*>(data), static_cast<size_t>(size));
      while (!path.empty() && path.back() == '\0')
        path.pop_back();
      const auto selected = std::filesystem::path(path).lexically_normal();
      mSelected = -1;
      for (size_t i = 0; i < mChoices.size(); ++i)
        if (std::filesystem::path(mChoices[i].second).lexically_normal() == selected)
          mSelected = static_cast<int>(i);
      mName = mSelected < 0 ? selected.filename().string() : mChoices[mSelected].first;
      mStatus = "IR ready";
    }
    else if (msgTag == kMsgTagLoadFailed)
    {
      mSelected = -1;
      mName.clear();
      mStatus = "IR unavailable";
    }
    else
      return;
    SetDirty(false);
  }

  void OnMouseDown(float x, float y, const IMouseMod&) override
  {
    if (LoadBounds().Contains(x, y))
      ChooseFile();
    else if (ToggleBounds().Contains(x, y))
    {
      SetDirty(true);
    }
    else
    {
      const int choice = ChoiceAt(x, y);
      if (choice >= 0)
        Load(mChoices[choice].second.c_str());
    }
  }

  void OnMouseOver(float x, float y, const IMouseMod& mod) override
  {
    const bool action = LoadBounds().Contains(x,y) || ToggleBounds().Contains(x,y) || ChoiceAt(x,y) >= 0;
    GetUI()->SetMouseCursor(action ? ECursor::HAND : ECursor::ARROW);
    IControl::OnMouseOver(x,y,mod);
  }

  void OnMouseOut() override { GetUI()->SetMouseCursor(); IControl::OnMouseOut(); }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const bool active = GetDelegate()->GetParam(kIRToggle)->Bool();
    const bool custom = mSelected < 0;
    const auto left = MainArea();
    const auto row = Actions();
    const auto title = left.GetFromTop(std::min(42.f, left.H() * 0.16f));
    const auto hero = IRECT(left.L, title.B, left.R, row.T - 6.f).GetPadded(-4.f);
    const auto* name = mName.empty() ? mStatus.c_str() : mName.c_str();
    g.DrawText(Text(19.f, TextPrimary), name, title);
    DrawShell(g, custom ? "" : mChoices[mSelected].first, hero, custom || !active ? &BLEND_25 : nullptr);
    if (custom)
      g.DrawText(Text(40.f, TextPrimary), "IR", hero);
    if (!mName.empty() && mStatus == "Loading IR")
      g.DrawText(Text(12.f, TextSecondary), mStatus.c_str(), hero.GetFromBottom(18.f));

    const auto shelf = Shelf();
    g.DrawText(Text(12.f, TextSecondary, EAlign::Near), "CABINETS", shelf.GetFromTop(18.f));
    for (size_t i = 0; i < mChoices.size(); ++i)
    {
      const bool selected = static_cast<int>(i) == mSelected;
      const auto card = Card(static_cast<int>(i));
      g.FillRoundRect(selected ? SurfaceRaised : Surface, card, 5.f);
      g.DrawRoundRect(selected ? Accent : Hairline, card, 5.f, nullptr, selected ? 2.f : 1.f);
      if (selected) g.FillCircle(Accent, card.R - 9.f, card.T + 9.f, 3.f);
      DrawShell(g, mChoices[i].first, card.GetReducedFromBottom(22.f).GetPadded(-5.f), nullptr);
      g.DrawText(Text(12.f, TextPrimary), mChoices[i].first.c_str(), card.GetFromBottom(22.f));
    }

    const auto load = LoadBounds(), toggle = ToggleBounds();
    g.FillRoundRect(SurfaceRaised, load, 4.f);
    g.DrawRoundRect(Hairline, load, 4.f, nullptr, 1.f);
    g.DrawText(Text(12.f, TextPrimary), "LOAD IR", load);
    g.FillRoundRect(active ? Accent.WithOpacity(0.24f) : Surface, toggle, 4.f);
    g.DrawRoundRect(active ? Accent : Hairline, toggle, 4.f, nullptr, 1.f);
    g.DrawText(Text(12.f, TextPrimary), active ? "CAB ON" : "CAB OFF", toggle);
  }

  void OnRescale() override
  {
    for (auto& image : mImages)
      image.second = GetUI()->GetScaledBitmap(image.second);
  }

private:
  bool Stacked() const { return mRECT.W() < 600.f && mRECT.H() > mRECT.W(); }
  IRECT MainArea() const
  {
    return (Stacked() ? mRECT.GetFromTop(mRECT.H() * 0.56f) : mRECT.SubRectHorizontal(2, 0)).GetPadded(-8.f);
  }
  IRECT Actions() const
  {
    const auto left = MainArea();
    const float height = Stacked() ? 44.f : std::min(44.f, left.H() * 0.21f);
    return left.GetFromBottom(height).GetCentredInside(std::min(186.f, left.W()), height);
  }

  IRECT Shelf() const
  {
    if (Stacked()) return mRECT.GetReducedFromTop(mRECT.H() * 0.56f).GetPadded(-8.f);
    const auto side = mRECT.SubRectHorizontal(2, 1).GetPadded(-10.f);
    return side.GetCentredInside(side.W(), side.H() * 0.55f);
  }

  IRECT ToggleBounds() const { return Actions().SubRectHorizontal(2, 0).GetReducedFromRight(3.f); }
  IRECT LoadBounds() const { return Actions().SubRectHorizontal(2, 1).GetReducedFromLeft(3.f); }

  IRECT Card(int slot) const
  {
    const int count = static_cast<int>(mChoices.size());
    const int columns = std::min(Stacked() ? 2 : 4, count);
    const auto area = Shelf().GetReducedFromTop(20.f);
    return area.SubRectVertical((count+columns-1)/columns,slot/columns)
               .SubRectHorizontal(columns,slot%columns).GetPadded(-2.f);
  }
  int ChoiceAt(float x,float y) const
  {
    for (size_t i = 0; i < mChoices.size(); ++i)
      if (Card(static_cast<int>(i)).Contains(x,y))
        return static_cast<int>(i);
    return -1;
  }
  void DrawShell(IGraphics& g, const std::string& name, const IRECT& bounds, const IBlend* blend) const
  {
    const auto found = std::find_if(mImages.begin(),mImages.end(),[&](const auto& image) {return image.first == name;});
    const auto& bitmap = (found == mImages.end() ? mImages.front() : *found).second;
    if (!bitmap.IsValid() || bounds.W() <= 0.f || bounds.H() <= 0.f)
      return;
    const float scale = std::min(bounds.W()/bitmap.W(), bounds.H()/bitmap.H());
    g.DrawFittedBitmap(bitmap,bounds.GetCentredInside(bitmap.W()*scale, bitmap.H()*scale),blend);
  }
  void Load(const char* path)
  {
    mStatus = "Loading IR";
    SetDirty(false);
    mLoadIR(path);
  }

  std::vector<IRChoice> mChoices;
  Images mImages;
  IRPathAction mLoadIR;
  std::string mName;
  std::string mStatus{"Select a cabinet"};
  int mSelected = -1;
};

class DessAmpEmptyStateControl : public IControl
{
public:
  explicit DessAmpEmptyStateControl(const IRECT& bounds) : IControl(bounds)
  { mIgnoreMouse = true; }
  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    g.FillRoundRect(Surface, mRECT, 8.f);
    g.DrawText(Text(18.f), "Select a head or choose CUSTOM to open a .dess", mRECT);
  }
};

class DessAmpIconButtonControl : public IControl
{
public:
  enum class Icon
  {
    Settings,
    Close,
    Tune
  };

  DessAmpIconButtonControl(const IRECT& bounds, Icon icon, IActionFunction action)
  : IControl(bounds, action)
  , mIcon(icon)
  {
  }

  void OnMouseDown(float x, float y, const IMouseMod& mod) override { SetDirty(true); }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    if (mMouseIsOver)
      g.FillRoundRect(COLOR_WHITE.WithOpacity(0.055f), mRECT, 6.f);
    const float cx = mRECT.MW(), cy = mRECT.MH();
    if (mIcon == Icon::Tune)
    {
      g.DrawText(Text(12.f, TextPrimary), "TUNE", mRECT);
      return;
    }
    if (mIcon == Icon::Close)
    {
      g.DrawLine(TextSecondary, cx - 7.f, cy - 7.f, cx + 7.f, cy + 7.f, nullptr, 2.f);
      g.DrawLine(TextSecondary, cx + 7.f, cy - 7.f, cx - 7.f, cy + 7.f, nullptr, 2.f);
      return;
    }
    g.DrawCircle(TextSecondary, cx, cy, 7.f, nullptr, 1.8f);
    g.DrawCircle(TextSecondary, cx, cy, 2.5f, nullptr, 1.8f);
    for (int i = 0; i < 8; ++i)
      g.DrawRadialLine(TextSecondary, cx, cy, static_cast<float>(i) * 45.f, 8.5f, 12.f, nullptr, 1.8f);
  }

private:
  Icon mIcon;
};

// One modal surface for editor actions. Labels are explicit on every platform.
class DessAmpChoicePanel : public IControl
{
public:
  explicit DessAmpChoicePanel(const IRECT& bounds) : IControl(bounds) {}
  using Action = std::function<void(int)>;
  void Present(std::string title, std::string message, std::array<std::string, 3> buttons, Action action)
  {
    mTitle = std::move(title); mMessage = std::move(message);
    mButtons = std::move(buttons); mAction = std::move(action);
    Hide(false); SetDirty(false);
  }
  void Dismiss() { mAction = {}; Hide(true); }
  void Choose(int index) { auto action = mAction; if (action) action(index); }
  void OnMouseDown(float x, float y, const IMouseMod&) override
  {
    for (int i = 0; i < 3; ++i)
      if (Button(i).Contains(x,y)) { Choose(i); return; }
  }
  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    g.FillRect(COLOR_BLACK.WithOpacity(0.79f), mRECT);
    const auto panel = Panel();
    g.FillRoundRect(SurfaceRaised, panel, 10.f); g.DrawRoundRect(Hairline, panel, 10.f);
    g.DrawText(Text(20.f), mTitle.c_str(), panel.GetHPadded(-8.f).GetFromTop(54.f));
    g.DrawText(Text(13.f, TextSecondary), mMessage.c_str(), panel.GetFromTop(84.f).GetFromBottom(30.f));
    for (int i = 0; i < 3; ++i)
    {
      const auto rect = Button(i);
      g.FillRoundRect(i == 2 ? Accent.WithOpacity(0.28f) : Surface,rect,4.f);
      g.DrawRoundRect(i == 2 ? Accent : Hairline,rect,4.f);
      g.DrawText(Text(12.f), mButtons[i].c_str(), rect);
    }
  }
private:
  IRECT Panel() const { return mRECT.GetCentredInside(std::min(418.f,mRECT.W()-16.f),160.f); }
  IRECT Button(int i) const
  {
    return Panel().GetHPadded(-8.f).GetReducedFromBottom(14.f).GetFromBottom(46.f)
           .SubRectHorizontal(3,i).GetHPadded(-2.f);
  }
  std::string mTitle, mMessage;
  std::array<std::string,3> mButtons;
  Action mAction;
};

class DessAmpTunerPanel : public IControl
{
public:
  DessAmpTunerPanel(const IRECT& bounds, IActionFunction close) : IControl(bounds, close) {}
  void Update(dessamp::GuitarTuner& tuner)
  {
    dessamp::GuitarTuner::Reading reading;
    const auto now = std::chrono::steady_clock::now();
    if (tuner.Read(reading))
    {
      mReading = reading;
      mLast = now;
      SetDirty(false);
    }
    else if (mReading.note >= 0 && now - mLast > std::chrono::milliseconds(400))
      Clear();
  }
  void Clear() { mReading = {}; SetDirty(false); }
  void OnMouseDown(float x, float y, const IMouseMod&) override
  {
    if (!Panel().Contains(x, y) || Close().Contains(x, y)) SetDirty(true);
  }
  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    const auto panel = Panel(), inner = panel.GetHPadded(-20.f);
    g.FillRect(COLOR_BLACK.WithOpacity(0.78f), mRECT);
    g.FillRoundRect(SurfaceRaised, panel, 12.f);
    g.DrawRoundRect(Hairline, panel, 12.f);
    g.DrawText(Text(16.f, TextPrimary, EAlign::Near), "TUNER", inner.GetFromTop(48.f));
    g.DrawText(Text(12.f, TextSecondary), "DONE", Close());
    const bool valid = mReading.note >= 0;
    const IColor signal = valid && std::abs(mReading.cents) <= 3 ? IColor(255, 110, 219, 167) : TextPrimary;
    const auto noteRow = IRECT(inner.L, panel.T + 46.f, inner.R, panel.T + panel.H() * 0.49f);
    std::string note;
    if (valid)
    {
      static const char* names[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
      note = std::string(names[mReading.note % 12]) + std::to_string(mReading.note / 12 - 1);
    }
    g.DrawText(Text(valid ? 54.f : 21.f, signal), valid ? note.c_str() : "Play one string", noteRow);
    const float cy = panel.T + panel.H()*0.59f;
    const float half = inner.W()*0.43f, cx = inner.MW();
    g.DrawLine(Hairline, cx-half, cy, cx+half, cy, nullptr, 2.f);
    for (int i = -5; i <= 5; ++i)
    {
      const float x = cx + i*half/5.f, h = i == 0 ? 10.f : 4.f;
      g.DrawLine(i == 0 ? Metal : Hairline, x, cy-h, x, cy+h);
    }
    if (valid) g.FillCircle(signal, cx + half * std::clamp(float(mReading.cents / 50), -1.f, 1.f), cy, 5.f);
    char value[64]{};
    if (valid)
    {
      const long cents = std::lround(mReading.cents);
      if (cents == 0) std::snprintf(value, sizeof(value), "In tune   0 cents");
      else std::snprintf(value, sizeof(value), "%s   %+ld %s",
        std::abs(mReading.cents) <= 3 ? "In tune" : cents < 0 ? "Flat" : "Sharp",
        cents, std::abs(cents) == 1 ? "cent" : "cents");
    }
    g.DrawText(Text(15.f, signal), value, IRECT(inner.L, cy+10.f, inner.R, cy+40.f));
    g.DrawText(Text(12.f, TextSecondary), "CLEAN INPUT     /     A4 = 440 Hz", inner.GetFromBottom(38.f));
  }
private:
  IRECT Panel() const {return mRECT.GetCentredInside(std::min(400.f, mRECT.W()-24.f), std::min(250.f, mRECT.H()-20.f));}
  IRECT Close() const { return Panel().GetFromTop(48.f).GetFromRight(62.f); }
  dessamp::GuitarTuner::Reading mReading;
  std::chrono::steady_clock::time_point mLast{};
};

class DessAmpSettingsPanelControl : public IControl
{
public:
  DessAmpSettingsPanelControl(const IRECT& bounds, std::function<void()> match, std::function<void()> reset,
                              std::function<void()> feedback)
  : IControl(bounds, {kCalibrateInput, kInputCalibrationLevel, kInputLevel}),
    mAutoInputAction(std::move(match)), mResetInputAction(std::move(reset)), mFeedbackAction(std::move(feedback))
  {
  }

  void SetModelInputReference(std::optional<double> reference, bool measured)
  {
    if (reference == mModelInputReference && measured == mModelInputReferenceMeasured) return;
    mModelInputReference = reference;
    mModelInputReferenceMeasured = measured;
    SetDirty(false);
  }

  void SetAudioStatus(double rate, int blockFrames, int latencyFrames)
  {
    if (rate == mHostRate && blockFrames == mHostBlock && latencyFrames == mPluginLatency)
      return;
    mHostRate = rate;
    mHostBlock = blockFrames;
    mPluginLatency = latencyFrames;
    SetDirty(false);
  }

  void SetAutoInputStatus(const std::string& status, bool active, bool matched)
  {
    // A reset can publish host parameters while this panel still holds the
    // touch. iPlug defers peer updates during capture. Reconcile on release.
    for (int i = 0; i < NVals(); ++i)
      SetValueFromDelegate(GetParam(i)->GetNormalized(), i);
    if (status == mAutoInputStatus && active == mAutoInputActive && matched == mInputMatched) return;
    mAutoInputStatus = status;
    mAutoInputActive = active;
    mInputMatched = matched;
    SetDirty(false);
  }

  bool OnKeyDown(float x, float y, const IKeyPress& key) override
  {
    if (key.VK != kVK_ESCAPE)
      return false;
    Hide(true);
    return true;
  }

  void OnMouseDown(float x, float y, const IMouseMod& mod) override
  {
    if (FeedbackBounds().Contains(x, y)) { mFeedbackAction(); return; }
    if (CloseBounds().Contains(x, y) || !PanelBounds().Contains(x, y))
    {
      Hide(true);
      return;
    }
    if (ResetInputBounds().Contains(x, y)) { mResetInputAction(); return; }
    if (AutoInputBounds().Contains(x, y))
    {
      mAutoInputAction();
      return;
    }

    if (ToggleBounds().Contains(x, y))
    {
      SetValueFromUserInput(GetValue(0) > 0.5 ? 0.0 : 1.0, 0);
      return;
    }
    if (MOTUInstrumentBounds().Contains(x, y))
    {
      SetInterfaceLevelDBU(16.0);
      SetValueFromUserInput(1.0, 0);
      return;
    }
    if (MOTUXLRBounds().Contains(x, y))
    {
      SetInterfaceLevelDBU(10.0);
      SetValueFromUserInput(1.0, 0);
      return;
    }
    if (MinusBounds().Contains(x, y))
    {
      AdjustInterfaceLevelDBU(mod.S ? -1.0 : -0.1);
      return;
    }
    if (PlusBounds().Contains(x, y))
    {
      AdjustInterfaceLevelDBU(mod.S ? 1.0 : 0.1);
      return;
    }
    if (ValueBounds().Contains(x, y))
      PromptUserInput(ValueBounds(), 1);
  }

  void OnMouseDblClick(float x, float y, const IMouseMod& mod) override
  {
    if (ValueBounds().Contains(x, y))
      PromptUserInput(ValueBounds(), 1);
  }

  void OnMouseWheel(float x, float y, const IMouseMod& mod, float d) override
  {
    if (ValueBounds().Contains(x, y))
      AdjustInterfaceLevelDBU((mod.S ? 1.0 : 0.1) * (d > 0.f ? 1.0 : -1.0));
  }

  void Draw(IGraphics& g) override
  {
    using namespace DessAmpUI;
    g.FillRect(COLOR_BLACK.WithOpacity(0.78f), mRECT);
    const auto panel = PanelBounds();
    const bool narrow = Narrow();
    const bool compact = !narrow && panel.H() < 370.f;
    const float left = panel.L + (narrow ? 16.f : 28.f);
    const float right = panel.R - (narrow ? 16.f : 28.f);
    const auto line = [&](float y) { return IRECT(left, y, right, y + 20.f); };
    g.FillRoundRect(SurfaceRaised, panel, 12.f);
    g.DrawRoundRect(Hairline, panel, 12.f);
    g.DrawText(Text(18.f, TextPrimary, EAlign::Near), "SETTINGS", line(panel.T + 20.f));
    DrawButton(g, FeedbackBounds(), "FEEDBACK", false);
    const auto close = CloseBounds();
    g.DrawLine(TextSecondary, close.MW()-8.f, close.MH()-8.f, close.MW()+8.f, close.MH()+8.f, nullptr, 2.f);
    g.DrawLine(TextSecondary, close.MW()+8.f, close.MH()-8.f, close.MW()-8.f, close.MH()+8.f, nullptr, 2.f);
    if (!narrow && mHostRate > 0.0)
    {
      std::ostringstream audio;
      audio << std::fixed << std::setprecision(1) << mHostRate / 1000.0 << " kHz / " << mHostBlock
            << " frames / Plugin delay " << std::setprecision(2) << 1000.0 * mPluginLatency / mHostRate << " ms";
      g.DrawText(Text(10.f, TextSecondary, EAlign::Far), audio.str().c_str(),
                 IRECT(left + 95.f, panel.T+18.f, FeedbackBounds().L-10.f, panel.T+44.f));
    }
    const bool enabled = GetValue(0) > 0.5;
    g.DrawText(Text(12.f, TextSecondary, EAlign::Near), "INPUT CALIBRATION", line(ToggleBounds().T-24.f));
    DrawButton(g, ToggleBounds(), enabled ? "CALIBRATION ON" : "CALIBRATION OFF", enabled);
    const float labelY = narrow ? MOTUInstrumentBounds().T-26.f : ToggleBounds().B+2.f;
    g.DrawText(Text(12.f, TextSecondary, EAlign::Near), "Interface maximum input level (dBu)", line(labelY));
    DrawButton(g, MOTUInstrumentBounds(), narrow ? "M2 INSTRUMENT  +16" : "M2 1/4\"  +16", false);
    DrawButton(g, MOTUXLRBounds(), "M2 XLR  +10", false);
    DrawButton(g, MinusBounds(), "-", false);
    DrawButton(g, PlusBounds(), "+", false);
    const auto formatDB = [](double db) {
      std::ostringstream value;
      value << std::showpos << std::fixed << std::setprecision(1) << db;
      return value.str();
    };
    DrawButton(g, ValueBounds(), (formatDB(InterfaceLevelDBU()) + " dBu").c_str(), false);

    const float referenceY = ValueBounds().B + (narrow ? 4.f : 7.f);
    const std::string reference = !mModelInputReference ? "Model reference: waiting for active amp"
      : "Model reference: " + formatDB(*mModelInputReference) +
        (mModelInputReferenceMeasured ? " dBu (measured)" : " dBu (fallback, unverified)");
    g.DrawText(Text(11.f, TextSecondary, EAlign::Near), reference.c_str(), line(referenceY));

    const double trim = GetParam(2)->FromNormalized(GetValue(2));
    const double correction = enabled && mModelInputReference ? InterfaceLevelDBU() - *mModelInputReference : 0.0;
    const std::string gain = "Input trim " + formatDB(trim) + " dB    |    Total gain " + formatDB(trim + correction) + " dB";
    g.DrawText(Text(13.f, TextPrimary, EAlign::Near), gain.c_str(), line(referenceY + 21.f));
    const auto match = AutoInputBounds();
    const float headingY = match.T - (compact ? 27.f : 51.f);
    g.DrawLine(Hairline, left, headingY-4.f, right, headingY-4.f);
    g.DrawText(Text(12.f, mInputMatched ? TextPrimary : TextSecondary, EAlign::Near),
               mInputMatched ? "INPUT MATCHED" : "MATCH INPUT LEVEL", line(headingY));
    g.DrawText(Text(12.f, TextSecondary, compact ? EAlign::Far : EAlign::Near),
               "Play strong chords for 5 seconds. Target: -12 dBFS.", line(compact ? headingY : match.T-28.f));
    DrawButton(g, match, mAutoInputActive ? "CANCEL MATCH" : mInputMatched ? "MATCH AGAIN" : "MATCH INPUT",
               mAutoInputActive || mInputMatched);
    DrawButton(g, ResetInputBounds(), "RESET INPUT", false);
    g.DrawText(Text(12.f, TextPrimary, EAlign::Near), mAutoInputStatus.c_str(), line(match.B+3.f));
    g.DrawText(Text(11.f, TextSecondary, compact ? EAlign::Far : EAlign::Near),
               compact ? "Reset: calibration off, trim 0 dB." : "Reset turns calibration off and sets Input to 0 dB.",
               line(match.B + (compact ? 3.f : 24.f)));
  }

private:
  bool Narrow() const { return mRECT.W() < 600.f && mRECT.H() > mRECT.W(); }
  IRECT NarrowRow(float top) const { return PanelBounds().GetHPadded(-16.f).GetFromTop(44.f).GetTranslated(0.f, top); }
  IRECT AutoInputBounds() const
  {
    if (Narrow()) return NarrowRow(388.f).SubRectHorizontal(2, 0).GetReducedFromRight(3.f);
    const auto panel = PanelBounds();
    const float top = panel.B - (panel.H() < 370.f ? 65.f : 82.f);
    return IRECT(panel.L+28.f, top, panel.L+173.f, top+38.f);
  }
  IRECT ResetInputBounds() const
  {
    if (Narrow()) return NarrowRow(388.f).SubRectHorizontal(2, 1).GetReducedFromLeft(3.f);
    const auto match = AutoInputBounds();
    return match.GetTranslated(match.W()+8.f, 0.f);
  }
  IRECT FeedbackBounds() const { return CloseBounds().GetTranslated(-78.f, 0.f).GetFromLeft(48.f).GetHPadded(10.f); }
  std::function<void()> mAutoInputAction;
  std::function<void()> mResetInputAction;
  std::function<void()> mFeedbackAction;
  std::string mAutoInputStatus;
  bool mInputMatched = false;
  bool mAutoInputActive = false;
  void DrawButton(IGraphics& g, const IRECT& bounds, const char* label, bool active)
  {
    using namespace DessAmpUI;
    g.FillRoundRect(active ? Accent.WithOpacity(0.28f) : Surface, bounds, 5.f);
    g.DrawRoundRect(active ? Accent : Hairline, bounds, 5.f, nullptr, active ? 1.5f : 1.f);
    g.DrawText(Text(bounds.H() < 32.f ? 10.f : 11.f, active ? COLOR_WHITE : TextPrimary, EAlign::Center), label,
               bounds);
  }

  double InterfaceLevelDBU() const
  {
    const auto* param = GetParam(1);
    return param == nullptr ? 0.0 : param->FromNormalized(GetValue(1));
  }

  void SetInterfaceLevelDBU(double levelDBU)
  {
    const auto* param = GetParam(1);
    if (param == nullptr)
      return;
    levelDBU = std::clamp(levelDBU, param->GetMin(), param->GetMax());
    SetValueFromUserInput(param->ToNormalized(levelDBU), 1);
  }

  void AdjustInterfaceLevelDBU(double deltaDB)
  {
    SetInterfaceLevelDBU(InterfaceLevelDBU() + deltaDB);
  }

  IRECT PanelBounds() const
  {
    const float width = std::min(760.f, mRECT.W() - 40.f);
    const float height = std::min(Narrow() ? 494.f : 398.f, mRECT.H() - 32.f);
    return IRECT(
      mRECT.MW() - width * 0.5f, mRECT.MH() - height * 0.5f, mRECT.MW() + width * 0.5f, mRECT.MH() + height * 0.5f);
  }
  IRECT CloseBounds() const
  {
    const auto panel = PanelBounds();
    return IRECT(panel.R - 64.f, panel.T + 16.f, panel.R - 16.f, panel.T + 64.f);
  }

  IRECT ToggleBounds() const
  {
    if (Narrow()) return NarrowRow(93.f);
    const auto panel = PanelBounds();
    const float top = panel.T + (panel.H() < 370.f ? 63.f : 88.f);
    return IRECT(panel.L + 28.f, top, panel.L + 173.f, top+36.f);
  }

  IRECT MOTUInstrumentBounds() const
  {
    if (Narrow()) return NarrowRow(180.f).SubRectHorizontal(2, 0).GetReducedFromRight(3.f);
    const auto panel = PanelBounds();
    const float top = panel.T + (panel.H() < 370.f ? 120.f : 146.f);
    return IRECT(panel.L+28.f, top, panel.L+155.f, top+34.f);
  }

  IRECT MOTUXLRBounds() const
  {
    if (Narrow()) return NarrowRow(180.f).SubRectHorizontal(2, 1).GetReducedFromLeft(3.f);
    const auto instrument = MOTUInstrumentBounds();
    return IRECT(instrument.R + 8.f, instrument.T, instrument.R + 113.f, instrument.B);
  }

  IRECT MinusBounds() const
  {
    if (Narrow()) return NarrowRow(233.f).GetFromLeft(44.f);
    const auto xlr = MOTUXLRBounds();
    return IRECT(xlr.R + 10.f, xlr.T, xlr.R + 34.f, xlr.B);
  }

  IRECT ValueBounds() const
  {
    if (Narrow()) return NarrowRow(233.f).GetHPadded(-50.f);
    const auto minus = MinusBounds();
    const auto panel = PanelBounds();
    return IRECT(minus.R + 5.f, minus.T, panel.R - 61.f, minus.B);
  }

  IRECT PlusBounds() const
  {
    if (Narrow()) return NarrowRow(233.f).GetFromRight(44.f);
    const auto value = ValueBounds();
    return IRECT(value.R + 5.f, value.T, value.R + 24.f, value.B);
  }

  bool mModelInputReferenceMeasured = false;
  std::optional<double> mModelInputReference;
  double mHostRate = 0.0;
  int mHostBlock = 0;
  int mPluginLatency = 0;
};
