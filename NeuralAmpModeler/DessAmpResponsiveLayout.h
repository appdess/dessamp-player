#pragma once

#include <algorithm>
#include <array>
#include "IGraphicsStructs.h"

namespace DessAmpUI
{
using iplug::igraphics::IRECT;

// All rectangles use host view points, including embedded AUv3 views.
struct ResponsiveLayout
{
  explicit ResponsiveLayout(const IRECT& bounds)
  : b(bounds)
  , phone(std::min(bounds.W(), bounds.H()) < 600.f || bounds.W() > 1.75f * bounds.H())
  , portraitPhone(phone && bounds.H() > bounds.W())
  {
    const float w = b.W();
    const float h = b.H();
    const bool tabletPortrait = !phone && h > w;
    const bool tabletLandscape = !phone && w > h && w <= 1600.f && h <= 1200.f;
    if (phone)
    {
      LayoutPhone();
      return;
    }
    const float side = tabletPortrait ? std::max(24.f, w * 0.027f) : std::max(18.f, w * 0.01875f);
    header = tabletPortrait ? IRECT(side, 18.f, w - side, 78.f) : IRECT(side, 8.f, w - side, 60.f);
    utility = tabletPortrait ? IRECT(side + 8.f, 94.f, w - side - 8.f, 318.f)
                             : IRECT(side, 66.f, w - side, 170.f);
    const float headWidth = w - 2.f * (side + 8.f);
    const float headHeight = std::min(headWidth / 2.19f, tabletPortrait ? h - 278.f : h - 240.f);
    footer = IRECT(side + 8.f, h - 42.f, w - side - 8.f, h - 10.f);
    const float headTop = tabletPortrait || tabletLandscape
                            ? utility.B + (footer.T - utility.B - headHeight) * 0.5f
                            : utility.B + 8.f;
    head = IRECT(side + 8.f, headTop, w - side - 8.f, headTop + headHeight);

    wordmark = IRECT(header.L + 8.f, header.T, header.L + 220.f, header.B);
    tabs = tabletPortrait ? IRECT(w * 0.5f - 70.f, header.T, w * 0.5f + 70.f, header.B)
                          : IRECT(header.L + 236.f, header.T, header.L + 412.f, header.B);
    // Landscape tablet headers are 52 points high. A four-point vertical
    // inset preserves the required 44-point touch target; the previous
    // eight-point top inset reduced both primary controls to 40 points.
    settings = IRECT(header.R - 100.f, header.T + 4.f, header.R - 52.f, header.B - 4.f);
    tuner = settings.GetTranslated(-54.f, 0.f);
    power = IRECT(header.R - 48.f, header.T + 4.f, header.R, header.B - 4.f);

    const float innerL = utility.L + 14.f;
    const float innerR = utility.R - 14.f;
    const float topRowBottom = tabletPortrait ? utility.T + 112.f : utility.B - 8.f;
    inputMeter = IRECT(innerL, utility.T + 24.f, innerL + 8.f, topRowBottom - 12.f);
    input = IRECT(innerL + 12.f, utility.T + 10.f, innerL + 100.f, topRowBottom);
    gate = IRECT(innerL + 108.f, utility.T + 10.f, innerL + 192.f, topRowBottom);
    gateToggle = IRECT(innerL + 190.f, utility.T + 16.f, innerL + 250.f, topRowBottom - 6.f);
    model = tabletPortrait ? IRECT(innerL + 266.f, utility.T + 10.f, innerR - 110.f, topRowBottom)
                           : IRECT(innerL + 266.f, utility.T + 10.f, w * 0.54f, topRowBottom);
    output = IRECT(innerR - 96.f, utility.T + 10.f, innerR - 8.f, topRowBottom);
    outputMeter = IRECT(innerR - 4.f, utility.T + 24.f, innerR + 4.f, topRowBottom - 12.f);

    constexpr std::array<float, 6> controlFractions{0.129f, 0.277f, 0.426f, 0.574f, 0.723f, 0.871f};
    const float controlTop = head.T + head.H() * 0.63f;
    for (std::size_t i = 0; i < controlFractions.size(); ++i)
    {
      const float cx = head.L + head.W() * controlFractions[i];
      conditions[i] = IRECT(cx - 46.f, controlTop, cx + 46.f, head.B - 24.f);
    }
    for (std::size_t i = controlFractions.size(); i < conditions.size(); ++i)
      conditions[i] = IRECT(-200.f, -200.f, -108.f, -70.f);
    bright = SwitchAt(head.L + head.W() * 0.205f, controlTop + 8.f);
    bottom = SwitchAt(head.L + head.W() * 0.275f, controlTop + 8.f);
    midBoost = SwitchAt(head.L + head.W() * 0.785f, controlTop + 8.f);
    empty = IRECT(w * 0.5f - 250.f, controlTop - 4.f, w * 0.5f + 250.f, head.B - 28.f);

    const float stageTop = utility.B + 18.f;
    const float stageBottom = footer.T - 8.f;
    preStage = IRECT(0.f, stageTop, w, stageBottom);
    model.R = output.L - 8.f;
    const float stripW = std::min(420.f, footer.W());
    factoryStrip = IRECT(footer.MW()-stripW/2, footer.B-44.f, footer.MW()+stripW/2, footer.B);
    pedals = IRECT(utility.L, stageTop, utility.R, stageBottom);
  }

  IRECT b;
  bool phone = false;
  bool portraitPhone = false;
  IRECT controls, levels, quickPreset;
  IRECT header, utility, head, footer, factoryStrip, pedals;
  IRECT wordmark, tabs, tuner, settings, power;
  IRECT inputMeter, input, gate, gateToggle, model, output, outputMeter;
  std::array<IRECT, 8> conditions{};
  IRECT bright, bottom, midBoost, empty;
  IRECT preStage;

  IRECT PhoneCondition(size_t count, size_t index, bool fireFace = true) const
  {
    if (!portraitPhone)
    {
      // The bottom panel is part of the head artwork. Leave room for the value above its trim.
      const float left = head.L + head.W() * 0.07f;
      const float width = head.W() * 0.86f;
      const float cell = width / static_cast<float>(std::max<size_t>(count, 1));
      const float cx = left + (static_cast<float>(index) + 0.5f) * cell;
      const float half = std::min(50.f, cell * 0.5f - 0.5f);
      return IRECT(cx - half, head.T + head.H() * (fireFace ? 0.65f : 0.62f),
                   cx + half, head.T + head.H() * (fireFace ? 0.90f : 0.86f));
    }
    const size_t columns = std::min<size_t>(count > 6 ? 3 : 2, std::max<size_t>(count, 1));
    const size_t rows = (count + columns - 1) / columns;
    const size_t row = index / columns;
    const size_t inRow = std::min(columns, count - row * columns);
    const float width = std::min(156.f, (controls.W() - 16.f) / static_cast<float>(columns));
    const float pitch = std::min(122.f, (controls.H() - 32.f) / static_cast<float>(std::max<size_t>(rows, 1)));
    const float height = std::min(112.f, pitch);
    const float x = controls.MW() - width * static_cast<float>(inRow) * 0.5f + width * static_cast<float>(index % columns);
    const float start = controls.T + 27.f + (controls.H() - 32.f - pitch * rows) * 0.5f;
    const float y = start + (pitch - height) * 0.5f + pitch * row;
    return IRECT(x + 2.f, y, x + width - 2.f, y + height - 2.f);
  }

private:
  void LayoutPhone()
  {
    const float w = b.W(), h = b.H();
    const float side = portraitPhone ? 14.f : std::max(44.f, w * 0.055f);
    const float top = portraitPhone ? (h >= 720.f ? 56.f : 22.f) : 4.f;
    header = IRECT(side, top, w - side, top + (portraitPhone ? 90.f : 46.f));
    const auto firstRow = header.GetFromTop(46.f);
    power = firstRow.GetFromRight(44.f);
    settings = power.GetTranslated(-46.f, 0.f);
    tuner = settings.GetTranslated(-46.f, 0.f);
    tabs = portraitPhone ? header.GetFromBottom(44.f) : firstRow.GetCentredInside(188.f, 44.f);
    wordmark = IRECT(header.L + 8.f, firstRow.T, portraitPhone ? tuner.L - 6.f : tabs.L - 6.f, firstRow.B);

    const float footerEnd = h - (portraitPhone ? 25.f : 21.f);
    const float stripHeight = portraitPhone ? 60.f : (h >= 375.f ? 50.f : 44.f);
    footer = IRECT(side, footerEnd - stripHeight, w - side, footerEnd);
    factoryStrip = footer.GetCentredInside(std::min(500.f, footer.W()), stripHeight);
    float stageEnd = footer.T - 8.f;
    if (portraitPhone)
    {
      quickPreset = IRECT(side, footer.T - 48.f, w - side, footer.T - 4.f);
      stageEnd = quickPreset.T - 6.f;
    }
    else
    {
      const float split = footer.L + footer.W() * 0.41f;
      quickPreset = IRECT(footer.L, footer.T, split - 4.f, footer.B);
      factoryStrip = IRECT(split + 4.f, footer.T, footer.R, footer.B);
    }
    const auto stage = IRECT(side, header.B + 8.f, w - side, stageEnd);
    pedals = preStage = stage;
    if (portraitPhone)
    {
      // Fit the whole head, including its handle, lower faceplate and feet.
      // Keep room for the control grid in a short AUv3 host view as well.
      const float headHeight = std::min(stage.W() / 2.19f, stage.H() * 0.40f);
      head = stage.GetFromTop(headHeight).GetCentredInside(headHeight * 2.19f, headHeight);
      controls = IRECT(stage.L, head.B + 8.f, stage.R, stage.B);
    }
    else
    {
      const float height = std::min(stage.H(), stage.W() / 2.25f);
      head = stage.GetCentredInside(height * 2.25f, height);
      controls = head.GetFromBottom(head.H() * 0.35f);
    }

    // RIG owns signal levels and presets on phones. Both sections have their own space.
    const float rigWidth = std::min(stage.W(), 540.f);
    const float rigHeight = std::min(stage.H(), 244.f);
    utility = stage.GetCentredInside(rigWidth, rigHeight);
    levels = utility.GetPadded(-8.f).GetFromTop(std::min(120.f, rigHeight * 0.49f));
    auto levelRow = levels.GetReducedFromTop(20.f);
    input = levelRow.SubRectHorizontal(4, 0);
    gate = levelRow.SubRectHorizontal(4, 1);
    gateToggle = levelRow.SubRectHorizontal(4, 2);
    output = levelRow.SubRectHorizontal(4, 3);
    inputMeter = input.GetFromLeft(3.f).GetVPadded(-10.f);
    outputMeter = output.GetFromRight(3.f).GetVPadded(-10.f);
    model = IRECT(utility.L + 10.f, levels.B + 4.f, utility.R - 10.f, utility.B - 8.f);
    for (size_t i = 0; i < conditions.size(); ++i)
      conditions[i] = PhoneCondition(conditions.size(), i);
    bright = bottom = midBoost = controls;
    empty = controls;
  }

  static IRECT SwitchAt(float cx, float top) { return IRECT(cx - 28.f, top, cx + 28.f, top + 82.f); }
};
} // namespace DessAmpUI
