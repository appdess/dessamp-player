#include "../DessAmpResponsiveLayout.h"
#include <array>
#include <cassert>
#include <iostream>

using DessAmpUI::ResponsiveLayout;
using DessAmpUI::IRECT;

int main()
{
  for (const auto [w,h] : std::array<std::pair<int,int>,4>{{{320,568},{375,667},{402,874},{440,956}}})
    for (const bool portrait : {false, true})
    {
      const ResponsiveLayout ui(IRECT(0, 0, portrait ? w : h, portrait ? h : w));
      assert(ui.phone && ui.portraitPhone == portrait);
      assert(ui.tabs.H() >= 44 && ui.tabs.W()/4 >= 44);
      assert(ui.tuner.W() >= 44 && ui.tuner.H() >= 44 && ui.header.Contains(ui.tuner));
      for (const auto& other : {ui.settings, ui.tabs, ui.wordmark})
        assert(ui.tuner.R <= other.L || ui.tuner.L >= other.R ||
               ui.tuner.B <= other.T || ui.tuner.T >= other.B);
      assert(ui.factoryStrip.H() >= (portrait ? 60 : 44));
      if (portrait) assert(std::abs(ui.head.W()/ui.head.H() - 2.19f) < 0.001f);
      assert(ui.b.Contains(ui.head) && ui.head.T >= ui.header.B && ui.head.B <= ui.footer.T);
      assert(ui.b.Contains(ui.utility) && ui.utility.Contains(ui.model));
      assert(ui.b.Contains(ui.quickPreset) && ui.quickPreset.H() >= 44);
      assert(ui.quickPreset.W() >= 138 && !ui.quickPreset.Intersects(ui.factoryStrip));
      assert(!ui.quickPreset.Intersects(ui.head));
      for (std::size_t count = 1; count <= 8; ++count)
        for (const bool fire : {false, true})
          for (std::size_t i = 0; i < count; ++i)
          {
            const auto r = ui.PhoneCondition(count, i, fire);
            assert((portrait ? ui.controls : ui.head).Contains(r));
            assert(r.W() >= 44 && r.H() >= 44);
            if (portrait) assert(r.T >= ui.head.B);
            else assert(r.B <= ui.head.T + ui.head.H() * (fire ? 0.9f : 0.86f));
            for (std::size_t j = 0; j < i; ++j)
              assert(!r.Intersects(ui.PhoneCondition(count, j, fire)));
          }
    }
  for (const auto [w,h] : std::array<std::pair<int,int>,3>{{{768,1024},{1024,768},{1280,800}}})
  {
    const ResponsiveLayout ui(IRECT(0,0,w,h));
    assert(!ui.phone && ui.utility.B <= ui.head.T && ui.head.B <= ui.footer.T);
  }
  std::cout << "Phone and tablet control layout checks passed\n";
}
