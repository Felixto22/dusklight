#pragma once

// Wind in the trees: while the stage geometry is drawn, alpha-tested leaves, bushes and
// vines sway with the game's own wind (direction and strength), with slow gusts and a
// quick leaf flutter. Done in the vertex shader (aurora patch, GX_AURORA_FOLIAGE_WIND).
namespace dusk::foliage {

void begin();
void end();

}  // namespace dusk::foliage
