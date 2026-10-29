#pragma once

#include <dolphin/mtx.h>

// Screen-space ambient occlusion: after the opaque scene is drawn, the depth buffer is
// read back and every pixel is darkened by how much nearby geometry surrounds it
// (corners, creases, under trees, where things meet the ground). Translucent geometry
// (water, effects) and the HUD are drawn afterwards, unaffected.
namespace dusk::ssao {

// From mDoGph_Painter right after the opaque lists, with the camera's projection.
void apply(const Mtx44 proj);

}  // namespace dusk::ssao
