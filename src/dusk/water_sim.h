#pragma once

#include <dolphin/types.h>

class cXyz;

// Dynamic water surface: a small wave-equation simulation that follows the player and
// lights the original water surfaces with the waves that walking, swimming, riding,
// enemies and objects make. Every contact with water already reaches
// dPa_control_c::setWaterRipple (wading/swimming/moving) or fopKyM_createWpillar
// (entering/falling in), so those two feed it.
namespace dusk::water {

// Continuous contact (an actor moving through or standing in water). `speed` is the
// value the game passes to setWaterRipple; it grows with how fast the actor moves.
void ripple(const cXyz& pos, f32 speed);

// A splash: something entered or fell into the water. `scale` is the water pillar's scale.
void splash(const cXyz& pos, f32 scale);

// Called once per game frame (from the weather move pass).
void update();

// Called once per frame from the weather draw pass; queues the surface packet.
void draw();

// Called when the stage changes so nothing carries over between maps.
void reset();

// Hide the game's flat ripple-ring effect while the dynamic surface is active.
bool hide_flat_ripples();

}  // namespace dusk::water
