#pragma once

class J3DModel;

// Real hair for Link: locks of fine hair laid over his hair mesh, following the way it is
// combed (from the crown down over the bangs and the back), coloured by the original hair
// texture and material, skinned to the same joints (so the game's own hair swing drives
// them) and with their own verlet physics on top: inertia, a little gravity and wind.
// Works for the three head models with hair (Ordon clothes, Hero's clothes, Magic Armor);
// anything else (Zora armour, wolf) is left alone.
namespace dusk::linkhair {

// From daAlink_c::draw right after Link's head (hat) model is queued.
void draw(J3DModel* headModel);

}  // namespace dusk::linkhair
