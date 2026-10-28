#pragma once

#include <dolphin/types.h>

class J3DModel;
class dKy_tevstr_c;

// Epona's hair and coat:
//  - spring physics on the mane flaps, forelock, tail chain, girth straps and stirrups,
//    driven by how each joint is actually accelerated (gallop bounce, starts, stops,
//    turns, jumps) plus the game's wind;
//  - strands of hair hanging from the mane and tail, each with its own verlet physics;
//  - shell fur on the body: the body is drawn again in thin inflated layers carrying a
//    hair pattern, so the coat catches the light like short hair.
// Only daHorse_c (Epona's hs.bmd) feeds this; the data it relies on is generated from
// that model (epona_data.inc) and everything turns itself off if the model differs.
namespace dusk::epona {

// From daHorse_c::modelCallBack after the game has posed `jointNo`.
void joint_callback(J3DModel* model, int jointNo);

// From daHorse_c::draw after the model has been queued.
void draw(J3DModel* model, dKy_tevstr_c* tevstr);

}  // namespace dusk::epona
