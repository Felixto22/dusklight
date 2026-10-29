#pragma once

class J3DModel;

// Fur for Link's wolf form: shell fur over the wolf body (wl.bmd), lit and coloured like
// the original body material. Length set by "Wolf Fur Length".
namespace dusk::wolffur {

// From daAlink_c::draw right after the wolf model is queued.
void draw(J3DModel* wolfModel);

}  // namespace dusk::wolffur
