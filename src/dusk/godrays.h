#pragma once

// Light shafts: when the sun is in front of the camera, sunlight streams past whatever
// covers the sky (trees, cliffs, buildings). Screen-space radial light scattering on the
// depth buffer, added after the translucent scene.
namespace dusk::godrays {

void apply();

}  // namespace dusk::godrays
