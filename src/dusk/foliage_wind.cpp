// Wind in the trees (see foliage_wind.h).

#include "dusk/foliage_wind.h"

#include "JSystem/J3DGraphBase/J3DSys.h"
#include "d/d_kankyo.h"
#include "d/d_kankyo_wether.h"
#include "dusk/settings.h"

#include <dolphin/gx/GXAurora.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace dusk::foliage {
namespace {

bool g_active = false;
f32 g_dirX = 1.0f, g_dirZ = 0.0f, g_pow = 0.0f;

f32 seconds() {
    using namespace std::chrono;
    static const auto t0 = steady_clock::now();
    return duration<f32>(steady_clock::now() - t0).count();
}

}  // namespace

void begin() {
    const auto& game = dusk::getSettings().game;
    if (!game.foliageWind.getValue()) return;
    const f32 t = seconds();

    // The game's wind, eased so a change of direction doesn't snap the trees.
    f32 wx = 1.0f, wz = 0.0f;
    if (cXyz* wv = dKyw_get_wind_vec()) {
        const f32 l = std::sqrt(wv->x * wv->x + wv->z * wv->z);
        if (l > 1e-3f) {
            wx = wv->x / l;
            wz = wv->z / l;
        }
    }
    g_dirX += (wx - g_dirX) * 0.02f;
    g_dirZ += (wz - g_dirZ) * 0.02f;
    const f32 dl = std::sqrt(g_dirX * g_dirX + g_dirZ * g_dirZ);
    if (dl > 1e-3f) {
        g_dirX /= dl;
        g_dirZ /= dl;
    }
    g_pow += (std::clamp(dKyw_get_wind_pow(), 0.0f, 1.0f) - g_pow) * 0.02f;
    // A light breeze even on calm days, with slow gusts on top.
    const f32 gust = 0.75f + 0.25f * std::sin(t * 0.37f) * std::sin(t * 0.23f + 1.3f);
    const f32 strength = std::clamp((0.18f + 0.82f * g_pow) * gust *
                                        (std::clamp(game.foliageWindStrength.getValue(), 0, 300) / 100.0f),
                                    0.0f, 2.0f);

    MtxP v = j3dSys.getViewMtx();
    f32 p[20];
    // Wind direction and world up, in view space.
    p[0] = v[0][0] * g_dirX + v[0][2] * g_dirZ;
    p[1] = v[1][0] * g_dirX + v[1][2] * g_dirZ;
    p[2] = v[2][0] * g_dirX + v[2][2] * g_dirZ;
    p[3] = strength;
    p[4] = v[0][1];
    p[5] = v[1][1];
    p[6] = v[2][1];
    p[7] = t;
    // Inverse view (rotation transposed, translation brought back) for world positions.
    for (int r = 0; r < 3; r++) {
        p[8 + r * 4 + 0] = v[0][r];
        p[8 + r * 4 + 1] = v[1][r];
        p[8 + r * 4 + 2] = v[2][r];
        p[8 + r * 4 + 3] = -(v[0][r] * v[0][3] + v[1][r] * v[1][3] + v[2][r] * v[2][3]);
    }
    GXSetFoliageWindAurora(GX_TRUE, p);
    g_active = true;
}

void end() {
    if (!g_active) return;
    GXSetFoliageWindAurora(GX_FALSE, nullptr);
    g_active = false;
}

namespace {
bool g_reliefActive = false;
f32 g_wetness = 0.0f;
f32 g_lastT = -1.0f;

// Surfaces get wet while it rains (about 15 s to soak) and dry slowly (about 90 s).
f32 update_wetness() {
    const f32 t = seconds();
    const f32 dt = g_lastT < 0.0f ? 0.0f : std::clamp(t - g_lastT, 0.0f, 0.5f);
    g_lastT = t;
    const f32 target = std::clamp(g_env_light.raincnt / 200.0f, 0.0f, 1.0f);
    if (target > g_wetness) {
        g_wetness = std::min(target, g_wetness + dt / 15.0f);
    } else {
        g_wetness = std::max(target, g_wetness - dt / 90.0f);
    }
    return g_wetness;
}
}

void begin_relief() {
    const auto& game = dusk::getSettings().game;
    u32 flags = 0;
    if (game.surfaceNormals.getValue()) flags |= 1;
    if (game.surfaceParallax.getValue()) flags |= 2;
    const f32 wet = update_wetness();
    if (game.wetSurfaces.getValue() && wet > 0.01f) flags |= 4;
    if (flags == 0) return;
    MtxP v = j3dSys.getViewMtx();
    // Sun (or moon) direction from the camera, in view space.
    const f32 ex = -(v[0][0] * v[0][3] + v[1][0] * v[1][3] + v[2][0] * v[2][3]);
    const f32 ey = -(v[0][1] * v[0][3] + v[1][1] * v[1][3] + v[2][1] * v[2][3]);
    const f32 ez = -(v[0][2] * v[0][3] + v[1][2] * v[1][3] + v[2][2] * v[2][3]);
    dScnKy_env_light_c* env = dKy_getEnvlight();
    f32 dx = env->sun_pos.x - ex, dy = env->sun_pos.y - ey, dz = env->sun_pos.z - ez;
    f32 l = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (l < 1.0f || dy < 0.1f * l) {
        dx = 0.3f;
        dy = 1.0f;
        dz = 0.2f;
        l = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    dx /= l;
    dy /= l;
    dz /= l;
    f32 p[16] = {};
    p[0] = v[0][0] * dx + v[0][1] * dy + v[0][2] * dz;
    p[1] = v[1][0] * dx + v[1][1] * dy + v[1][2] * dz;
    p[2] = v[2][0] * dx + v[2][1] * dy + v[2][2] * dz;
    p[3] = std::clamp(game.surfaceNormalsStrength.getValue(), 0, 300) / 100.0f;
    p[4] = 0.035f * std::clamp(game.surfaceParallaxStrength.getValue(), 0, 300) / 100.0f;
    p[5] = wet;
    p[6] = seconds();
    // World up in view space, and the sky colour wet ground reflects.
    p[8] = v[0][1];
    p[9] = v[1][1];
    p[10] = v[2][1];
    const GXColorS10 sky = env->vrbox_sky_col;
    p[12] = std::clamp(sky.r / 255.0f, 0.0f, 1.0f);
    p[13] = std::clamp(sky.g / 255.0f, 0.0f, 1.0f);
    p[14] = std::clamp(sky.b / 255.0f, 0.0f, 1.0f);
    GXSetReliefAurora(flags, p);
    g_reliefActive = true;
}

void end_relief() {
    if (!g_reliefActive) return;
    GXSetReliefAurora(0, nullptr);
    g_reliefActive = false;
}

}  // namespace dusk::foliage
