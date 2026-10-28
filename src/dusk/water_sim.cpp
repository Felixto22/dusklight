// Dynamic water surface (see water_sim.h).
//
// Three nested layers, all drawn as transparent sheets riding on the stage's own water:
//  - fine:    a 160x160 damped wave-equation height field (9-unit cells, ~14 m) around the
//             player, fed by every contact the game reports (setWaterRipple/createWpillar);
//  - coarse:  the same solver at 24-unit cells (~30 m), for contacts farther away
//             (enemies, Epona at a distance, things falling in);
//  - ambient: wind-driven swell over all the water within ~48 m, from the game's wind.
// A water map sampled from the collision every 2 m gives the real surface height under
// every sheet (rivers slope), and each sim keeps its own finer land/water mask so waves
// reflect off shores, rocks and pillars. Each sheet draws only what the waves change on a
// flat surface: light on the slope facing the sun and shadow behind it, crests brighter
// than troughs, the sky reflection and sun glint, foam on steep or tall crests. Inner
// layers fade out at their edges and the outer layers fade in under them.

#include "dusk/water_sim.h"

#include "JSystem/J3DGraphBase/J3DDrawBuffer.h"
#include "JSystem/J3DGraphBase/J3DPacket.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DSys.h"
#include "d/d_bg_s_gnd_chk.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "d/d_kankyo_wether.h"
#include "f_op/f_op_camera_mng.h"
#include "dusk/settings.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dusk::water {
namespace {

constexpr int SUBSTEPS = 2;        // per game frame (30 Hz logic)
constexpr f32 SAME_WATER = 120.0f; // contacts farther than this from a sim's plane are other water
constexpr int IDLE_FRAMES = 240;   // frames without contact before a sim may sleep
constexpr f32 NO_WATER = -1.0e20f;
constexpr f32 UNKNOWN = -2.0e20f;

enum MaskState : u8 { MASK_UNKNOWN = 0, MASK_WATER = 1, MASK_LAND = 2 };

bool enabled() {
    return dusk::getSettings().game.dynamicWater.getValue();
}

u32 g_frame = 0;

// ---------------------------------------------------------------------------------------
// Water map: the water surface height every 2 m around the player, from the collision.

struct WaterMap {
    static constexpr int M = 48;
    static constexpr f32 S = 200.0f;
    std::vector<f32> y = std::vector<f32>(M * M, UNKNOWN);
    f32 originX = 0.0f, originZ = 0.0f;
    bool placed = false;
    int cursor = 0;
    f32 refY = 0.0f;  // height of the player, to look for water around it

    const std::vector<int>& order() {
        static std::vector<int> o = [] {
            std::vector<int> v(M * M);
            for (int i = 0; i < M * M; i++) v[i] = i;
            auto d = [](int i) {
                const f32 x = (i % M) - (M - 1) * 0.5f, z = (i / M) - (M - 1) * 0.5f;
                return x * x + z * z;
            };
            std::stable_sort(v.begin(), v.end(), [&](int a, int b) { return d(a) < d(b); });
            return v;
        }();
        return o;
    }

    void clear() {
        std::fill(y.begin(), y.end(), UNKNOWN);
        cursor = 0;
        placed = false;
    }

    void follow(f32 x, f32 z) {
        if (!placed) {
            originX = std::floor(x / S) * S - (M / 2) * S;
            originZ = std::floor(z / S) * S - (M / 2) * S;
            placed = true;
            return;
        }
        const f32 cx = originX + (M / 2) * S, cz = originZ + (M / 2) * S;
        const int bx = int(std::round((x - cx) / S)), bz = int(std::round((z - cz) / S));
        if (std::abs(bx) < 3 && std::abs(bz) < 3) {
            return;
        }
        std::vector<f32> out(M * M, UNKNOWN);
        for (int zz = 0; zz < M; zz++) {
            const int sz = zz + bz;
            if (sz < 0 || sz >= M) continue;
            for (int xx = 0; xx < M; xx++) {
                const int sx = xx + bx;
                if (sx < 0 || sx >= M) continue;
                out[zz * M + xx] = y[sz * M + sx];
            }
        }
        y.swap(out);
        originX += bx * S;
        originZ += bz * S;
        cursor = 0;
    }

    void refine(int budget) {
        dBgS_ObjGndChk_Wtr wtr;
        dBgS_ObjGndChk gnd;
        const std::vector<int>& ord = order();
        while (budget > 0 && cursor < M * M) {
            const int i = ord[cursor++];
            if (y[i] != UNKNOWN) continue;
            budget--;
            const f32 wx = originX + (i % M) * S, wz = originZ + (i / M) * S;
            cXyz top(wx, refY + 1500.0f, wz);
            wtr.SetPos(&top);
            const f32 wy = dComIfG_Bgsp().GroundCross(&wtr);
            if (wy <= -1.0e8f || wy < refY - 3000.0f) {
                y[i] = NO_WATER;
                continue;
            }
            cXyz gtop(wx, wy + 400.0f, wz);
            gnd.SetPos(&gtop);
            const f32 gy = dComIfG_Bgsp().GroundCross(&gnd);
            y[i] = gy > wy - 3.0f ? NO_WATER : wy;
        }
    }

    // Bilinear water height; false where any of the four samples is not water.
    bool height(f32 wx, f32 wz, f32& out) const {
        const f32 gx = (wx - originX) / S, gz = (wz - originZ) / S;
        const int x0 = int(std::floor(gx)), z0 = int(std::floor(gz));
        if (x0 < 0 || z0 < 0 || x0 + 1 >= M || z0 + 1 >= M) return false;
        const f32 a = y[z0 * M + x0], b = y[z0 * M + x0 + 1];
        const f32 c = y[(z0 + 1) * M + x0], d = y[(z0 + 1) * M + x0 + 1];
        if (a < -1.0e19f || b < -1.0e19f || c < -1.0e19f || d < -1.0e19f) return false;
        if (std::fabs(a - b) > 80.0f || std::fabs(a - c) > 80.0f || std::fabs(a - d) > 80.0f) {
            return false;  // a waterfall or a step between two bodies of water
        }
        const f32 fx = gx - x0, fz = gz - z0;
        out = (a * (1 - fx) + b * fx) * (1 - fz) + (c * (1 - fx) + d * fx) * fz;
        return true;
    }

    bool any_water_near() const {
        for (f32 v : y) {
            if (v > -1.0e19f) return true;
        }
        return false;
    }
} g_map;

// ---------------------------------------------------------------------------------------
// Wave simulation (used at two resolutions)

struct Sim {
    int n = 0;
    f32 dx = 0.0f;
    int mstep = 4;
    int mn = 0;
    f32 waveK = 0.24f;
    f32 damp = 0.9935f;
    f32 minRadius = 0.0f;
    std::vector<f32> h, prev, next;
    std::vector<u8> mask;
    std::vector<int> order;
    f32 originX = 0.0f, originZ = 0.0f;
    f32 waterY = 0.0f;
    bool active = false;
    int idle = 0;
    int maskCursor = 0;
    f32 energy = 0.0f;

    void init(int n_, f32 dx_, int mstep_, f32 waveK_, f32 damp_) {
        n = n_;
        dx = dx_;
        mstep = mstep_;
        mn = n / mstep;
        waveK = waveK_;
        damp = damp_;
        minRadius = dx * 1.6f;
        h.assign(n * n, 0.0f);
        prev.assign(n * n, 0.0f);
        next.assign(n * n, 0.0f);
        mask.assign(mn * mn, MASK_UNKNOWN);
        order.resize(mn * mn);
        for (int i = 0; i < mn * mn; i++) order[i] = i;
        const int m = mn;
        auto d = [m](int i) {
            const f32 x = (i % m) - (m - 1) * 0.5f, z = (i / m) - (m - 1) * 0.5f;
            return x * x + z * z;
        };
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return d(a) < d(b); });
    }

    inline int idx(int x, int z) const { return z * n + x; }
    inline bool cell_sim(int x, int z) const {
        return mask[(z / mstep) * mn + (x / mstep)] != MASK_LAND;
    }
    inline bool cell_water(int x, int z) const {
        return mask[(z / mstep) * mn + (x / mstep)] == MASK_WATER;
    }

    void clear() {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(prev.begin(), prev.end(), 0.0f);
        std::fill(mask.begin(), mask.end(), MASK_UNKNOWN);
        active = false;
        idle = 0;
        energy = 0.0f;
        maskCursor = 0;
    }

    void center_on(f32 x, f32 z) {
        originX = std::floor(x / dx) * dx - (n / 2) * dx;
        originZ = std::floor(z / dx) * dx - (n / 2) * dx;
    }

    void activate(const cXyz& pos, f32 y) {
        clear();
        waterY = y;
        center_on(pos.x, pos.z);
        active = true;
    }

    // Distance in cells from the grid edge (negative outside).
    f32 edge_cells(f32 wx, f32 wz) const {
        const f32 gx = (wx - originX) / dx, gz = (wz - originZ) / dx;
        return std::min(std::min(gx, (n - 1) - gx), std::min(gz, (n - 1) - gz));
    }

    // Weight this layer has at a point once its edge fade is applied (0..1).
    f32 coverage(f32 wx, f32 wz) const {
        if (!active) return 0.0f;
        const f32 e = edge_cells(wx, wz);
        return std::max(0.0f, std::min(1.0f, (e - 1.0f) / 15.0f));
    }

    void follow(f32 x, f32 z) {
        const f32 cx = originX + (n / 2) * dx, cz = originZ + (n / 2) * dx;
        const f32 limit = mstep * dx * 3;
        if (std::fabs(x - cx) < limit && std::fabs(z - cz) < limit) return;
        const int bx = int(std::round((x - cx) / (mstep * dx)));
        const int bz = int(std::round((z - cz) / (mstep * dx)));
        if (bx == 0 && bz == 0) return;
        const int sx0 = bx * mstep, sz0 = bz * mstep;
        auto shift = [&](std::vector<f32>& v) {
            std::vector<f32> out(n * n, 0.0f);
            for (int zz = 0; zz < n; zz++) {
                const int sz = zz + sz0;
                if (sz < 0 || sz >= n) continue;
                for (int xx = 0; xx < n; xx++) {
                    const int sx = xx + sx0;
                    if (sx < 0 || sx >= n) continue;
                    out[idx(xx, zz)] = v[idx(sx, sz)];
                }
            }
            v.swap(out);
        };
        shift(h);
        shift(prev);
        std::vector<u8> m(mn * mn, MASK_UNKNOWN);
        for (int zz = 0; zz < mn; zz++) {
            const int sz = zz + bz;
            if (sz < 0 || sz >= mn) continue;
            for (int xx = 0; xx < mn; xx++) {
                const int sx = xx + bx;
                if (sx < 0 || sx >= mn) continue;
                m[zz * mn + xx] = mask[sz * mn + sx];
            }
        }
        mask.swap(m);
        maskCursor = 0;
        originX += sx0 * dx;
        originZ += sz0 * dx;
    }

    void refine_mask(int budget) {
        dBgS_ObjGndChk_Wtr wtr;
        dBgS_ObjGndChk gnd;
        while (budget > 0 && maskCursor < mn * mn) {
            const int i = order[maskCursor++];
            if (mask[i] != MASK_UNKNOWN) continue;
            budget--;
            const f32 wx = originX + ((i % mn) * mstep + mstep * 0.5f) * dx;
            const f32 wz = originZ + ((i / mn) * mstep + mstep * 0.5f) * dx;
            f32 ref = waterY;
            g_map.height(wx, wz, ref);
            cXyz top(wx, ref + 300.0f, wz);
            wtr.SetPos(&top);
            const f32 wy = dComIfG_Bgsp().GroundCross(&wtr);
            bool water = std::fabs(wy - ref) < 60.0f;
            if (water) {
                cXyz gtop(wx, wy + 300.0f, wz);
                gnd.SetPos(&gtop);
                if (dComIfG_Bgsp().GroundCross(&gnd) > wy - 3.0f) water = false;
            }
            mask[i] = water ? MASK_WATER : MASK_LAND;
        }
    }

    // Press a smooth dip into the surface (a raised cosine, no grid-sized spikes).
    void push(f32 wx, f32 wz, f32 radius, f32 depth) {
        const f32 gx = (wx - originX) / dx, gz = (wz - originZ) / dx;
        const f32 r = std::max(radius, minRadius) / dx;
        // Spread over a larger area at coarse resolution: keep the same volume.
        const f32 vol = (radius * radius) / (std::max(radius, minRadius) * std::max(radius, minRadius));
        const int x0 = std::max(1, int(gx - r)), x1 = std::min(n - 2, int(gx + r) + 1);
        const int z0 = std::max(1, int(gz - r)), z1 = std::min(n - 2, int(gz + r) + 1);
        for (int zz = z0; zz <= z1; zz++) {
            for (int xx = x0; xx <= x1; xx++) {
                const f32 d = std::sqrt((xx - gx) * (xx - gx) + (zz - gz) * (zz - gz)) / r;
                if (d >= 1.0f) continue;
                h[idx(xx, zz)] -= depth * vol * (0.5f + 0.5f * std::cos(d * 3.14159265f));
            }
        }
    }

    void step() {
        for (int it = 0; it < SUBSTEPS; it++) {
            f32 e = 0.0f;
            for (int zz = 1; zz < n - 1; zz++) {
                for (int xx = 1; xx < n - 1; xx++) {
                    const int i = idx(xx, zz);
                    if (!cell_sim(xx, zz)) {
                        next[i] = 0.0f;
                        continue;
                    }
                    const f32 lap = h[i - 1] + h[i + 1] + h[i - n] + h[i + n] - 4.0f * h[i];
                    f32 v = (2.0f * h[i] - prev[i] + waveK * lap) * damp;
                    const int edge = std::min(std::min(xx, n - 1 - xx), std::min(zz, n - 1 - zz));
                    if (edge < 10) v *= 0.90f + 0.01f * edge;  // soak up waves at the edge
                    v = std::max(-30.0f, std::min(30.0f, v));    // a wake, not a tsunami
                    next[i] = v;
                    e += v * v;
                }
            }
            prev.swap(h);
            h.swap(next);
            energy = e;
        }
    }

    bool contains(f32 wx, f32 wz, f32 margin) const {
        return active && edge_cells(wx, wz) > margin;
    }
};

Sim g_fine, g_coarse;
bool g_initialized = false;
char g_stage[8] = {};

void init_once() {
    if (g_initialized) return;
    g_fine.init(160, 9.0f, 4, 0.24f, 0.9935f);
    // Same wave speed as the fine grid: (c dt / dx)^2 scales with 1/dx^2.
    g_coarse.init(128, 24.0f, 4, 0.24f * (9.0f / 24.0f) * (9.0f / 24.0f), 0.9965f);
    g_initialized = true;
}

void clear_all() {
    init_once();
    g_fine.clear();
    g_coarse.clear();
}

f32 water_height_at(const cXyz& pos) {
    f32 y = pos.y;
    g_map.height(pos.x, pos.z, y);
    return y;
}

// Wake a sim for a contact; false when the contact belongs to some other water.
bool engage(Sim& s, const cXyz& pos) {
    if (!s.active) {
        s.activate(pos, pos.y);
    } else if (std::fabs(pos.y - s.waterY) > SAME_WATER && std::fabs(pos.y - water_height_at(pos)) > 30.0f) {
        if (s.idle < 20) return false;
        s.activate(pos, pos.y);
    }
    s.idle = 0;
    return true;
}

// Route a contact to the finest layer that holds it.
void contact(const cXyz& pos, f32 radius, f32 depth) {
    if (!enabled()) return;
    init_once();
    fopAc_ac_c* pl = dComIfGp_getPlayer(0);
    const cXyz center = pl != NULL ? pl->current.pos : pos;
    if (!g_fine.active) {
        // The fine layer lives around the player: start it only for nearby water.
        const f32 dxp = pos.x - center.x, dzp = pos.z - center.z;
        if (dxp * dxp + dzp * dzp < 500.0f * 500.0f) {
            g_fine.activate(cXyz(center.x, pos.y, center.z), pos.y);
        }
    }
    if (g_fine.contains(pos.x, pos.z, 18.0f) && engage(g_fine, pos)) {
        g_fine.push(pos.x, pos.z, radius, depth);
        return;
    }
    if (!g_coarse.active) {
        g_coarse.activate(cXyz(center.x, pos.y, center.z), pos.y);
    }
    if (g_coarse.contains(pos.x, pos.z, 10.0f) && engage(g_coarse, pos)) {
        g_coarse.push(pos.x, pos.z, radius, depth);
    }
}

// ---------------------------------------------------------------------------------------
// Ambient swell

struct Swell {
    f32 dirX, dirZ;  // unit
    f32 k;           // wave number, per unit
    f32 w;           // angular speed, per second
    f32 amp;         // units
    f32 phase;
};
Swell g_swells[4];
f32 g_time = 0.0f;
f32 g_windPow = 0.0f;
f32 g_windX = 1.0f, g_windZ = 0.0f;

void update_swell() {
    g_time += 1.0f / 30.0f;
    cXyz* wv = dKyw_get_wind_vec();
    const f32 pw = dKyw_get_wind_pow();
    f32 wx = 1.0f, wz = 0.0f;
    if (wv != NULL) {
        const f32 l = std::sqrt(wv->x * wv->x + wv->z * wv->z);
        if (l > 1e-3f) {
            wx = wv->x / l;
            wz = wv->z / l;
        }
    }
    // Ease toward the current wind so a wind change does not snap the waves.
    g_windX += (wx - g_windX) * 0.01f;
    g_windZ += (wz - g_windZ) * 0.01f;
    const f32 l = std::sqrt(g_windX * g_windX + g_windZ * g_windZ);
    if (l > 1e-3f) {
        g_windX /= l;
        g_windZ /= l;
    }
    g_windPow += (std::max(0.0f, std::min(1.0f, pw)) - g_windPow) * 0.02f;
    static const f32 kLen[4] = {560.0f, 340.0f, 210.0f, 130.0f};       // units
    static const f32 kAng[4] = {0.0f, 0.42f, -0.61f, 1.05f};           // radians from the wind
    static const f32 kAmp[4] = {1.0f, 0.62f, 0.40f, 0.25f};
    static const f32 kPhase[4] = {0.0f, 1.7f, 4.1f, 2.6f};
    const f32 base = 1.6f + 4.2f * g_windPow;  // ~2 cm calm, ~6 cm windy
    for (int i = 0; i < 4; i++) {
        const f32 c = std::cos(kAng[i]), s = std::sin(kAng[i]);
        Swell& sw = g_swells[i];
        sw.dirX = g_windX * c - g_windZ * s;
        sw.dirZ = g_windX * s + g_windZ * c;
        sw.k = 6.2831853f / kLen[i];
        // Deep-water dispersion, 100 units to the meter.
        sw.w = std::sqrt(9.81f * sw.k * 100.0f);
        sw.amp = base * kAmp[i];
        sw.phase = kPhase[i];
    }
}

inline void swell_at(f32 x, f32 z, f32& h, f32& gx, f32& gz) {
    h = gx = gz = 0.0f;
    for (const Swell& s : g_swells) {
        const f32 a = s.k * (s.dirX * x + s.dirZ * z) - s.w * g_time + s.phase;
        const f32 sn = std::sin(a), cs = std::cos(a);
        h += s.amp * sn;
        const f32 d = s.amp * s.k * cs;
        gx += d * s.dirX;
        gz += d * s.dirZ;
    }
}

// ---------------------------------------------------------------------------------------
// Drawing

struct Light {
    cXyz sun;
    cXyz eye;
    f32 strength;
    GXColor tint;
};

Light current_light() {
    Light l;
    dScnKy_env_light_c* env = dKy_getEnvlight();
    camera_process_class* cam = dComIfGp_getCamera(0);
    l.eye = cam != NULL ? cam->view.lookat.eye : cXyz(g_map.originX, g_fine.waterY + 200.0f, g_map.originZ);
    cXyz d = env->sun_pos - l.eye;
    if (d.abs() < 1.0f || d.y < 0.05f * d.abs()) d.set(0.3f, 1.0f, 0.2f);
    l.sun = d.normZP();
    const f32 t = env->getDaytime();
    const bool day = t >= 90.0f && t < 285.0f;
    l.strength = day ? 1.0f : 0.45f;
    if (dKy_darkworld_check()) l.strength *= 0.7f;
    l.tint = day ? GXColor{255, 250, 236, 255} : GXColor{190, 210, 255, 255};
    return l;
}

struct Shader {
    Light L;
    f32 spec0, fres0, lam0;

    explicit Shader(const Light& light, f32 y) : L(light) {
        const cXyz mid(g_map.originX + (WaterMap::M / 2) * WaterMap::S, y,
                       g_map.originZ + (WaterMap::M / 2) * WaterMap::S);
        const cXyz v0 = (L.eye - mid).normZP();
        const cXyz hv0 = (L.sun + v0).normZP();
        spec0 = std::pow(std::max(0.0f, hv0.y), 80.0f);
        fres0 = std::pow(1.0f - std::max(0.0f, v0.y), 5.0f);
        lam0 = std::max(0.0f, L.sun.y);
    }

    // h: height above the flat surface; gx/gz: slope; lap: curvature (sim only);
    // presenceGain: how strongly calm slopes show; fade: layer weight at this point.
    GXColor shade(f32 wx, f32 wy, f32 wz, f32 h, f32 gx, f32 gz, f32 lap, f32 presenceGain,
                  f32 fade) const {
        constexpr f32 NORMAL_BOOST = 2.2f;
        const f32 bx = gx * NORMAL_BOOST, bz = gz * NORMAL_BOOST;
        const cXyz n = cXyz(-bx, 1.0f, -bz).normZP();
        const cXyz v = cXyz(L.eye.x - wx, L.eye.y - wy, L.eye.z - wz).normZP();
        const cXyz hv = (L.sun + v).normZP();
        const f32 spec = std::pow(std::max(0.0f, n.x * hv.x + n.y * hv.y + n.z * hv.z), 80.0f);
        const f32 nv = std::max(0.0f, n.x * v.x + n.y * v.y + n.z * v.z);
        const f32 fres = std::pow(1.0f - nv, 5.0f);
        const f32 lam = std::max(0.0f, n.x * L.sun.x + n.y * L.sun.y + n.z * L.sun.z);
        const f32 slope = std::sqrt(bx * bx + bz * bz);
        const f32 d = (lam - lam0) * 2.4f + (fres - fres0) * 1.3f + (spec - spec0) * 2.6f +
                      h * 0.045f - lap * 0.05f;
        const f32 foam = std::max(0.0f, std::min(1.0f, (slope - 0.9f) * 1.2f + (h - 9.0f) * 0.08f));
        const f32 presence = std::min(1.0f, (std::fabs(h) * 0.5f + slope * 5.0f) * presenceGain);
        f32 a = (0.22f + std::min(1.0f, std::fabs(d)) * 0.75f) * presence + foam * 0.65f;
        a = std::min(a, 0.92f) * fade * L.strength;
        GXColor c;
        if (foam > 0.05f) {
            c = GXColor{u8(L.tint.r * 0.96f), u8(L.tint.g * 0.97f), L.tint.b, 0};
        } else if (d >= 0.0f) {
            const f32 m = 0.72f + 0.28f * std::min(1.0f, d);
            c = GXColor{u8(L.tint.r * m), u8(L.tint.g * m), u8(L.tint.b * m), 0};
        } else {
            const f32 m = std::min(1.0f, -d);
            c = GXColor{u8(28 - 18 * m), u8(58 - 30 * m), u8(66 - 30 * m), 0};
        }
        c.a = u8(std::max(0.0f, std::min(255.0f, a * 255.0f)));
        return c;
    }
};

void setup_gx() {
    j3dSys.reinitGX();
    GXSetNumIndStages(0);
    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
    GXSetVtxDesc(GX_VA_CLR0, GX_DIRECT);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
    GXSetNumChans(1);
    GXSetChanCtrl(GX_COLOR0A0, GX_DISABLE, GX_SRC_REG, GX_SRC_VTX, GX_LIGHT_NULL, GX_DF_NONE,
                  GX_AF_NONE);
    GXSetNumTexGens(0);
    GXSetNumTevStages(1);
    GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD_NULL, GX_TEXMAP_NULL, GX_COLOR0A0);
    GXSetTevOp(GX_TEVSTAGE0, GX_PASSCLR);
    GXLoadPosMtxImm(j3dSys.getViewMtx(), GX_PNMTX0);
    GXSetCurrentMtx(GX_PNMTX0);
    GXSetBlendMode(GX_BM_BLEND, GX_BL_SRCALPHA, GX_BL_INVSRCALPHA, GX_LO_SET);
    GXSetCullMode(GX_CULL_NONE);
    GXSetZMode(GX_ENABLE, GX_LEQUAL, GX_DISABLE);
    GXSetAlphaCompare(GX_GREATER, 0, GX_AOP_OR, GX_GREATER, 0);
    GXSetFog(GX_FOG_NONE, 0.0f, 0.0f, 0.0f, 0.0f, GXColor{0, 0, 0, 0});
}

struct Vert {
    f32 x, y, z;
    GXColor c;
    bool ok;
};

// Emit a grid as triangle strips, skipping runs where neither row has anything to show.
template <typename F>
void draw_grid(int nx, int nz, F&& vertex) {
    std::vector<Vert> row0(nx), row1(nx);
    for (int x = 0; x < nx; x++) row0[x] = vertex(x, 0);
    for (int z = 0; z + 1 < nz; z++) {
        for (int x = 0; x < nx; x++) row1[x] = vertex(x, z + 1);
        int x = 0;
        while (x < nx) {
            while (x < nx && !((row0[x].ok && row0[x].c.a) || (row1[x].ok && row1[x].c.a))) x++;
            int end = x;
            while (end < nx && (row0[end].ok || row1[end].ok)) end++;
            const int a = std::max(0, x - 1), b = std::min(nx, end + 1);
            if (b - a >= 2) {
                GXBegin(GX_TRIANGLESTRIP, GX_VTXFMT0, (b - a) * 2);
                for (int i = a; i < b; i++) {
                    const Vert& p = row0[i];
                    const Vert& q = row1[i];
                    GXPosition3f32(p.x, p.y, p.z);
                    GXColor4u8(p.c.r, p.c.g, p.c.b, p.ok ? p.c.a : 0);
                    GXPosition3f32(q.x, q.y, q.z);
                    GXColor4u8(q.c.r, q.c.g, q.c.b, q.ok ? q.c.a : 0);
                }
                GXEnd();
            }
            x = std::max(end, x + 1);
        }
        row0.swap(row1);
    }
}

void draw_sim(const Sim& s, const Shader& sh, const Sim* inner) {
    draw_grid(s.n - 2, s.n - 2, [&](int gx, int gz) -> Vert {
        const int x = gx + 1, z = gz + 1;
        const f32 wx = s.originX + x * s.dx, wz = s.originZ + z * s.dx;
        Vert v{wx, s.waterY, wz, GXColor{0, 0, 0, 0}, false};
        if (!s.cell_water(x, z)) return v;
        f32 base = s.waterY;
        g_map.height(wx, wz, base);
        const int i = s.idx(x, z);
        f32 ah, agx, agz;
        swell_at(wx, wz, ah, agx, agz);
        const f32 hh = s.h[i] + ah;
        const f32 gxx = (s.h[i + 1] - s.h[i - 1]) / (2.0f * s.dx) + agx;
        const f32 gzz = (s.h[i + s.n] - s.h[i - s.n]) / (2.0f * s.dx) + agz;
        const f32 lap = (s.h[i - 1] + s.h[i + 1] + s.h[i - s.n] + s.h[i + s.n] - 4.0f * s.h[i]) *
                        (9.0f / s.dx);
        f32 fade = s.coverage(wx, wz);
        if (inner != nullptr) fade *= 1.0f - inner->coverage(wx, wz);
        v.y = base + 0.8f + hh;
        v.c = sh.shade(wx, v.y, wz, hh, gxx, gzz, lap, 1.0f, fade);
        v.ok = fade > 0.0f;
        return v;
    });
}

void draw_ambient(const Shader& sh) {
    constexpr f32 STEP = 80.0f;
    const int n = int((WaterMap::M - 1) * WaterMap::S / STEP);
    const f32 ox = g_map.originX, oz = g_map.originZ;
    const f32 cx = ox + (WaterMap::M - 1) * WaterMap::S * 0.5f, cz = oz + (WaterMap::M - 1) * WaterMap::S * 0.5f;
    const f32 radius = (WaterMap::M - 1) * WaterMap::S * 0.5f;
    draw_grid(n, n, [&](int x, int z) -> Vert {
        const f32 wx = ox + x * STEP, wz = oz + z * STEP;
        Vert v{wx, 0.0f, wz, GXColor{0, 0, 0, 0}, false};
        f32 base;
        if (!g_map.height(wx, wz, base)) return v;
        f32 h, gx, gz;
        swell_at(wx, wz, h, gx, gz);
        // Fade toward the edge of the map, and under the interactive layers.
        const f32 r = std::sqrt((wx - cx) * (wx - cx) + (wz - cz) * (wz - cz)) / radius;
        f32 fade = std::max(0.0f, std::min(1.0f, (1.0f - r) * 3.0f));
        fade *= 1.0f - std::max(g_fine.coverage(wx, wz), g_coarse.coverage(wx, wz));
        v.y = base + 0.8f + h;
        v.c = sh.shade(wx, v.y, wz, h, gx, gz, 0.0f, 0.55f, fade);
        v.ok = fade > 0.0f;
        return v;
    });
}

class SurfacePacket : public J3DPacket {
public:
    void draw() override;
};
SurfacePacket l_packet;

void SurfacePacket::draw() {
    if (!enabled()) return;
    const Light L = current_light();
    f32 y = g_fine.active ? g_fine.waterY : g_coarse.waterY;
    if (fopAc_ac_c* pl = dComIfGp_getPlayer(0)) {
        g_map.height(pl->current.pos.x, pl->current.pos.z, y);
    }
    const Shader sh(L, y);
    setup_gx();
    if (g_map.any_water_near()) draw_ambient(sh);
    if (g_coarse.active) draw_sim(g_coarse, sh, g_fine.active ? &g_fine : nullptr);
    if (g_fine.active) draw_sim(g_fine, sh, nullptr);
    J3DShape::resetVcdVatCache();
}

// While it rains, let each drop that lands on nearby water leave a small ring.
void rain() {
    const int cnt = g_env_light.raincnt;
    if (cnt <= 3) return;
    fopAc_ac_c* pl = dComIfGp_getPlayer(0);
    if (pl == NULL) return;
    if (!g_fine.active || !g_coarse.active) {
        f32 wy;
        if (!g_map.height(pl->current.pos.x, pl->current.pos.z, wy)) {
            // Look for water a little around the player.
            bool found = false;
            for (int k = 0; k < 8 && !found; k++) {
                const f32 a = k * 0.785398f;
                found = g_map.height(pl->current.pos.x + std::cos(a) * 600.0f,
                                     pl->current.pos.z + std::sin(a) * 600.0f, wy);
            }
            if (!found) return;
        }
        const cXyz c(pl->current.pos.x, wy, pl->current.pos.z);
        if (!g_fine.active) g_fine.activate(c, wy);
        if (!g_coarse.active) g_coarse.activate(c, wy);
    }
    g_fine.idle = 0;
    g_coarse.idle = 0;
    static u32 seed = 0x2545F491u;
    auto rnd = [] {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return (seed & 0xFFFFFF) / f32(0x1000000);
    };
    const int drops = 2 + cnt * 14 / 250;
    for (Sim* s : {&g_fine, &g_coarse}) {
        for (int d = 0; d < drops; d++) {
            const int x = 2 + int(rnd() * (s->n - 4)), z = 2 + int(rnd() * (s->n - 4));
            if (!s->cell_water(x, z)) continue;
            const f32 wx = s->originX + x * s->dx, wz = s->originZ + z * s->dx;
            if (s == &g_coarse && g_fine.coverage(wx, wz) > 0.0f) continue;  // fine has these
            s->push(wx, wz, 7.0f + rnd() * 5.0f, 1.0f + rnd() * 1.4f);
        }
    }
}

}  // namespace

void ripple(const cXyz& pos, f32 speed) {
    // `speed` is how far the actor moved this frame (Link walking ~5-10, running ~20,
    // Epona galloping 40+). Tuned offline: ~4 cm crests walking, ~11-12 cm running.
    const f32 v = std::max(0.0f, speed);
    contact(pos, 16.0f + std::min(20.0f, 0.5f * v), std::min(0.05f + 0.6f * std::pow(v, 0.7f), 12.0f));
}

void splash(const cXyz& pos, f32 scale) {
    const f32 sc = std::max(0.3f, std::min(scale, 4.0f));
    contact(pos, 40.0f * sc, std::min(56.0f * sc, 60.0f));  // ~7 cm ring
}

void update() {
    g_frame++;
    if (!enabled()) {
        if (g_initialized && (g_fine.active || g_coarse.active)) clear_all();
        return;
    }
    init_once();
    const char* stage = dComIfGp_getStartStageName();
    if (stage != NULL && std::strncmp(stage, g_stage, sizeof(g_stage)) != 0) {
        std::strncpy(g_stage, stage, sizeof(g_stage));
        clear_all();
        g_map.clear();
        return;
    }
    fopAc_ac_c* pl = dComIfGp_getPlayer(0);
    if (pl == NULL) return;
    g_map.refY = pl->current.pos.y;
    g_map.follow(pl->current.pos.x, pl->current.pos.z);
    g_map.refine(96);
    update_swell();
    rain();
    for (Sim* s : {&g_fine, &g_coarse}) {
        if (!s->active) continue;
        s->follow(pl->current.pos.x, pl->current.pos.z);
        s->refine_mask(s == &g_fine ? 160 : 120);
        s->step();
        s->idle++;
        if (s->idle > IDLE_FRAMES && s->energy < 0.02f) s->clear();
    }
}

void draw() {
    if (!enabled()) return;
    if (!g_initialized) return;
    if (!g_fine.active && !g_coarse.active && !g_map.any_water_near()) return;
    // Translucent object list: drawn after the stage's water (XluListBG) and its particles.
    dComIfGd_setList();
    j3dSys.getDrawBuffer(J3DSysDrawBuf_Xlu)->entryImm(&l_packet, 0);
}

void reset() {
    clear_all();
    g_map.clear();
}

bool hide_flat_ripples() {
    return enabled() && g_initialized && (g_fine.active || g_coarse.active);
}

}  // namespace dusk::water
