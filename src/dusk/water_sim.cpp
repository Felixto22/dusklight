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
// reflect off shores, rocks and pillars. The sheets redraw the stage's own water, bent by
// the waves, with screen-space reflections, sun glint and foam (see "Drawing" below).
// Inner layers fade out at their edges and the outer layers fade in under them.

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

#include <aurora/gfx.hpp>
#include <dolphin/gx/GXAurora.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <array>
#include <mutex>
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

// ---------------------------------------------------------------------------------------
// Drawing
//
// The sheets no longer paint a colour of their own. After the stage's water has been
// drawn, the frame (colour + depth) is snapshotted and the sheets redraw it: the picture of
// the original water is shifted by the waves' slope (refraction), mixed with real-time
// screen-space reflections of whatever is on screen above the water (Link, Epona, trees,
// cliffs) by the Fresnel term, plus the sun glint and foam on steep crests. With flat water
// the result is the original water itself (plus reflections). Samples that would come from
// something in front of the water (Link's arm, a leg) are never used.

struct WaterVtx {
    f32 pos[3];
    f32 nrm[3];
    f32 alpha;
    f32 foam;
};
static_assert(sizeof(WaterVtx) == 32);

struct WaterParams {
    f32 vp[4][4];   // view-projection rows (reversed-Z clip, as aurora draws)
    f32 eye[4];     // xyz, time
    f32 sun[4];     // direction xyz, glint strength
    f32 sky[4];     // fallback reflection colour, rgb
    f32 prj[4];     // p22, p23, width, height
    f32 misc[4];    // refraction strength, reflection strength, foam strength, -
};
static_assert(sizeof(WaterParams) == 144);

struct WaterPayload {
    WGPUTextureView color;
    WGPUTextureView depth;
    u32 vtxOffset, vtxSize;
    u32 idxOffset, idxCount;
    u32 uniOffset;
    u32 width, height;
};

constexpr const char* kWaterShader = R"(
struct P {
    vp0: vec4f, vp1: vec4f, vp2: vec4f, vp3: vec4f,
    eye: vec4f, sun: vec4f, sky: vec4f, prj: vec4f, misc: vec4f,
};
@group(0) @binding(0) var<uniform> u: P;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;

struct VI {
    @location(0) pos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) alpha: f32,
    @location(3) foam: f32,
};
struct VO {
    @builtin(position) pos: vec4f,
    @location(0) wpos: vec3f,
    @location(1) nrm: vec3f,
    @location(2) alpha: f32,
    @location(3) foam: f32,
};

fn project(p: vec3f) -> vec4f {
    let h = vec4f(p, 1.0);
    return vec4f(dot(u.vp0, h), dot(u.vp1, h), dot(u.vp2, h), dot(u.vp3, h));
}

@vertex fn vs_main(in: VI) -> VO {
    var o: VO;
    o.pos = project(in.pos);
    o.wpos = in.pos;
    o.nrm = in.nrm;
    o.alpha = in.alpha;
    o.foam = in.foam;
    return o;
}

fn lin(d: f32) -> f32 { return u.prj.y / (d - u.prj.x); }   // view-space z (negative)

fn depth_at(uv: vec2f) -> f32 {
    let sz = vec2i(textureDimensions(depth_tex));
    let c = clamp(vec2i(uv * vec2f(sz)), vec2i(0), sz - vec2i(1));
    return textureLoad(depth_tex, c, 0).r;
}

@fragment fn fs_main(in: VO) -> @location(0) vec4f {
    let size = u.prj.zw;
    let uv = in.pos.xy / size;
    let n = normalize(in.nrm);
    let V = normalize(u.eye.xyz - in.wpos);
    let my_d = in.pos.z;
    let dist = max(-lin(my_d), 50.0);

    // Refraction: shift the picture of the water by the wave slope (less with distance).
    var ruv = uv + n.xz * vec2f(1.0, -1.0) * (u.misc.x * 900.0 / dist) / size.y;
    ruv = clamp(ruv, vec2f(0.0), vec2f(1.0));
    if (depth_at(ruv) > my_d + 1e-5) { ruv = uv; }   // that pixel is in front of the water
    let refr = textureSampleLevel(scene, samp, ruv, 0.0).rgb;

    // Screen-space reflection along the reflected view ray.
    let R = reflect(-V, n);
    var hit = false;
    var huv = vec2f(0.0);
    let jit = fract(52.9829189 * fract(dot(in.pos.xy, vec2f(0.06711056, 0.00583715))));
    if (R.y > 0.0) {
        for (var i = 0; i < 32; i++) {
            let f = (f32(i) + jit) / 32.0;
            let t = 4.0 + f * f * 3000.0;
            let c = project(in.wpos + R * t);
            if (c.w <= 0.0) { break; }
            let ndc = c.xy / c.w;
            let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (any(suv < vec2f(0.0)) || any(suv > vec2f(1.0))) { break; }
            let rd = c.z / c.w;
            let sd = depth_at(suv);
            if (sd > rd) {   // the scene is in front of the ray point
                let gap = lin(sd) - lin(rd);
                if (gap < 25.0 + t * 0.06) { hit = true; huv = suv; }
                break;
            }
        }
    }
    var refl = u.sky.rgb;
    var reflW = 0.5;
    if (hit) {
        refl = textureSampleLevel(scene, samp, huv, 0.0).rgb;
        let e = min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y));
        reflW = clamp(e * 12.0, 0.0, 1.0);
    }
    let ndv = clamp(dot(n, V), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    var col = mix(refr, refl, clamp(fres * u.misc.y * reflW, 0.0, 0.85));

    // Sun glint on the waves and foam on steep crests.
    let H = normalize(u.sun.xyz + V);
    let spec = pow(max(dot(n, H), 0.0), 220.0) * u.sun.w;
    col += vec3f(spec);
    col = mix(col, vec3f(0.92, 0.95, 0.97) * (0.55 + 0.45 * u.sun.w), clamp(in.foam * u.misc.z, 0.0, 0.8));
    return vec4f(col, in.alpha);
}
)";

struct WaterGpu {
    std::mutex mutex;
    u64 layoutKey = ~0ull;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroupLayout bgl;
    wgpu::Sampler sampler;
};
WaterGpu g_gpu;
aurora::gfx::DrawTypeId g_drawType = aurora::gfx::InvalidDrawType;

void build_pipeline(const aurora::gfx::DrawContext& ctx) {
    const auto& layout = ctx.layout;
    if (g_gpu.pipeline && g_gpu.layoutKey == layout.key) return;
    const wgpu::ShaderSourceWGSL src{wgpu::ShaderSourceWGSL::Init{.code = kWaterShader}};
    const wgpu::ShaderModuleDescriptor md{.nextInChain = &src, .label = "Felix water"};
    const auto module = ctx.device.CreateShaderModule(&md);
    if (!g_gpu.bgl) {
        const wgpu::BindGroupLayoutEntry e[4] = {
            {.binding = 0,
             .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
             .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(WaterParams)}},
            {.binding = 1,
             .visibility = wgpu::ShaderStage::Fragment,
             .texture = {.sampleType = wgpu::TextureSampleType::Float,
                         .viewDimension = wgpu::TextureViewDimension::e2D}},
            {.binding = 2,
             .visibility = wgpu::ShaderStage::Fragment,
             .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
            {.binding = 3,
             .visibility = wgpu::ShaderStage::Fragment,
             .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                         .viewDimension = wgpu::TextureViewDimension::e2D}},
        };
        const wgpu::BindGroupLayoutDescriptor bd{.label = "Felix water", .entryCount = 4, .entries = e};
        g_gpu.bgl = ctx.device.CreateBindGroupLayout(&bd);
        const wgpu::SamplerDescriptor sd{
            .label = "Felix water",
            .addressModeU = wgpu::AddressMode::ClampToEdge,
            .addressModeV = wgpu::AddressMode::ClampToEdge,
            .magFilter = wgpu::FilterMode::Linear,
            .minFilter = wgpu::FilterMode::Linear,
        };
        g_gpu.sampler = ctx.device.CreateSampler(&sd);
    }
    const wgpu::PipelineLayoutDescriptor pld{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_gpu.bgl};
    const auto pl = ctx.device.CreatePipelineLayout(&pld);
    const wgpu::VertexAttribute attrs[4] = {
        {.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0},
        {.format = wgpu::VertexFormat::Float32x3, .offset = 12, .shaderLocation = 1},
        {.format = wgpu::VertexFormat::Float32, .offset = 24, .shaderLocation = 2},
        {.format = wgpu::VertexFormat::Float32, .offset = 28, .shaderLocation = 3},
    };
    const wgpu::VertexBufferLayout vbl{
        .stepMode = wgpu::VertexStepMode::Vertex, .arrayStride = sizeof(WaterVtx), .attributeCount = 4,
        .attributes = attrs};
    const wgpu::BlendState blend{
        .color = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::SrcAlpha,
                  .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
        .alpha = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::Zero,
                  .dstFactor = wgpu::BlendFactor::One},
    };
    std::array<wgpu::ColorTargetState, aurora::gfx::MaxColorAttachments> targets{};
    for (uint32_t i = 0; i < layout.colorAttachmentCount; i++) {
        targets[i] = {.format = layout.colorAttachments[i].format, .writeMask = wgpu::ColorWriteMask::None};
    }
    targets[aurora::gfx::SceneColorAttachmentIndex].blend = &blend;
    targets[aurora::gfx::SceneColorAttachmentIndex].writeMask =
        wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
    const wgpu::FragmentState fs{
        .module = module, .entryPoint = "fs_main", .targetCount = layout.colorAttachmentCount,
        .targets = targets.data()};
    const wgpu::DepthStencilState ds{
        .format = layout.depthStencilFormat,
        .depthWriteEnabled = false,
        .depthCompare = aurora::gfx::uses_reversed_z() ? wgpu::CompareFunction::GreaterEqual
                                                      : wgpu::CompareFunction::LessEqual,
    };
    const wgpu::RenderPipelineDescriptor pd{
        .label = "Felix water",
        .layout = pl,
        .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main", .bufferCount = 1, .buffers = &vbl},
        .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
        .depthStencil = layout.depthStencilFormat != wgpu::TextureFormat::Undefined ? &ds : nullptr,
        .multisample = wgpu::MultisampleState{.count = layout.sampleCount},
        .fragment = &fs,
    };
    g_gpu.pipeline = ctx.device.CreateRenderPipeline(&pd);
    g_gpu.layoutKey = layout.key;
}

void draw_cb(const aurora::gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void* payload,
             size_t payloadSize, void*) {
    if (payloadSize != sizeof(WaterPayload)) return;
    WaterPayload p;
    std::memcpy(&p, payload, sizeof(p));
    if (p.color == nullptr || p.depth == nullptr || p.idxCount == 0) return;
    std::lock_guard lock{g_gpu.mutex};
    build_pipeline(ctx);
    if (!g_gpu.pipeline) return;
    const wgpu::TextureView color{p.color}, depth{p.depth};
    const wgpu::BindGroupEntry e[4] = {
        {.binding = 0, .buffer = ctx.uniformBuffer, .offset = p.uniOffset, .size = sizeof(WaterParams)},
        {.binding = 1, .textureView = color},
        {.binding = 2, .sampler = g_gpu.sampler},
        {.binding = 3, .textureView = depth},
    };
    const wgpu::BindGroupDescriptor bd{.layout = g_gpu.bgl, .entryCount = 4, .entries = e};
    const auto bg = ctx.device.CreateBindGroup(&bd);
    pass.SetPipeline(g_gpu.pipeline);
    pass.SetBindGroup(0, bg);
    pass.SetVertexBuffer(0, ctx.vertexBuffer, p.vtxOffset, p.vtxSize);
    pass.SetIndexBuffer(ctx.indexBuffer, wgpu::IndexFormat::Uint32, p.idxOffset, p.idxCount * 4);
    pass.SetViewport(0.0f, 0.0f, f32(p.width), f32(p.height), 0.0f, 1.0f);
    pass.SetScissorRect(0, 0, p.width, p.height);
    pass.DrawIndexed(p.idxCount);
}

std::vector<WaterVtx> g_vtx;
std::vector<u32> g_idx;

// Add a grid: `vertex(x, z, out)` fills a vertex and returns whether it is on water.
template <typename F>
void add_grid(int nx, int nz, F&& vertex) {
    const u32 base = u32(g_vtx.size());
    std::vector<u8> ok(size_t(nx) * nz);
    for (int z = 0; z < nz; z++) {
        for (int x = 0; x < nx; x++) {
            WaterVtx v{};
            ok[size_t(z) * nx + x] = vertex(x, z, v) ? 1 : 0;
            g_vtx.push_back(v);
        }
    }
    for (int z = 0; z + 1 < nz; z++) {
        for (int x = 0; x + 1 < nx; x++) {
            const u32 a = z * nx + x, b = a + 1, c = a + nx, d = c + 1;
            if (!(ok[a] || ok[b] || ok[c] || ok[d])) continue;
            const auto has = [&](u32 i) { return g_vtx[base + i].alpha > 0.0f; };
            if (!(has(a) || has(b) || has(c) || has(d))) continue;
            g_idx.insert(g_idx.end(), {base + a, base + c, base + b, base + b, base + c, base + d});
        }
    }
}

void fill_vertex(WaterVtx& v, f32 x, f32 y, f32 z, f32 gx, f32 gz, f32 h, f32 alpha) {
    constexpr f32 NORMAL_BOOST = 2.2f;
    const f32 bx = gx * NORMAL_BOOST, bz = gz * NORMAL_BOOST;
    const f32 l = std::sqrt(bx * bx + 1.0f + bz * bz);
    v.pos[0] = x;
    v.pos[1] = y;
    v.pos[2] = z;
    v.nrm[0] = -bx / l;
    v.nrm[1] = 1.0f / l;
    v.nrm[2] = -bz / l;
    v.alpha = alpha;
    const f32 slope = std::sqrt(bx * bx + bz * bz);
    v.foam = std::max(0.0f, std::min(1.0f, (slope - 0.9f) * 1.2f + (h - 9.0f) * 0.08f));
}

void add_sim(const Sim& s, const Sim* inner) {
    add_grid(s.n - 2, s.n - 2, [&](int gx, int gz, WaterVtx& v) -> bool {
        const int x = gx + 1, z = gz + 1;
        const f32 wx = s.originX + x * s.dx, wz = s.originZ + z * s.dx;
        f32 base = s.waterY;
        g_map.height(wx, wz, base);
        if (!s.cell_water(x, z)) {
            fill_vertex(v, wx, base + 0.8f, wz, 0.0f, 0.0f, 0.0f, 0.0f);
            return false;
        }
        const int i = s.idx(x, z);
        f32 ah, agx, agz;
        swell_at(wx, wz, ah, agx, agz);
        const f32 hh = s.h[i] + ah;
        const f32 gxx = (s.h[i + 1] - s.h[i - 1]) / (2.0f * s.dx) + agx;
        const f32 gzz = (s.h[i + s.n] - s.h[i - s.n]) / (2.0f * s.dx) + agz;
        f32 fade = s.coverage(wx, wz);
        if (inner != nullptr) fade *= 1.0f - inner->coverage(wx, wz);
        fill_vertex(v, wx, base + 0.8f + hh, wz, gxx, gzz, hh, fade);
        return fade > 0.0f;
    });
}

void add_ambient() {
    constexpr f32 STEP = 80.0f;
    const int n = int((WaterMap::M - 1) * WaterMap::S / STEP);
    const f32 ox = g_map.originX, oz = g_map.originZ;
    const f32 cx = ox + (WaterMap::M - 1) * WaterMap::S * 0.5f, cz = oz + (WaterMap::M - 1) * WaterMap::S * 0.5f;
    const f32 radius = (WaterMap::M - 1) * WaterMap::S * 0.5f;
    add_grid(n, n, [&](int x, int z, WaterVtx& v) -> bool {
        const f32 wx = ox + x * STEP, wz = oz + z * STEP;
        f32 base;
        if (!g_map.height(wx, wz, base)) {
            fill_vertex(v, wx, 0.0f, wz, 0.0f, 0.0f, 0.0f, 0.0f);
            return false;
        }
        f32 h, gx, gz;
        swell_at(wx, wz, h, gx, gz);
        const f32 r = std::sqrt((wx - cx) * (wx - cx) + (wz - cz) * (wz - cz)) / radius;
        f32 fade = std::max(0.0f, std::min(1.0f, (1.0f - r) * 3.0f));
        fade *= 1.0f - std::max(g_fine.coverage(wx, wz), g_coarse.coverage(wx, wz));
        fill_vertex(v, wx, base + 0.8f + h, wz, gx, gz, h, fade);
        return fade > 0.0f;
    });
}

class SurfacePacket : public J3DPacket {
public:
    void draw() override;
};
SurfacePacket l_packets[4];
u32 l_packetNext = 0;

void SurfacePacket::draw() {
    if (!enabled()) return;
    camera_process_class* cam = dComIfGp_getCamera(0);
    if (cam == NULL || aurora::gfx::is_offscreen()) return;
    if (g_drawType == aurora::gfx::InvalidDrawType) {
        g_drawType = aurora::gfx::register_draw_type({.label = "Felix water", .draw = draw_cb});
        if (g_drawType == aurora::gfx::InvalidDrawType) return;
    }

    g_vtx.clear();
    g_idx.clear();
    if (g_map.any_water_near()) add_ambient();
    if (g_coarse.active) add_sim(g_coarse, g_fine.active ? &g_fine : nullptr);
    if (g_fine.active) add_sim(g_fine, nullptr);
    if (g_idx.empty()) return;

    // View-projection as aurora draws it: GX projection with reversed Z.
    const Mtx44& P = cam->view.projMtx;
    MtxP Vw = j3dSys.getViewMtx();
    WaterParams prm{};
    for (int r = 0; r < 4; r++) {
        f32 pr[4] = {P[r][0], P[r][1], P[r][2], P[r][3]};
        if (r == 2 && aurora::gfx::uses_reversed_z()) {
            for (f32& c : pr) c = -c;
        }
        for (int c = 0; c < 4; c++) {
            f32 s = pr[3] * (c == 3 ? 1.0f : 0.0f);
            for (int k = 0; k < 3; k++) s += pr[k] * Vw[k][c];
            prm.vp[r][c] = s;
        }
    }
    // Recover the camera position from the view matrix (inverse of the rotation part).
    const f32 ex = -(Vw[0][0] * Vw[0][3] + Vw[1][0] * Vw[1][3] + Vw[2][0] * Vw[2][3]);
    const f32 ey = -(Vw[0][1] * Vw[0][3] + Vw[1][1] * Vw[1][3] + Vw[2][1] * Vw[2][3]);
    const f32 ez = -(Vw[0][2] * Vw[0][3] + Vw[1][2] * Vw[1][3] + Vw[2][2] * Vw[2][3]);
    prm.eye[0] = ex;
    prm.eye[1] = ey;
    prm.eye[2] = ez;
    prm.eye[3] = g_time;
    dScnKy_env_light_c* env = dKy_getEnvlight();
    cXyz sd(env->sun_pos.x - ex, env->sun_pos.y - ey, env->sun_pos.z - ez);
    if (sd.abs() < 1.0f || sd.y < 0.05f * sd.abs()) sd.set(0.3f, 1.0f, 0.2f);
    sd = sd.normZP();
    const f32 t = env->getDaytime();
    const bool day = t >= 90.0f && t < 285.0f;
    prm.sun[0] = sd.x;
    prm.sun[1] = sd.y;
    prm.sun[2] = sd.z;
    prm.sun[3] = (day ? 1.6f : 0.35f) * (dKy_darkworld_check() ? 0.6f : 1.0f);
    const GXColorS10 sky = env->vrbox_sky_col;
    prm.sky[0] = std::clamp(sky.r / 255.0f, 0.0f, 1.0f);
    prm.sky[1] = std::clamp(sky.g / 255.0f, 0.0f, 1.0f);
    prm.sky[2] = std::clamp(sky.b / 255.0f, 0.0f, 1.0f);
    prm.prj[0] = aurora::gfx::uses_reversed_z() ? P[2][2] : P[2][2];
    prm.prj[1] = P[2][3];
    prm.misc[0] = 1.0f;  // refraction
    prm.misc[1] = 1.0f;  // reflection
    prm.misc[2] = 1.0f;  // foam

    AuroraGXSync();
    aurora::gfx::ResolvedTargets targets;
    if (!aurora::gfx::resolve_pass({.color = true, .depth = true, .normal = false}, targets) ||
        !targets.color || !targets.depth || targets.width == 0)
    {
        return;
    }
    prm.prj[2] = f32(targets.width);
    prm.prj[3] = f32(targets.height);
    const auto vr = aurora::gfx::push_verts(reinterpret_cast<const uint8_t*>(g_vtx.data()),
                                            g_vtx.size() * sizeof(WaterVtx), 4);
    const auto ir = aurora::gfx::push_indices(reinterpret_cast<const uint8_t*>(g_idx.data()),
                                              g_idx.size() * sizeof(u32), 4);
    const auto ur = aurora::gfx::push_uniform(reinterpret_cast<const uint8_t*>(&prm), sizeof(prm));
    if (vr.size == 0 || ir.size == 0 || ur.size == 0) return;
    const WaterPayload pl{
        .color = targets.color.Get(),
        .depth = targets.depth.Get(),
        .vtxOffset = vr.offset,
        .vtxSize = vr.size,
        .idxOffset = ir.offset,
        .idxCount = u32(g_idx.size()),
        .uniOffset = ur.offset,
        .width = targets.width,
        .height = targets.height,
    };
    aurora::gfx::push_custom_draw(g_drawType, &pl, sizeof(pl));
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
    j3dSys.getDrawBuffer(J3DSysDrawBuf_Xlu)->entryImm(&l_packets[l_packetNext++ & 3], 0);
}

void reset() {
    clear_all();
    g_map.clear();
}

bool hide_flat_ripples() {
    return enabled() && g_initialized && (g_fine.active || g_coarse.active);
}

}  // namespace dusk::water
