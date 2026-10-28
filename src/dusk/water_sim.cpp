// Dynamic water surface (see water_sim.h).
//
// A height field of N x N cells follows the player. Every contact the game reports
// (setWaterRipple / createWpillar) pushes the surface down; a damped wave equation then
// spreads the disturbance, reflects it off shores and lets crossing waves interfere.
// The original water geometry is flat, so instead of moving it the surface is drawn as a
// transparent sheet just above it that carries only what the waves change: the sun's
// highlight sliding along the crests, the sky reflection brightening and darkening with
// each slope, and foam where the surface breaks.

#include "dusk/water_sim.h"

#include "JSystem/J3DGraphBase/J3DDrawBuffer.h"
#include "JSystem/J3DGraphBase/J3DPacket.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DSys.h"
#include "d/d_bg_s_gnd_chk.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "f_op/f_op_camera_mng.h"
#include "dusk/settings.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dusk::water {
namespace {

constexpr int N = 160;            // cells per side
constexpr f32 DX = 9.0f;          // cell size in game units (~9 cm) -> 14.4 m square
constexpr int MASK_STEP = 4;      // one land/water sample per 4x4 cells
constexpr int MN = N / MASK_STEP; // mask samples per side
constexpr f32 WAVE_K = 0.24f;     // (c*dt/dx)^2 per substep, < 0.5 for stability
constexpr int SUBSTEPS = 2;       // per game frame (30 Hz logic)
constexpr f32 DAMP = 0.9935f;     // energy kept per substep
constexpr f32 SAME_WATER = 120.0f;// contacts farther than this from the active plane are other water
constexpr int IDLE_FRAMES = 240;  // frames without contact before the surface may sleep
constexpr int MASK_PER_FRAME = 160;

enum MaskState : u8 { MASK_UNKNOWN = 0, MASK_WATER = 1, MASK_LAND = 2 };

struct State {
    std::vector<f32> h, prev, next;
    std::vector<u8> mask;      // MN x MN
    f32 originX = 0.0f;        // world position of cell (0,0)
    f32 originZ = 0.0f;
    f32 waterY = 0.0f;
    bool active = false;
    int idle = 0;
    int maskCursor = 0;
    f32 energy = 0.0f;
    char stage[8] = {};

    State() : h(N * N, 0.0f), prev(N * N, 0.0f), next(N * N, 0.0f), mask(MN * MN, MASK_UNKNOWN) {}
} s;

bool enabled() {
    return dusk::getSettings().game.dynamicWater.getValue();
}

inline int idx(int x, int z) { return z * N + x; }

void clear_all() {
    std::fill(s.h.begin(), s.h.end(), 0.0f);
    std::fill(s.prev.begin(), s.prev.end(), 0.0f);
    std::fill(s.mask.begin(), s.mask.end(), MASK_UNKNOWN);
    s.active = false;
    s.idle = 0;
    s.energy = 0.0f;
    s.maskCursor = 0;
}

void center_on(f32 x, f32 z) {
    s.originX = std::floor(x / DX) * DX - (N / 2) * DX;
    s.originZ = std::floor(z / DX) * DX - (N / 2) * DX;
}

// Move the grid by whole mask blocks so the player stays near the middle; the part that
// leaves is dropped and the part that enters starts calm and unclassified.
void follow(f32 x, f32 z) {
    const f32 cx = s.originX + (N / 2) * DX;
    const f32 cz = s.originZ + (N / 2) * DX;
    const f32 limit = MASK_STEP * DX * 3;
    if (std::fabs(x - cx) < limit && std::fabs(z - cz) < limit) {
        return;
    }
    const int bx = int(std::round((x - cx) / (MASK_STEP * DX)));
    const int bz = int(std::round((z - cz) / (MASK_STEP * DX)));
    if (bx == 0 && bz == 0) {
        return;
    }
    const int dx = bx * MASK_STEP, dz = bz * MASK_STEP;
    auto shift = [&](std::vector<f32>& v) {
        std::vector<f32> out(N * N, 0.0f);
        for (int z = 0; z < N; z++) {
            const int sz = z + dz;
            if (sz < 0 || sz >= N) continue;
            for (int xx = 0; xx < N; xx++) {
                const int sx = xx + dx;
                if (sx < 0 || sx >= N) continue;
                out[idx(xx, z)] = v[idx(sx, sz)];
            }
        }
        v.swap(out);
    };
    shift(s.h);
    shift(s.prev);
    std::vector<u8> m(MN * MN, MASK_UNKNOWN);
    for (int z = 0; z < MN; z++) {
        const int sz = z + bz;
        if (sz < 0 || sz >= MN) continue;
        for (int xx = 0; xx < MN; xx++) {
            const int sx = xx + bx;
            if (sx < 0 || sx >= MN) continue;
            m[z * MN + xx] = s.mask[sz * MN + sx];
        }
    }
    s.mask.swap(m);
    s.maskCursor = 0;
    s.originX += dx * DX;
    s.originZ += dz * DX;
}

// Classify a few mask samples per frame: water where a water polygon sits at the active
// plane and no ground rises above it. Land cells hold the surface still, which makes
// waves reflect off shores, rocks and pillars.
// Mask samples ordered from the middle of the grid outwards, so the area around the
// player is classified first.
const std::vector<int>& mask_order() {
    static std::vector<int> order = [] {
        std::vector<int> o(MN * MN);
        for (int i = 0; i < MN * MN; i++) o[i] = i;
        auto d = [](int i) {
            const f32 x = (i % MN) - (MN - 1) * 0.5f, z = (i / MN) - (MN - 1) * 0.5f;
            return x * x + z * z;
        };
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return d(a) < d(b); });
        return o;
    }();
    return order;
}

void refine_mask() {
    dBgS_ObjGndChk_Wtr wtr;
    dBgS_ObjGndChk gnd;
    const std::vector<int>& order = mask_order();
    for (int n = 0; n < MASK_PER_FRAME; n++) {
        int i = -1;
        while (s.maskCursor < MN * MN) {
            const int c = order[s.maskCursor];
            if (s.mask[c] == MASK_UNKNOWN) {
                i = c;
                break;
            }
            s.maskCursor++;
        }
        if (i < 0) {
            return;
        }
        const f32 wx = s.originX + ((i % MN) * MASK_STEP + MASK_STEP * 0.5f) * DX;
        const f32 wz = s.originZ + ((i / MN) * MASK_STEP + MASK_STEP * 0.5f) * DX;
        cXyz top(wx, s.waterY + 300.0f, wz);
        wtr.SetPos(&top);
        const f32 wy = dComIfG_Bgsp().GroundCross(&wtr);
        bool water = std::fabs(wy - s.waterY) < 40.0f;
        if (water) {
            cXyz gtop(wx, s.waterY + 300.0f, wz);
            gnd.SetPos(&gtop);
            const f32 gy = dComIfG_Bgsp().GroundCross(&gnd);
            if (gy > s.waterY - 3.0f) {
                water = false;
            }
        }
        s.mask[i] = water ? MASK_WATER : MASK_LAND;
    }
}

// Simulate anything not known to be land (so the first waves after entering the water
// are not lost), but only draw where water has been confirmed.
inline bool cell_sim(int x, int z) {
    return s.mask[(z / MASK_STEP) * MN + (x / MASK_STEP)] != MASK_LAND;
}
inline bool cell_water(int x, int z) {
    return s.mask[(z / MASK_STEP) * MN + (x / MASK_STEP)] == MASK_WATER;
}

bool begin_contact(const cXyz& pos) {
    if (!enabled()) {
        return false;
    }
    if (!s.active) {
        clear_all();
        s.waterY = pos.y;
        center_on(pos.x, pos.z);
        s.active = true;
    } else if (std::fabs(pos.y - s.waterY) > SAME_WATER) {
        // Another body of water; take it over only once this one has gone quiet.
        if (s.idle < 20) {
            return false;
        }
        clear_all();
        s.waterY = pos.y;
        center_on(pos.x, pos.z);
        s.active = true;
    }
    s.idle = 0;
    return true;
}

// Press a smooth dip into the surface (a raised cosine, so no grid-sized spikes).
void push(f32 wx, f32 wz, f32 radius, f32 depth) {
    const f32 gx = (wx - s.originX) / DX, gz = (wz - s.originZ) / DX;
    const f32 r = std::max(radius / DX, 1.5f);
    const int x0 = std::max(1, int(gx - r)), x1 = std::min(N - 2, int(gx + r) + 1);
    const int z0 = std::max(1, int(gz - r)), z1 = std::min(N - 2, int(gz + r) + 1);
    for (int z = z0; z <= z1; z++) {
        for (int x = x0; x <= x1; x++) {
            const f32 d = std::sqrt((x - gx) * (x - gx) + (z - gz) * (z - gz)) / r;
            if (d >= 1.0f) continue;
            const f32 w = 0.5f + 0.5f * std::cos(d * 3.14159265f);
            s.h[idx(x, z)] -= depth * w;
        }
    }
}

void step() {
    for (int it = 0; it < SUBSTEPS; it++) {
        f32 e = 0.0f;
        for (int z = 1; z < N - 1; z++) {
            for (int x = 1; x < N - 1; x++) {
                const int i = idx(x, z);
                if (!cell_sim(x, z)) {
                    s.next[i] = 0.0f;
                    continue;
                }
                const f32 lap = s.h[i - 1] + s.h[i + 1] + s.h[i - N] + s.h[i + N] - 4.0f * s.h[i];
                f32 v = (2.0f * s.h[i] - s.prev[i] + WAVE_K * lap) * DAMP;
                // Soak up waves near the grid edge instead of bouncing them back.
                const int edge = std::min(std::min(x, N - 1 - x), std::min(z, N - 1 - z));
                if (edge < 10) {
                    v *= 0.90f + 0.01f * edge;
                }
                s.next[i] = v;
                e += v * v;
            }
        }
        s.prev.swap(s.h);
        s.h.swap(s.next);
        s.energy = e;
    }
}

class SurfacePacket : public J3DPacket {
public:
    void draw() override;
};
SurfacePacket l_packet;

struct Light {
    cXyz sun;   // direction to the light, normalized
    cXyz eye;
    f32 strength;  // dims at night and in dark places
    GXColor tint;
};

Light current_light() {
    Light l;
    dScnKy_env_light_c* env = dKy_getEnvlight();
    camera_process_class* cam = dComIfGp_getCamera(0);
    l.eye = cam != NULL ? cam->view.lookat.eye : cXyz(s.originX + (N / 2) * DX, s.waterY + 200.0f, s.originZ + (N / 2) * DX);
    cXyz d = env->sun_pos - l.eye;
    if (d.abs() < 1.0f || d.y < 0.05f * d.abs()) {
        d.set(0.3f, 1.0f, 0.2f);
    }
    l.sun = d.normZP();
    const f32 t = env->getDaytime();
    const bool day = t >= 90.0f && t < 285.0f;
    l.strength = day ? 1.0f : 0.45f;
    if (dKy_darkworld_check()) {
        l.strength *= 0.7f;
    }
    l.tint = day ? GXColor{255, 250, 236, 255} : GXColor{190, 210, 255, 255};
    return l;
}

void SurfacePacket::draw() {
    if (!s.active) {
        return;
    }
    const Light L = current_light();

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

    // Baseline: what a flat surface already shows, so only the waves' change is drawn.
    const f32 y = s.waterY + 0.6f;
    const cXyz mid(s.originX + (N / 2) * DX, y, s.originZ + (N / 2) * DX);
    const cXyz v0 = (L.eye - mid).normZP();
    const cXyz hv0 = (L.sun + v0).normZP();
    const f32 spec0 = std::pow(std::max(0.0f, hv0.y), 96.0f);
    const f32 fres0 = std::pow(1.0f - std::max(0.0f, v0.y), 5.0f);

    constexpr f32 SLOPE_GAIN = 0.55f;  // height units -> normal tilt
    auto shade = [&](int x, int z, f32 wx, f32 wz) -> GXColor {
        const int i = idx(x, z);
        const f32 gx = (s.h[i + 1] - s.h[i - 1]) * SLOPE_GAIN / (2.0f * DX) * 9.0f;
        const f32 gz = (s.h[i + N] - s.h[i - N]) * SLOPE_GAIN / (2.0f * DX) * 9.0f;
        cXyz n(-gx, 1.0f, -gz);
        n = n.normZP();
        cXyz v = cXyz(L.eye.x - wx, L.eye.y - y, L.eye.z - wz).normZP();
        const cXyz hv = (L.sun + v).normZP();
        const f32 spec = std::pow(std::max(0.0f, n.x * hv.x + n.y * hv.y + n.z * hv.z), 96.0f);
        const f32 nv = std::max(0.0f, n.x * v.x + n.y * v.y + n.z * v.z);
        const f32 fres = std::pow(1.0f - nv, 5.0f);
        const f32 slope = std::sqrt(gx * gx + gz * gz);
        // Crest brightness and trough shading from the sky reflection, sun glint on top.
        f32 d = (fres - fres0) * 1.6f + (spec - spec0) * 2.2f + (s.h[i] * -0.035f);
        f32 foam = std::max(0.0f, std::min(1.0f, (slope - 0.55f) * 1.6f));
        f32 a = std::min(1.0f, std::fabs(d)) * 0.85f + foam * 0.6f;
        // Fade at the grid edge so it never ends in a visible line.
        const int edge = std::min(std::min(x, N - 1 - x), std::min(z, N - 1 - z));
        if (edge < 16) a *= edge / 16.0f;
        a *= L.strength;
        if (!cell_water(x, z)) a = 0.0f;
        GXColor c;
        if (d >= 0.0f || foam > 0.0f) {
            const f32 m = 0.8f + 0.2f * foam;
            c = GXColor{u8(L.tint.r * m), u8(L.tint.g * m), u8(L.tint.b * m), 0};
        } else {
            c = GXColor{8, 22, 30, 0};
        }
        c.a = u8(std::min(255.0f, a * 255.0f));
        return c;
    };

    for (int z = 1; z < N - 2; z++) {
        GXBegin(GX_TRIANGLESTRIP, GX_VTXFMT0, (N - 2) * 2);
        for (int x = 1; x < N - 1; x++) {
            const f32 wx = s.originX + x * DX;
            const f32 wz0 = s.originZ + z * DX;
            const f32 wz1 = wz0 + DX;
            const GXColor c0 = shade(x, z, wx, wz0);
            const GXColor c1 = shade(x, z + 1, wx, wz1);
            GXPosition3f32(wx, y, wz0);
            GXColor4u8(c0.r, c0.g, c0.b, c0.a);
            GXPosition3f32(wx, y, wz1);
            GXColor4u8(c1.r, c1.g, c1.b, c1.a);
        }
        GXEnd();
    }
    J3DShape::resetVcdVatCache();
}

}  // namespace

void ripple(const cXyz& pos, f32 speed) {
    if (!begin_contact(pos)) {
        return;
    }
    // `speed` is how far the actor moved this frame (Link walking ~5-10, running ~20,
    // Epona galloping 40+). Standing still barely stirs the water; moving drives a wake.
    const f32 v = std::max(0.0f, speed);
    const f32 depth = std::min(0.06f + 0.12f * v, 3.0f);
    const f32 radius = 14.0f + std::min(16.0f, 0.5f * v);
    push(pos.x, pos.z, radius, depth);
}

void splash(const cXyz& pos, f32 scale) {
    if (!begin_contact(pos)) {
        return;
    }
    const f32 sc = std::max(0.3f, std::min(scale, 4.0f));
    push(pos.x, pos.z, 38.0f * sc, 9.0f * sc);
}

void update() {
    if (!s.active) {
        return;
    }
    if (!enabled()) {
        clear_all();
        return;
    }
    const char* stage = dComIfGp_getStartStageName();
    if (stage != NULL && std::strncmp(stage, s.stage, sizeof(s.stage)) != 0) {
        std::strncpy(s.stage, stage, sizeof(s.stage));
        clear_all();
        return;
    }
    if (fopAc_ac_c* pl = dComIfGp_getPlayer(0)) {
        follow(pl->current.pos.x, pl->current.pos.z);
    }
    refine_mask();
    step();
    s.idle++;
    if (s.idle > IDLE_FRAMES && s.energy < 0.02f) {
        clear_all();
    }
}

void draw() {
    if (!s.active || !enabled()) {
        return;
    }
    // Translucent object list: drawn after the stage's water (XluListBG) and its particles.
    dComIfGd_setList();
    j3dSys.getDrawBuffer(J3DSysDrawBuf_Xlu)->entryImm(&l_packet, 0);
}

void reset() {
    clear_all();
}

bool hide_flat_ripples() {
    return enabled() && s.active;
}

}  // namespace dusk::water
