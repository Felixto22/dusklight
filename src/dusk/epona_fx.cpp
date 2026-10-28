// Epona's hair and coat (see epona_fx.h).

#include "dusk/epona_fx.h"

#include "JSystem/J3DGraphAnimator/J3DModel.h"
#include "JSystem/J3DGraphAnimator/J3DModelData.h"
#include "JSystem/J3DGraphBase/J3DDrawBuffer.h"
#include "JSystem/J3DGraphBase/J3DMaterial.h"
#include "JSystem/J3DGraphBase/J3DPacket.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DShapeMtx.h"
#include "JSystem/J3DGraphBase/J3DSys.h"
#include "JSystem/J3DGraphBase/J3DTexture.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "d/d_kankyo_tev_str.h"
#include "d/d_kankyo_wether.h"
#include "f_op/f_op_camera_mng.h"
#include "dusk/settings.h"
#include "m_Do/m_Do_mtx.h"

#include <aurora/dl.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dusk::epona {
namespace {

#include "dusk/epona_data.inc"
#include "dusk/epona_groom.inc"

bool hair_enabled() { return dusk::getSettings().game.eponaHairPhysics.getValue(); }
bool fur_enabled() { return dusk::getSettings().game.eponaFur.getValue(); }
bool groom_enabled() { return dusk::getSettings().game.eponaRealMane.getValue(); }

u32 g_frame = 0;

struct V3 {
    f32 x, y, z;
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(f32 s) const { return {x * s, y * s, z * s}; }
    V3& operator+=(const V3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    V3& operator-=(const V3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
};
inline f32 dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(const V3& a, const V3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline f32 len(const V3& a) { return std::sqrt(dot(a, a)); }
inline V3 norm(const V3& a) {
    const f32 l = len(a);
    return l > 1e-6f ? a * (1.0f / l) : V3{0.0f, 1.0f, 0.0f};
}
inline V3 xform(MtxP m, const f32 p[3]) {
    return {m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3]};
}
inline V3 rotate(MtxP m, const f32 v[3]) {
    return {m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]};
}
// World vector into a joint's local frame (rotation part transposed).
inline V3 to_local(MtxP m, const V3& v) {
    return {m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
            m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
            m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z};
}

V3 wind_at(const V3& p) {
    cXyz pos(p.x, p.y, p.z), dir(0.0f, 0.0f, 0.0f);
    f32 pow = 0.0f;
    dKyw_get_AllWind_vec(&pos, &dir, &pow);
    return V3{dir.x, dir.y, dir.z} * pow;
}

// ---------------------------------------------------------------------------------------
// Joint springs

struct BoneParams {
    f32 stiff;  // pull back to the animated pose, per frame
    f32 damp;   // velocity lost per frame
    f32 gain;   // radians per (unit/frame^2) of felt acceleration
    f32 wind;   // radians per unit of wind
    f32 limit;  // max deflection, radians
};

struct Bone {
    int joint = -1;
    V3 dir{1.0f, 0.0f, 0.0f};
    BoneParams prm{};
    V3 th{}, w{};   // rotation vector and its rate, joint-local
    V3 p1{}, p2{};  // pivot world position one and two frames ago
    int seen = 0;
    u32 frame = 0;
};

Bone g_bones[10];
int g_boneCount = 0;

BoneParams params_for(int joint) {
    // Joint numbers from hs.bmd: 13 hair_L, 14 hair_R, 18 hair_F, 22/24 belts,
    // 23/25 stirrups, 35-37 tail.
    switch (joint) {
    case 13: case 14: return {0.10f, 0.20f, 0.011f, 0.12f, 0.55f};
    case 18:          return {0.14f, 0.22f, 0.010f, 0.10f, 0.45f};
    case 35:          return {0.08f, 0.14f, 0.010f, 0.10f, 0.60f};
    case 36: case 37: return {0.06f, 0.12f, 0.012f, 0.14f, 0.75f};
    case 22: case 24: return {0.16f, 0.22f, 0.008f, 0.02f, 0.40f};
    default:          return {0.12f, 0.18f, 0.012f, 0.03f, 0.60f};
    }
}

void init_bones() {
    if (g_boneCount != 0) {
        return;
    }
    for (const EponaHang& h : kEponaHang) {
        Bone& b = g_bones[g_boneCount++];
        b.joint = h.joint;
        b.dir = norm(V3{h.dir[0], h.dir[1], h.dir[2]});
        b.prm = params_for(h.joint);
    }
}

Bone* bone_for(int joint) {
    for (int i = 0; i < g_boneCount; i++) {
        if (g_bones[i].joint == joint) {
            return &g_bones[i];
        }
    }
    return nullptr;
}

void step_bone(Bone& b, MtxP m) {
    const V3 pivot{m[0][3], m[1][3], m[2][3]};
    if (b.seen >= 2 && len(pivot - b.p1) < 150.0f) {
        // What the hair feels is the opposite of how its root was just accelerated.
        const V3 acc = pivot - b.p1 * 2.0f + b.p2;
        V3 f = to_local(m, acc * -1.0f) * b.prm.gain + to_local(m, wind_at(pivot)) * b.prm.wind;
        V3 tq = cross(b.dir, f);
        tq -= b.dir * dot(tq, b.dir);  // no twist around the hair itself
        b.w += tq;
        b.w -= b.th * b.prm.stiff;
        b.w = b.w * (1.0f - b.prm.damp);
        b.th += b.w;
        const f32 a = len(b.th);
        if (a > b.prm.limit) {
            b.th = b.th * (b.prm.limit / a);
            b.w = b.w * 0.5f;
        }
    } else {
        b.th = {};
        b.w = {};
    }
    b.p2 = b.seen >= 1 ? b.p1 : pivot;
    b.p1 = pivot;
    b.seen = std::min(b.seen + 1, 2);
}

void apply_rotation(MtxP m, const V3& th) {
    const f32 a = len(th);
    if (a < 1e-5f) {
        return;
    }
    const V3 k = th * (1.0f / a);
    const f32 c = std::cos(a), s = std::sin(a), t = 1.0f - c;
    // Rotation about the joint's own origin, in its local frame: m = m * R.
    const f32 r[3][3] = {
        {t * k.x * k.x + c, t * k.x * k.y - s * k.z, t * k.x * k.z + s * k.y},
        {t * k.x * k.y + s * k.z, t * k.y * k.y + c, t * k.y * k.z - s * k.x},
        {t * k.x * k.z - s * k.y, t * k.y * k.z + s * k.x, t * k.z * k.z + c},
    };
    f32 out[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            out[i][j] = m[i][0] * r[0][j] + m[i][1] * r[1][j] + m[i][2] * r[2][j];
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m[i][j] = out[i][j];
        }
    }
}

bool model_matches(J3DModel* model) {
    J3DModelData* data = model->getModelData();
    return data->getJointNum() == 38 && data->getVertexData().getVtxNum() == u32(kEponaPosCount);
}

// ---------------------------------------------------------------------------------------
// Strands

constexpr int kManeSeg = 6;
constexpr int kTailSeg = 8;
constexpr int kMaxSeg = 8;

struct Strand {
    const EponaAnchor* anchor;
    f32 jitter[3];  // root offset, joint-local
    int seg;
    f32 segLen;
    f32 bend;       // pull toward the rest direction per frame
    f32 width0, width1;
    bool tail;
    V3 p[kMaxSeg + 1], q[kMaxSeg + 1];  // positions and previous positions
};

std::vector<Strand> g_strands;
bool g_strandsReady = false;
u32 g_strandFrame = 0;

u32 g_rng = 0x9E3779B9u;
f32 rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (g_rng & 0xFFFFFF) / f32(0x1000000);
}

void build_strands() {
    g_strands.clear();
    for (const EponaAnchor& a : kEponaManeAnchors) {
        const bool forelock = a.joint == 15;  // head
        for (int k = 0; k < 2; k++) {
            Strand s{};
            s.anchor = &a;
            for (int c = 0; c < 3; c++) s.jitter[c] = (rnd() - 0.5f) * 3.0f;
            s.seg = kManeSeg;
            s.segLen = (forelock ? 3.4f : 6.2f) * (0.8f + 0.4f * rnd());
            s.bend = forelock ? 0.10f : 0.06f;
            s.width0 = 1.4f;
            s.width1 = 0.35f;
            s.tail = false;
            g_strands.push_back(s);
        }
    }
    for (const EponaAnchor& a : kEponaTailAnchors) {
        const bool tip = a.joint == 37;
        for (int k = 0; k < 3; k++) {
            Strand s{};
            s.anchor = &a;
            for (int c = 0; c < 3; c++) s.jitter[c] = (rnd() - 0.5f) * 4.0f;
            s.seg = kTailSeg;
            s.segLen = (tip ? 11.0f : 8.0f) * (0.75f + 0.5f * rnd());
            s.bend = 0.025f;
            s.width0 = 1.7f;
            s.width1 = 0.45f;
            s.tail = true;
            g_strands.push_back(s);
        }
    }
}

struct Root {
    V3 pos, nrm, axis;
};

Root root_of(J3DModel* model, const Strand& s) {
    MtxP m = model->getAnmMtx(s.anchor->joint);
    const f32 lp[3] = {s.anchor->pos[0] + s.jitter[0], s.anchor->pos[1] + s.jitter[1],
                       s.anchor->pos[2] + s.jitter[2]};
    const f32 ax[3] = {1.0f, 0.0f, 0.0f};
    return {xform(m, lp), norm(rotate(m, s.anchor->nrm)), norm(rotate(m, ax))};
}

V3 rest_dir(const Strand& s, const Root& r) {
    if (s.tail) {
        // Along the tail bone, falling a little.
        return norm(r.axis + V3{0.0f, -0.45f, 0.0f});
    }
    // Mane hangs down the side of the neck, slightly away from it.
    return norm(r.nrm * 0.55f + V3{0.0f, -1.0f, 0.0f});
}

void place_strand(Strand& s, const Root& r) {
    const V3 d = rest_dir(s, r);
    for (int i = 0; i <= s.seg; i++) {
        s.p[i] = r.pos + d * (s.segLen * i);
        s.q[i] = s.p[i];
    }
}

void simulate_strands(J3DModel* model) {
    if (!g_strandsReady) {
        build_strands();
        for (Strand& s : g_strands) place_strand(s, root_of(model, s));
        g_strandsReady = true;
        return;
    }
    const V3 gravity{0.0f, -0.55f, 0.0f};
    for (Strand& s : g_strands) {
        const Root r = root_of(model, s);
        if (len(r.pos - s.p[0]) > 200.0f) {
            place_strand(s, r);  // warped or reloaded: start over instead of whipping
            continue;
        }
        const V3 wind = wind_at(r.pos) * (s.tail ? 0.9f : 0.6f);
        const V3 d = rest_dir(s, r);
        s.q[0] = s.p[0];
        s.p[0] = r.pos;
        for (int i = 1; i <= s.seg; i++) {
            const V3 v = (s.p[i] - s.q[i]) * 0.93f;
            s.q[i] = s.p[i];
            s.p[i] += v + gravity + wind;
        }
        for (int it = 0; it < 3; it++) {
            for (int i = 1; i <= s.seg; i++) {
                // Keep each segment its length.
                V3 e = s.p[i] - s.p[i - 1];
                const f32 l = len(e);
                if (l > 1e-4f) {
                    const V3 corr = e * ((l - s.segLen) / l);
                    if (i == 1) {
                        s.p[i] -= corr;
                    } else {
                        s.p[i] -= corr * 0.5f;
                        s.p[i - 1] += corr * 0.5f;
                    }
                }
                // Hair keeps a little shape: lean toward the rest direction.
                const V3 target = s.p[i - 1] + d * s.segLen;
                s.p[i] += (target - s.p[i]) * s.bend;
                // Stay on the outside of the body at the root.
                const V3 o = s.p[i] - r.pos;
                const f32 below = dot(o, r.nrm);
                if (below < 0.0f) {
                    s.p[i] -= r.nrm * below;
                }
            }
            s.p[0] = r.pos;
        }
    }
}

class StrandPacket : public J3DPacket {
public:
    void draw() override;
    GXColor light{255, 255, 255, 255};
};
StrandPacket l_strandPacket;

void StrandPacket::draw() {
    if (g_strands.empty()) {
        return;
    }
    camera_process_class* cam = dComIfGp_getCamera(0);
    if (cam == NULL) {
        return;
    }
    const V3 eye{cam->view.lookat.eye.x, cam->view.lookat.eye.y, cam->view.lookat.eye.z};

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

    for (const Strand& s : g_strands) {
        // Epona's mane and tail are a pale cream; roots sit in shadow.
        const f32 cr = s.tail ? 0.92f : 1.0f;
        GXBegin(GX_TRIANGLESTRIP, GX_VTXFMT0, (s.seg + 1) * 2);
        for (int i = 0; i <= s.seg; i++) {
            const f32 t = f32(i) / s.seg;
            const V3 a = s.p[std::max(0, i - 1)], b = s.p[std::min(s.seg, i + 1)];
            const V3 dir = norm(b - a);
            const V3 side = norm(cross(dir, eye - s.p[i])) * (0.5f * (s.width0 + (s.width1 - s.width0) * t));
            const f32 shade = (0.62f + 0.38f * t) * cr;
            const u8 r = u8(std::min(255.0f, light.r * shade * 0.93f));
            const u8 g = u8(std::min(255.0f, light.g * shade * 0.89f));
            const u8 bl = u8(std::min(255.0f, light.b * shade * 0.80f));
            const u8 al = u8(230.0f * (1.0f - t * t));
            const V3 p0 = s.p[i] - side, p1 = s.p[i] + side;
            GXPosition3f32(p0.x, p0.y, p0.z);
            GXColor4u8(r, g, bl, al);
            GXPosition3f32(p1.x, p1.y, p1.z);
            GXColor4u8(r, g, bl, al);
        }
        GXEnd();
    }
    J3DShape::resetVcdVatCache();
}


// ---------------------------------------------------------------------------------------
// Real mane and tail
//
// The original mane and tail are a few flat cards with hair painted on them. With this
// option they are hidden (their triangles are collapsed in the model's display lists) and
// replaced by ~2400 locks of hair grown along those same cards: each lock follows the
// painted hair from root to tip, takes its colours from the original texture, is skinned
// to the same joints (so it follows the animation and the hair-bone springs), and then
// swings on its own with verlet physics: inertia, gravity, wind, and the pull back to
// its groomed shape getting weaker toward the tip. Each lock is drawn as a ribbon of
// eight fine hairs lit along the hair (bright sheen across the strands, darker roots).

constexpr int kGroomMaxPts = 12;

struct GroomState {
    V3 p[kGroomMaxPts], q[kGroomMaxPts];
    f32 rest[kGroomMaxPts];
    bool placed = false;
};
std::vector<GroomState> g_groom;
u32 g_groomFrame = 0;

V3 groom_target(J3DModel* model, const GroomPoint& gp) {
    const f32 a[3] = {gp.a[0] * kGroomScale, gp.a[1] * kGroomScale, gp.a[2] * kGroomScale};
    const V3 pa = xform(model->getAnmMtx(gp.j0), a);
    if (gp.w0 >= 255) {
        return pa;
    }
    const f32 b[3] = {gp.b[0] * kGroomScale, gp.b[1] * kGroomScale, gp.b[2] * kGroomScale};
    const V3 pb = xform(model->getAnmMtx(gp.j1), b);
    const f32 w = gp.w0 / 255.0f;
    return pa * w + pb * (1.0f - w);
}

void simulate_groom(J3DModel* model) {
    if (g_groom.size() != size_t(kGroomStrandCount)) {
        g_groom.assign(kGroomStrandCount, GroomState{});
    }
    const V3 gravity{0.0f, -0.16f, 0.0f};
    const V3 windHere = wind_at(groom_target(model, kGroomPoints[0]));
    for (int si = 0; si < kGroomStrandCount; si++) {
        const GroomStrand& st = kGroomStrands[si];
        GroomState& g = g_groom[si];
        const int n = std::min<int>(st.count, kGroomMaxPts);
        const bool tail = st.group == 1;
        V3 t[kGroomMaxPts];
        for (int k = 0; k < n; k++) t[k] = groom_target(model, kGroomPoints[st.first + k]);
        if (!g.placed || len(t[0] - g.p[0]) > 200.0f) {
            for (int k = 0; k < n; k++) {
                g.p[k] = g.q[k] = t[k];
                g.rest[k] = k == 0 ? 0.0f : len(t[k] - t[k - 1]);
            }
            g.placed = true;
            continue;
        }
        const V3 wind = windHere * (tail ? 0.35f : 0.25f);
        g.q[0] = g.p[0];
        g.p[0] = t[0];
        for (int k = 1; k < n; k++) {
            const f32 u = f32(k) / (n - 1);
            const V3 v = (g.p[k] - g.q[k]) * 0.90f;
            g.q[k] = g.p[k];
            g.p[k] += v + gravity + wind * u;
            // Pull back toward the groomed shape: firm at the root, loose at the tip.
            const f32 stiff = tail ? 0.42f - 0.37f * u : 0.50f - 0.42f * u;
            g.p[k] += (t[k] - g.p[k]) * stiff;
        }
        // Keep every segment its length, following the root.
        for (int k = 1; k < n; k++) {
            const V3 e = g.p[k] - g.p[k - 1];
            const f32 l = len(e);
            if (l > 1e-4f) {
                g.p[k] = g.p[k - 1] + e * (g.rest[k] / l);
            }
        }
    }
}

// Hairs across each lock: 8 hairs of slightly different shade and length (I8, GX tiles).
alignas(32) u8 g_hairTex[32 * 64];
GXTexObj g_hairObj;
bool g_hairTexReady = false;

void build_hair_tex() {
    u32 seed = 0xA5A5F00Du;
    auto r = [&] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / f32(1 << 24);
    };
    static u8 linear[64][32];
    for (int hair = 0; hair < 8; hair++) {
        const f32 shade = 0.72f + 0.28f * r();
        const f32 end = 0.78f + 0.22f * r();
        const f32 wob = (r() - 0.5f) * 0.9f;
        for (int y = 0; y < 64; y++) {
            const f32 t = (y + 0.5f) / 64.0f;
            const f32 thin = t < end ? 1.0f - 0.55f * (t / end) * (t / end) : 0.0f;
            for (int x = 0; x < 4; x++) {
                const f32 c = std::fabs((x + 0.5f + wob * t) - 2.0f) / 2.0f;  // 0 centre .. 1 edge
                const f32 v = std::max(0.0f, 1.0f - c / std::max(0.2f, thin)) * shade;
                linear[y][hair * 4 + x] = u8(std::clamp(v * 255.0f, 0.0f, 255.0f));
            }
        }
    }
    int o = 0;
    for (int by = 0; by < 64; by += 4) {
        for (int bx = 0; bx < 32; bx += 8) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 8; x++) {
                    g_hairTex[o++] = linear[by + y][bx + x];
                }
            }
        }
    }
    GXInitTexObj(&g_hairObj, g_hairTex, 32, 64, GX_TF_I8, GX_CLAMP, GX_CLAMP, GX_FALSE);
    GXInitTexObjLOD(&g_hairObj, GX_LINEAR, GX_LINEAR, 0.0f, 0.0f, 0.0f, GX_FALSE, GX_FALSE,
                    GX_ANISO_1);
    g_hairTexReady = true;
}

// Collapse (or restore) the triangles of the original cards in the shared model data.
struct DlBackup {
    J3DShapeDraw* draw;
    std::vector<u8> bytes;
};
std::vector<DlBackup> g_dlBackup;
J3DModelData* g_dlData = nullptr;
bool g_dlHidden = false;

bool in_list(u16 v, const u16* list, int n) {
    for (int i = 0; i < n; i++) {
        if (list[i] == v) return true;
    }
    return false;
}

void hide_cards(J3DShape* shape, const u16* list, int n) {
    namespace dl = aurora::gx::dl;
    for (u16 g = 0; g < shape->getMtxGroupNum(); g++) {
        J3DShapeDraw* sd = shape->getShapeDraw(g);
        if (sd == nullptr) continue;
        u8* data = sd->getDisplayList();
        const u32 size = sd->getDisplayListSize();
        g_dlBackup.push_back({sd, std::vector<u8>(data, data + size)});
        dl::Reader reader{data, size, shape->getVtxDesc()};
        while (const auto cmd = reader.next()) {
            if (cmd->kind == dl::Command::Kind::Passthrough) continue;
            const dl::DrawCmd& d = cmd->draw;
            const u32 stride = d.layout->stride;
            auto hidden = [&](u32 v) { return in_list(d.attr_idx(v, GX_VA_POS), list, n); };
            if (cmd->kind == dl::Command::Kind::DrawIndexed) {
                u8* idx = const_cast<u8*>(d.indices);
                for (u32 i = 0; i + 2 < d.indexCount; i += 3) {
                    const u16 a = d.index(i), b = d.index(i + 1), c = d.index(i + 2);
                    if (hidden(a) || hidden(b) || hidden(c)) {
                        std::memcpy(idx + (i + 1) * 2, idx + i * 2, 2);
                        std::memcpy(idx + (i + 2) * 2, idx + i * 2, 2);
                    }
                }
                continue;
            }
            u8* verts = const_cast<u8*>(d.vertices);
            if (d.prim == GX_TRIANGLES) {
                for (u32 i = 0; i + 2 < d.vtxCount; i += 3) {
                    if (hidden(i) || hidden(i + 1) || hidden(i + 2)) {
                        std::memcpy(verts + (i + 1) * stride, verts + i * stride, stride);
                        std::memcpy(verts + (i + 2) * stride, verts + i * stride, stride);
                    }
                }
            } else {
                bool any = false;
                for (u32 i = 0; i < d.vtxCount && !any; i++) any = hidden(i);
                if (any) {
                    for (u32 i = 1; i < d.vtxCount; i++) {
                        std::memcpy(verts + i * stride, verts, stride);
                    }
                }
            }
        }
        DCStoreRange(data, size);
    }
}

void restore_cards() {
    for (DlBackup& b : g_dlBackup) {
        std::memcpy(b.draw->getDisplayList(), b.bytes.data(), b.bytes.size());
        DCStoreRange(b.draw->getDisplayList(), b.bytes.size());
    }
    g_dlBackup.clear();
    g_dlHidden = false;
}

void set_cards_hidden(J3DModel* model, bool hide) {
    J3DModelData* data = model->getModelData();
    if (data != g_dlData) {
        // New model data (archive reloaded): the old backups point at freed memory.
        g_dlBackup.clear();
        g_dlHidden = false;
        g_dlData = data;
    }
    if (hide == g_dlHidden) return;
    if (!hide) {
        restore_cards();
        return;
    }
    if (data->getShapeNum() < 3) return;
    hide_cards(data->getShapeNodePointer(0), kGroomHideTail,
               int(sizeof(kGroomHideTail) / sizeof(kGroomHideTail[0])));
    hide_cards(data->getShapeNodePointer(2), kGroomHideMane,
               int(sizeof(kGroomHideMane) / sizeof(kGroomHideMane[0])));
    g_dlHidden = true;
}

class GroomPacket : public J3DPacket {
public:
    void draw() override;
    GXColorS10 light{255, 255, 255, 255};
};
GroomPacket l_groomPacket;

inline f32 hash01(u32 x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (x & 0xFFFFFF) / f32(0x1000000);
}

void GroomPacket::draw() {
    if (g_groom.size() != size_t(kGroomStrandCount) || !g_hairTexReady) {
        return;
    }
    camera_process_class* cam = dComIfGp_getCamera(0);
    if (cam == NULL) {
        return;
    }
    const V3 eye{cam->view.lookat.eye.x, cam->view.lookat.eye.y, cam->view.lookat.eye.z};
    dScnKy_env_light_c* env = dKy_getEnvlight();
    const V3 sun{env->sun_pos.x, env->sun_pos.y, env->sun_pos.z};

    j3dSys.reinitGX();
    GXSetNumIndStages(0);
    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
    GXSetVtxDesc(GX_VA_CLR0, GX_DIRECT);
    GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);
    GXSetNumChans(1);
    GXSetChanCtrl(GX_COLOR0A0, GX_DISABLE, GX_SRC_REG, GX_SRC_VTX, GX_LIGHT_NULL, GX_DF_NONE,
                  GX_AF_NONE);
    GXLoadTexObj(&g_hairObj, GX_TEXMAP0);
    GXSetNumTexGens(1);
    GXSetTexCoordGen(GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY);
    GXSetNumTevStages(1);
    GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
    GXSetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO);
    GXSetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_2, GX_TRUE, GX_TEVPREV);
    GXSetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA);
    GXSetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXLoadPosMtxImm(j3dSys.getViewMtx(), GX_PNMTX0);
    GXSetCurrentMtx(GX_PNMTX0);
    GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_SET);
    GXSetCullMode(GX_CULL_NONE);
    GXSetZMode(GX_ENABLE, GX_LEQUAL, GX_ENABLE);
    GXSetAlphaCompare(GX_GREATER, 70, GX_AOP_AND, GX_ALWAYS, 0);
    GXSetFog(GX_FOG_NONE, 0.0f, 0.0f, 0.0f, 0.0f, GXColor{0, 0, 0, 0});
    J3DDifferedTexMtx::sTexGenBlock = NULL;

    const f32 lr = light.r / 255.0f, lg = light.g / 255.0f, lb = light.b / 255.0f;
    for (int si = 0; si < kGroomStrandCount; si++) {
        const GroomStrand& st = kGroomStrands[si];
        const GroomState& g = g_groom[si];
        const int n = std::min<int>(st.count, kGroomMaxPts);
        if (!g.placed || n < 2) continue;
        const bool tail = st.group == 1;
        const f32 h = hash01(u32(si) * 2654435761u);
        const f32 w0 = (tail ? 2.6f : 2.2f) * (0.8f + 0.4f * h);
        const f32 w1 = w0 * 0.42f;
        const f32 sheen = 0.85f + 0.3f * hash01(u32(si) * 40503u + 7u);
        const f32 s0 = 0.0f, s1 = 1.0f;
        GXBegin(GX_TRIANGLESTRIP, GX_VTXFMT0, u16(n * 2));
        for (int k = 0; k < n; k++) {
            const f32 t = f32(k) / (n - 1);
            const V3 a = g.p[std::max(0, k - 1)], b = g.p[std::min(n - 1, k + 1)];
            const V3 dir = norm(b - a);
            const V3 view = norm(eye - g.p[k]);
            const V3 side = norm(cross(dir, view)) * (0.5f * (w0 + (w1 - w0) * t));
            // Lit along the hair: diffuse from the angle to the light, a sheen where the
            // half vector is across the strands.
            const V3 L = norm(sun - g.p[k]);
            const V3 H = norm(L + view);
            const f32 tl = dot(dir, L), th = dot(dir, H);
            const f32 diff = 0.55f + 0.45f * std::sqrt(std::max(0.0f, 1.0f - tl * tl));
            const f32 spec = std::pow(std::sqrt(std::max(0.0f, 1.0f - th * th)), 48.0f) * 0.30f * sheen;
            const f32 ao = 0.62f + 0.38f * std::min(1.0f, t * 1.6f);
            const u32 rgb = kGroomPoints[st.first + k].rgb;
            const f32 cr = ((rgb >> 16) & 0xFF) / 255.0f, cg = ((rgb >> 8) & 0xFF) / 255.0f,
                      cb = (rgb & 0xFF) / 255.0f;
            const f32 m = diff * ao * 0.62f;  // x2 in the TEV
            const u8 r = u8(std::clamp((cr * m + spec * 0.5f) * lr * 255.0f, 0.0f, 255.0f));
            const u8 gg = u8(std::clamp((cg * m + spec * 0.48f) * lg * 255.0f, 0.0f, 255.0f));
            const u8 bb = u8(std::clamp((cb * m + spec * 0.44f) * lb * 255.0f, 0.0f, 255.0f));
            const V3 p0 = g.p[k] - side, p1 = g.p[k] + side;
            GXPosition3f32(p0.x, p0.y, p0.z);
            GXColor4u8(r, gg, bb, 255);
            GXTexCoord2f32(s0, t);
            GXPosition3f32(p1.x, p1.y, p1.z);
            GXColor4u8(r, gg, bb, 255);
            GXTexCoord2f32(s1, t);
        }
        GXEnd();
    }
    J3DShape::resetVcdVatCache();
    j3dSys.reinitGX();
}

// ---------------------------------------------------------------------------------------
// Shell fur

// The coat length comes from the "Epona Fur Length" setting (1-10): 1-3 a short real
// horse coat, higher values a longer, fluffier plush coat in pointed tufts.
constexpr int kMaxShells = 26;

struct FurStyle {
    int level = 0;
    int shells = 8;
    f32 length = 1.6f;   // game units (1 unit ~ 1 cm)
    f32 density = 34.0f; // pattern repeats across the body texture
    f32 rootShade = 0.78f;
    f32 tipShade = 1.08f;
    bool tufts = false;
};

FurStyle fur_style() {
    const int v = std::clamp(dusk::getSettings().game.eponaFurLength.getValue(), 1, 10);
    FurStyle f;
    f.level = v;
    const f32 t = (v - 1) / 9.0f;  // 0 short .. 1 very fluffy
    f.shells = 8 + int(std::round(18.0f * t));
    f.length = 1.2f + 8.8f * t * t + 1.4f * t;
    f.density = 34.0f / (1.0f + 2.4f * t);
    f.rootShade = 0.78f - 0.40f * t;  // deep coat gets dark between the tufts
    f.tipShade = 1.08f + 0.12f * t;   // bright fuzzy tips
    f.tufts = v >= 4;
    return f;
}

std::vector<Vec> g_shellPos[kMaxShells];
int g_shellLevel = 0;
alignas(32) u8 g_noise[64 * 64];
alignas(32) u8 g_tufts[64 * 64];
GXTexObj g_noiseObj, g_tuftObj;
bool g_noiseReady = false;

void to_tiles(const u8* linear, u8* out) {
    // I8 texture in the GameCube tile layout (8x4 texel blocks).
    int o = 0;
    for (int by = 0; by < 64; by += 4) {
        for (int bx = 0; bx < 64; bx += 8) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 8; x++) {
                    out[o++] = linear[(by + y) * 64 + bx + x];
                }
            }
        }
    }
}

void build_noise() {
    u32 seed = 0x1234567u;
    auto r = [&] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / f32(1 << 24);
    };
    static u8 linear[64 * 64];
    for (int i = 0; i < 64 * 64; i++) {
        const f32 v = r();
        linear[i] = v < 0.6f ? u8(70 + 185 * r()) : 0;  // most texels grow a hair
    }
    to_tiles(linear, g_noise);

    // Tufts: overlapping cones (wrapping around the tile). Each shell cuts the cones
    // higher up, so every tuft narrows to a soft point, like a plush or a squirrel's coat.
    static f32 acc[64 * 64];
    std::fill(acc, acc + 64 * 64, 0.0f);
    for (int c = 0; c < 340; c++) {
        const f32 cx = r() * 64.0f, cz = r() * 64.0f;
        const f32 rad = 2.2f + r() * 3.2f;
        const f32 peak = 0.55f + 0.45f * r();
        const int x0 = int(std::floor(cx - rad)), x1 = int(std::ceil(cx + rad));
        const int z0 = int(std::floor(cz - rad)), z1 = int(std::ceil(cz + rad));
        for (int z = z0; z <= z1; z++) {
            for (int x = x0; x <= x1; x++) {
                const f32 d = std::sqrt((x - cx) * (x - cx) + (z - cz) * (z - cz)) / rad;
                if (d >= 1.0f) continue;
                f32& a = acc[(z & 63) * 64 + (x & 63)];
                a = std::max(a, peak * (1.0f - d));
            }
        }
    }
    for (int i = 0; i < 64 * 64; i++) {
        // A little per-texel grain so the tuft edges read as separate hairs.
        const f32 v = acc[i] * (0.82f + 0.18f * r());
        linear[i] = u8(std::clamp(v * 255.0f, 0.0f, 255.0f));
    }
    to_tiles(linear, g_tufts);

    for (GXTexObj* obj : {&g_noiseObj, &g_tuftObj}) {
        GXInitTexObj(obj, obj == &g_noiseObj ? g_noise : g_tufts, 64, 64, GX_TF_I8, GX_REPEAT,
                     GX_REPEAT, GX_FALSE);
        GXInitTexObjLOD(obj, GX_LINEAR, GX_LINEAR, 0.0f, 0.0f, 0.0f, GX_FALSE, GX_FALSE,
                        GX_ANISO_1);
    }
    g_noiseReady = true;
}

void build_shells(J3DModelData* data, const FurStyle& f) {
    const Vec* base = (const Vec*)data->getVertexData().getVtxPosArray();
    for (int k = 0; k < f.shells; k++) {
        g_shellPos[k].resize(kEponaPosCount);
        // Shells packed closer near the skin, where the coat is densest.
        const f32 u = f32(k + 1) / f.shells;
        const f32 h = f.length * (0.35f * u + 0.65f * u * u);
        for (int i = 0; i < kEponaPosCount; i++) {
            const f32 off = kEponaFur[i] > 0 ? h : (kEponaFur[i] < 0 ? -2.5f : 0.0f);
            g_shellPos[k][i].x = base[i].x + kEponaNormals[i][0] * off;
            g_shellPos[k][i].y = base[i].y + kEponaNormals[i][1] * off;
            g_shellPos[k][i].z = base[i].z + kEponaNormals[i][2] * off;
        }
    }
    g_shellLevel = f.level;
}

class FurPacket : public J3DPacket {
public:
    void draw() override;
    J3DModel* model = nullptr;
    GXColorS10 light{255, 255, 255, 255};
    FurStyle style;
};
FurPacket l_furPacket;

void FurPacket::draw() {
    if (model == nullptr || g_shellLevel != style.level || !g_noiseReady) {
        return;
    }
    J3DModelData* data = model->getModelData();
    J3DShapePacket* sp = model->getShapePacket(kEponaBodyShape);
    J3DShape* shape = sp != nullptr ? sp->getShape() : nullptr;
    if (shape == nullptr || shape->checkFlag(J3DShpFlag_Visible) /* hide() sets this one */) {
        return;
    }
    J3DMaterial* mat = shape->getMaterial();
    if (mat == nullptr || data->getTexture() == nullptr) {
        return;
    }

    j3dSys.reinitGX();
    GXSetNumIndStages(0);
    data->getTexture()->loadGX(mat->getTexNo(0), GX_TEXMAP0);
    GXLoadTexObj(style.tufts ? &g_tuftObj : &g_noiseObj, GX_TEXMAP1);

    Mtx tm;
    PSMTXScale(tm, style.density, style.density, 1.0f);
    GXLoadTexMtxImm(tm, GX_TEXMTX0, GX_MTX2x4);
    GXSetNumTexGens(2);
    GXSetTexCoordGen(GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY);
    GXSetTexCoordGen(GX_TEXCOORD1, GX_TG_MTX2x4, GX_TG_TEX0, GX_TEXMTX0);
    GXSetNumChans(0);
    GXSetNumTevStages(2);
    GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR_NULL);
    GXSetTevColorIn(GX_TEVSTAGE0, GX_CC_ZERO, GX_CC_TEXC, GX_CC_C0, GX_CC_ZERO);
    GXSetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST);
    GXSetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetTevOrder(GX_TEVSTAGE1, GX_TEXCOORD1, GX_TEXMAP1, GX_COLOR_NULL);
    GXSetTevColorIn(GX_TEVSTAGE1, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV);
    GXSetTevColorOp(GX_TEVSTAGE1, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetTevAlphaIn(GX_TEVSTAGE1, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA);
    GXSetTevAlphaOp(GX_TEVSTAGE1, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_SET);
    GXSetZMode(GX_ENABLE, GX_LEQUAL, GX_ENABLE);
    GXSetCullMode(GX_CULL_BACK);
    GXSetFog(GX_FOG_NONE, 0.0f, 0.0f, 0.0f, 0.0f, GXColor{0, 0, 0, 0});
    J3DDifferedTexMtx::sTexGenBlock = NULL;

    for (int k = 0; k < style.shells; k++) {
        const f32 t = f32(k + 1) / style.shells;
        // Darker deep in the coat, lighter at the tips; tips thin out.
        const f32 shade = style.rootShade + (style.tipShade - style.rootShade) * std::pow(t, 0.8f);
        GXColorS10 c{s16(light.r * shade), s16(light.g * shade), s16(light.b * shade), 255};
        GXSetTevColorS10(GX_TEVREG0, c);
        const f32 cut = style.tufts ? 12.0f + 215.0f * t : 40.0f + 200.0f * t;
        GXSetAlphaCompare(GX_GREATER, u8(cut), GX_AOP_AND, GX_ALWAYS, 0);

        sp->prepareDraw();
        j3dSys.setVtxPos(g_shellPos[k].data(), kEponaPosCount);
        shape->draw();
    }
    J3DShape::resetVcdVatCache();
    j3dSys.reinitGX();
}

GXColorS10 light_of(dKy_tevstr_c* tevstr) {
    GXColorS10 out{180, 170, 160, 255};
    if (tevstr == nullptr) {
        return out;
    }
    const GXColor lc = tevstr->mLightObj.getLightInfo()->mColor;
    const GXColorS10 amb = tevstr->AmbCol;
    out.r = s16(std::min(255, amb.r + lc.r * 3 / 4));
    out.g = s16(std::min(255, amb.g + lc.g * 3 / 4));
    out.b = s16(std::min(255, amb.b + lc.b * 3 / 4));
    return out;
}

}  // namespace

void joint_callback(J3DModel* model, int jointNo) {
    if (jointNo == 0) {
        g_frame++;
    }
    if (!hair_enabled() || !model_matches(model)) {
        return;
    }
    init_bones();
    Bone* b = bone_for(jointNo);
    if (b == nullptr) {
        return;
    }
    MtxP m = model->getAnmMtx(jointNo);
    if (b->frame != g_frame) {
        b->frame = g_frame;
        step_bone(*b, m);
    }
    apply_rotation(m, b->th);
    cMtx_copy(m, J3DSys::mCurrentMtx);
}

void draw(J3DModel* model, dKy_tevstr_c* tevstr) {
    if (!model_matches(model)) {
        return;
    }
    const GXColorS10 l = light_of(tevstr);
    const bool groom = groom_enabled();
    set_cards_hidden(model, groom);
    if (groom) {
        if (!g_hairTexReady) build_hair_tex();
        if (g_groomFrame != g_frame) {
            g_groomFrame = g_frame;
            simulate_groom(model);
        }
        l_groomPacket.light = l;
        dComIfGd_setList();
        j3dSys.getDrawBuffer(J3DSysDrawBuf_Opa)->entryImm(&l_groomPacket, 0);
        g_strandsReady = false;
    } else if (hair_enabled()) {
        if (g_strandFrame != g_frame) {
            g_strandFrame = g_frame;
            simulate_strands(model);
        }
        l_strandPacket.light = GXColor{u8(std::min<int>(255, l.r)), u8(std::min<int>(255, l.g)),
                                       u8(std::min<int>(255, l.b)), 255};
        dComIfGd_setList();
        j3dSys.getDrawBuffer(J3DSysDrawBuf_Xlu)->entryImm(&l_strandPacket, 0);
    } else {
        g_strandsReady = false;
    }
    if (fur_enabled()) {
        if (!g_noiseReady) build_noise();
        const FurStyle style = fur_style();
        if (g_shellLevel != style.level) build_shells(model->getModelData(), style);
        l_furPacket.model = model;
        l_furPacket.light = l;
        l_furPacket.style = style;
        dComIfGd_setList();
        j3dSys.getDrawBuffer(J3DSysDrawBuf_Opa)->entryImm(&l_furPacket, 0);
    }
}

}  // namespace dusk::epona
