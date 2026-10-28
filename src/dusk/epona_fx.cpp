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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace dusk::epona {
namespace {

#include "dusk/epona_data.inc"

bool hair_enabled() { return dusk::getSettings().game.eponaHairPhysics.getValue(); }
bool fur_enabled() { return dusk::getSettings().game.eponaFur.getValue(); }

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
// Shell fur

constexpr int kShells = 8;
constexpr f32 kFurLength = 1.6f;   // game units (~1.6 cm)
constexpr f32 kFurDensity = 34.0f; // pattern repeats across the body texture

std::vector<Vec> g_shellPos[kShells];
bool g_shellsReady = false;
alignas(32) u8 g_noise[64 * 64];
GXTexObj g_noiseObj;
bool g_noiseReady = false;

void build_noise() {
    // I8 texture in the GameCube tile layout (8x4 texel blocks).
    u32 seed = 0x1234567u;
    auto r = [&] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / f32(1 << 24);
    };
    u8 linear[64 * 64];
    for (int i = 0; i < 64 * 64; i++) {
        const f32 v = r();
        linear[i] = v < 0.6f ? u8(70 + 185 * r()) : 0;  // most texels grow a hair
    }
    int o = 0;
    for (int by = 0; by < 64; by += 4) {
        for (int bx = 0; bx < 64; bx += 8) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 8; x++) {
                    g_noise[o++] = linear[(by + y) * 64 + bx + x];
                }
            }
        }
    }
    GXInitTexObj(&g_noiseObj, g_noise, 64, 64, GX_TF_I8, GX_REPEAT, GX_REPEAT, GX_FALSE);
    GXInitTexObjLOD(&g_noiseObj, GX_NEAR, GX_NEAR, 0.0f, 0.0f, 0.0f, GX_FALSE, GX_FALSE,
                    GX_ANISO_1);
    g_noiseReady = true;
}

void build_shells(J3DModelData* data) {
    const Vec* base = (const Vec*)data->getVertexData().getVtxPosArray();
    for (int k = 0; k < kShells; k++) {
        g_shellPos[k].resize(kEponaPosCount);
        const f32 h = kFurLength * f32(k + 1) / kShells;
        for (int i = 0; i < kEponaPosCount; i++) {
            const f32 off = kEponaFur[i] > 0 ? h : (kEponaFur[i] < 0 ? -2.5f : 0.0f);
            g_shellPos[k][i].x = base[i].x + kEponaNormals[i][0] * off;
            g_shellPos[k][i].y = base[i].y + kEponaNormals[i][1] * off;
            g_shellPos[k][i].z = base[i].z + kEponaNormals[i][2] * off;
        }
    }
    g_shellsReady = true;
}

class FurPacket : public J3DPacket {
public:
    void draw() override;
    J3DModel* model = nullptr;
    GXColorS10 light{255, 255, 255, 255};
};
FurPacket l_furPacket;

void FurPacket::draw() {
    if (model == nullptr || !g_shellsReady || !g_noiseReady) {
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
    GXLoadTexObj(&g_noiseObj, GX_TEXMAP1);

    Mtx tm;
    PSMTXScale(tm, kFurDensity, kFurDensity, 1.0f);
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

    for (int k = 0; k < kShells; k++) {
        const f32 t = f32(k + 1) / kShells;
        // Darker deep in the coat, lighter at the tips; tips thin out.
        const f32 shade = 0.78f + 0.30f * t;
        GXColorS10 c{s16(light.r * shade), s16(light.g * shade), s16(light.b * shade), 255};
        GXSetTevColorS10(GX_TEVREG0, c);
        GXSetAlphaCompare(GX_GREATER, u8(40 + 200 * t), GX_AOP_AND, GX_ALWAYS, 0);

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
    if (hair_enabled()) {
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
        if (!g_shellsReady) build_shells(model->getModelData());
        l_furPacket.model = model;
        l_furPacket.light = l;
        dComIfGd_setList();
        j3dSys.getDrawBuffer(J3DSysDrawBuf_Opa)->entryImm(&l_furPacket, 0);
    }
}

}  // namespace dusk::epona
