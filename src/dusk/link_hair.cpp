// Real hair for Link (see link_hair.h).

#include "dusk/link_hair.h"

#include "JSystem/J3DGraphAnimator/J3DModel.h"
#include "JSystem/J3DGraphAnimator/J3DModelData.h"
#include "JSystem/J3DGraphBase/J3DDrawBuffer.h"
#include "JSystem/J3DGraphBase/J3DMaterial.h"
#include "JSystem/J3DGraphBase/J3DPacket.h"
#include "JSystem/J3DGraphBase/J3DShape.h"
#include "JSystem/J3DGraphBase/J3DShapeMtx.h"
#include "JSystem/J3DGraphBase/J3DSys.h"
#include "JSystem/J3DGraphBase/J3DTexture.h"
#include "SSystem/SComponent/c_counter.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo_wether.h"
#include "f_op/f_op_camera_mng.h"
#include "dusk/settings.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dusk::linkhair {
namespace {

#include "dusk/link_hair.inc"

bool enabled() { return dusk::getSettings().game.linkRealHair.getValue(); }

struct V3 {
    f32 x, y, z;
    V3 operator+(const V3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    V3 operator*(f32 s) const { return {x * s, y * s, z * s}; }
    V3& operator+=(const V3& o) { x += o.x; y += o.y; z += o.z; return *this; }
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

constexpr int kMaxPts = 8;

struct State {
    V3 p[kMaxPts], q[kMaxPts];
    f32 rest[kMaxPts];
    bool placed = false;
};

const LinkHairModel* g_model = nullptr;
J3DModelData* g_modelData = nullptr;
std::vector<State> g_state;
u32 g_simFrame = 0xFFFFFFFFu;
V3 g_headCenter{};

const LinkHairModel* identify(J3DModel* model) {
    J3DModelData* data = model->getModelData();
    for (const LinkHairModel& m : kLinkHairModels) {
        if (data->getVertexData().getVtxNum() == m.vtxNum && data->getJointNum() == m.jointNum &&
            data->getShapeNum() > m.hairShape && data->getMaterialNum() > m.hairMat) {
            return &m;
        }
    }
    return nullptr;
}

V3 target(J3DModel* model, const LinkHairPoint& p) {
    const f32 a[3] = {p.a[0] * kLinkHairScale, p.a[1] * kLinkHairScale, p.a[2] * kLinkHairScale};
    const V3 pa = xform(model->getAnmMtx(p.j0), a);
    if (p.w0 >= 255) return pa;
    const f32 b[3] = {p.b[0] * kLinkHairScale, p.b[1] * kLinkHairScale, p.b[2] * kLinkHairScale};
    const V3 pb = xform(model->getAnmMtx(p.j1), b);
    const f32 w = p.w0 / 255.0f;
    return pa * w + pb * (1.0f - w);
}

void simulate(J3DModel* model) {
    const LinkHairModel& m = *g_model;
    if (g_state.size() != m.strandCount) {
        g_state.assign(m.strandCount, State{});
    }
    // Head centre: in head_root's frame +x is forward and -y is up.
    const f32 c[3] = {3.0f, -12.0f, 0.0f};
    g_headCenter = xform(model->getAnmMtx(0), c);
    cXyz wpos(g_headCenter.x, g_headCenter.y, g_headCenter.z), wdir(0.0f, 0.0f, 0.0f);
    f32 wpow = 0.0f;
    dKyw_get_AllWind_vec(&wpos, &wdir, &wpow);
    const V3 wind = V3{wdir.x, wdir.y, wdir.z} * (wpow * 0.15f);
    const V3 gravity{0.0f, -0.05f, 0.0f};
    for (int si = 0; si < m.strandCount; si++) {
        const LinkHairStrand& st = kLinkHairStrands[m.firstStrand + si];
        State& g = g_state[si];
        const int n = std::min<int>(st.count, kMaxPts);
        V3 t[kMaxPts];
        for (int k = 0; k < n; k++) t[k] = target(model, kLinkHairPoints[st.first + k]);
        if (!g.placed || len(t[0] - g.p[0]) > 100.0f) {
            for (int k = 0; k < n; k++) {
                g.p[k] = g.q[k] = t[k];
                g.rest[k] = k == 0 ? 0.0f : len(t[k] - t[k - 1]);
            }
            g.placed = true;
            continue;
        }
        g.q[0] = g.p[0];
        g.p[0] = t[0];
        for (int k = 1; k < n; k++) {
            const f32 u = f32(k) / (n - 1);
            const V3 v = (g.p[k] - g.q[k]) * 0.88f;
            g.q[k] = g.p[k];
            g.p[k] += v + gravity + wind * u;
            // Short hair keeps its style: firm pull back to the combed shape.
            g.p[k] += (t[k] - g.p[k]) * (0.60f - 0.38f * u);
        }
        for (int k = 1; k < n; k++) {
            const V3 e = g.p[k] - g.p[k - 1];
            const f32 l = len(e);
            if (l > 1e-4f) g.p[k] = g.p[k - 1] + e * (g.rest[k] / l);
        }
    }
}

// Fine hairs across each lock (I8, GX tile layout): 4 hairs of varying shade and length.
alignas(32) u8 g_hairTex[32 * 64];
GXTexObj g_hairObj;
bool g_hairReady = false;

void build_hair_tex() {
    u32 seed = 0x5EEDF00Du;
    auto r = [&] {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) / f32(1 << 24);
    };
    static u8 linear[64][32];
    for (int hair = 0; hair < 4; hair++) {
        const f32 shade = 0.75f + 0.25f * r();
        const f32 end = 0.75f + 0.25f * r();
        const f32 wob = (r() - 0.5f) * 0.9f;
        for (int y = 0; y < 64; y++) {
            const f32 t = (y + 0.5f) / 64.0f;
            const f32 thin = t < end ? 1.0f - 0.6f * (t / end) * (t / end) : 0.0f;
            for (int x = 0; x < 8; x++) {
                const f32 c = std::fabs((x + 0.5f + wob * 2.0f * t) - 4.0f) / 3.2f;
                const f32 v = std::max(0.0f, 1.0f - c / std::max(0.2f, thin)) * shade;
                linear[y][hair * 8 + x] = u8(std::clamp(v * 255.0f, 0.0f, 255.0f));
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
    g_hairReady = true;
}

// Light and colour like the original hair material (its lights, colours and its TEV
// formula (D + K*(1-RAS) + TEX*RAS) x4), with our texture coordinates.
void load_material(J3DMaterial* mat) {
    J3DColorBlock* cb = mat->getColorBlock();
    J3DColorChan* ch = cb != nullptr ? cb->getColorChan(0) : nullptr;
    GXSetNumChans(1);
    if (ch != nullptr) {
        for (u32 i = 0; i < 8; i++) {
            J3DLightObj* l = cb->getLight(i);
            if (l == nullptr) continue;
            const J3DLightInfo& li = *l->getLightInfo();
            GXLightObj o;
            GXInitLightPos(&o, li.mLightPosition.x, li.mLightPosition.y, li.mLightPosition.z);
            GXInitLightDir(&o, li.mLightDirection.x, li.mLightDirection.y, li.mLightDirection.z);
            GXInitLightAttn(&o, li.mCosAtten.x, li.mCosAtten.y, li.mCosAtten.z, li.mDistAtten.x,
                            li.mDistAtten.y, li.mDistAtten.z);
            GXInitLightColor(&o, li.mColor);
            GXLoadLightObjImm(&o, GXLightID(1 << i));
        }
        if (J3DGXColor* amb = cb->getAmbColor(0)) GXSetChanAmbColor(GX_COLOR0A0, *amb);
        if (J3DGXColor* mc = cb->getMatColor(0)) GXSetChanMatColor(GX_COLOR0A0, *mc);
        GXSetChanCtrl(GX_COLOR0, ch->getEnable() ? GX_ENABLE : GX_DISABLE,
                      GXColorSrc(ch->getAmbSrc()), GXColorSrc(ch->getMatSrc()), ch->getLightMask(),
                      GXDiffuseFn(ch->getDiffuseFn()), GXAttnFn(ch->getAttnFn()));
    } else {
        GXSetChanCtrl(GX_COLOR0, GX_DISABLE, GX_SRC_REG, GX_SRC_REG, GX_LIGHT_NULL, GX_DF_NONE,
                      GX_AF_NONE);
        GXSetChanMatColor(GX_COLOR0A0, GXColor{255, 255, 255, 255});
    }
    GXSetChanCtrl(GX_ALPHA0, GX_DISABLE, GX_SRC_REG, GX_SRC_REG, GX_LIGHT_NULL, GX_DF_NONE,
                  GX_AF_NONE);

    J3DTevBlock* tb = mat->getTevBlock();
    GXTevColorArg d = GX_CC_C0;
    u8 ksel = GX_TEV_KCSEL_K0;
    if (tb != nullptr) {
        for (u32 i = 0; i < 3; i++) {
            if (J3DGXColorS10* c = tb->getTevColor(i)) GXSetTevColorS10(GXTevRegID(GX_TEVREG0 + i), *c);
        }
        for (u32 i = 0; i < 4; i++) {
            if (J3DGXColor* k = tb->getTevKColor(i)) GXSetTevKColor(GXTevKColorID(i), *k);
        }
        if (J3DTevStage* s = tb->getTevStage(0)) d = GXTevColorArg(s->mTevColorCD & 0xF);
        ksel = tb->getTevKColorSel(0);
    }
    GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
    GXSetTevKColorSel(GX_TEVSTAGE0, GXTevKColorSel(ksel));
    GXSetTevColorIn(GX_TEVSTAGE0, GX_CC_KONST, GX_CC_TEXC, GX_CC_RASC, d);
    GXSetTevColorOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_4, GX_TRUE, GX_TEVPREV);
    GXSetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO);
    GXSetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
}

class HairPacket : public J3DPacket {
public:
    void draw() override;
    J3DModel* model = nullptr;
};
HairPacket l_packet;

inline f32 hash01(u32 x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (x & 0xFFFFFF) / f32(0x1000000);
}

void HairPacket::draw() {
    if (model == nullptr || g_model == nullptr || g_state.size() != g_model->strandCount) return;
    camera_process_class* cam = dComIfGp_getCamera(0);
    if (cam == NULL) return;
    J3DModelData* data = model->getModelData();
    J3DMaterial* mat = data->getMaterialNodePointer(g_model->hairMat);
    if (mat == nullptr || data->getTexture() == nullptr) return;
    const V3 eye{cam->view.lookat.eye.x, cam->view.lookat.eye.y, cam->view.lookat.eye.z};

    j3dSys.reinitGX();
    GXSetNumIndStages(0);
    GXClearVtxDesc();
    GXSetVtxDesc(GX_VA_POS, GX_DIRECT);
    GXSetVtxDesc(GX_VA_NRM, GX_DIRECT);
    GXSetVtxDesc(GX_VA_TEX0, GX_DIRECT);
    GXSetVtxDesc(GX_VA_TEX1, GX_DIRECT);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_POS, GX_POS_XYZ, GX_F32, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_NRM, GX_NRM_XYZ, GX_F32, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX0, GX_TEX_ST, GX_F32, 0);
    GXSetVtxAttrFmt(GX_VTXFMT0, GX_VA_TEX1, GX_TEX_ST, GX_F32, 0);
    data->getTexture()->loadGX(mat->getTexNo(0), GX_TEXMAP0);
    GXLoadTexObj(&g_hairObj, GX_TEXMAP1);
    GXSetNumTexGens(2);
    GXSetTexCoordGen(GX_TEXCOORD0, GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY);
    GXSetTexCoordGen(GX_TEXCOORD1, GX_TG_MTX2x4, GX_TG_TEX1, GX_IDENTITY);
    load_material(mat);
    GXSetNumTevStages(2);
    // Stage 1: keep the colour, take the alpha from the fine-hair pattern.
    GXSetTevOrder(GX_TEVSTAGE1, GX_TEXCOORD1, GX_TEXMAP1, GX_COLOR_NULL);
    GXSetTevColorIn(GX_TEVSTAGE1, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_CPREV);
    GXSetTevColorOp(GX_TEVSTAGE1, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetTevAlphaIn(GX_TEVSTAGE1, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA);
    GXSetTevAlphaOp(GX_TEVSTAGE1, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXLoadPosMtxImm(j3dSys.getViewMtx(), GX_PNMTX0);
    GXLoadNrmMtxImm(j3dSys.getViewMtx(), GX_PNMTX0);
    GXSetCurrentMtx(GX_PNMTX0);
    GXSetBlendMode(GX_BM_NONE, GX_BL_ONE, GX_BL_ZERO, GX_LO_SET);
    GXSetCullMode(GX_CULL_NONE);
    GXSetZMode(GX_ENABLE, GX_LEQUAL, GX_ENABLE);
    GXSetAlphaCompare(GX_GREATER, 70, GX_AOP_AND, GX_ALWAYS, 0);
    GXSetFog(GX_FOG_NONE, 0.0f, 0.0f, 0.0f, 0.0f, GXColor{0, 0, 0, 0});
    J3DDifferedTexMtx::sTexGenBlock = NULL;

    const LinkHairModel& m = *g_model;
    for (int si = 0; si < m.strandCount; si++) {
        const LinkHairStrand& st = kLinkHairStrands[m.firstStrand + si];
        const State& g = g_state[si];
        const int n = std::min<int>(st.count, kMaxPts);
        if (!g.placed || n < 2) continue;
        const f32 w0 = 0.95f * (0.8f + 0.4f * hash01(u32(si) * 2654435761u));
        const f32 w1 = w0 * 0.35f;
        GXBegin(GX_TRIANGLESTRIP, GX_VTXFMT0, u16(n * 2));
        for (int k = 0; k < n; k++) {
            const f32 t = f32(k) / (n - 1);
            const V3 a = g.p[std::max(0, k - 1)], b = g.p[std::min(n - 1, k + 1)];
            const V3 dir = norm(b - a);
            const V3 view = norm(eye - g.p[k]);
            const V3 side = norm(cross(dir, view)) * (0.5f * (w0 + (w1 - w0) * t));
            const V3 nrm = norm(g.p[k] - g_headCenter);
            const LinkHairPoint& hp = kLinkHairPoints[st.first + k];
            const V3 p0 = g.p[k] - side, p1 = g.p[k] + side;
            GXPosition3f32(p0.x, p0.y, p0.z);
            GXNormal3f32(nrm.x, nrm.y, nrm.z);
            GXTexCoord2f32(hp.uv[0], hp.uv[1]);
            GXTexCoord2f32(0.0f, t);
            GXPosition3f32(p1.x, p1.y, p1.z);
            GXNormal3f32(nrm.x, nrm.y, nrm.z);
            GXTexCoord2f32(hp.uv[0], hp.uv[1]);
            GXTexCoord2f32(1.0f, t);
        }
        GXEnd();
    }
    for (int i = 0; i < GX_MAX_TEVSTAGE; i++) {
        GXSetTevKColorSel(GXTevStageID(i), GX_TEV_KCSEL_1_4);
        GXSetTevKAlphaSel(GXTevStageID(i), GX_TEV_KASEL_1);
    }
    J3DShape::resetVcdVatCache();
    j3dSys.reinitGX();
}

}  // namespace

void draw(J3DModel* headModel) {
    if (headModel == nullptr || !enabled()) return;
    J3DModelData* data = headModel->getModelData();
    if (data != g_modelData) {
        g_modelData = data;
        g_model = identify(headModel);
        g_state.clear();
    }
    if (g_model == nullptr) return;
    if (!g_hairReady) build_hair_tex();
    const u32 frame = u32(g_Counter.mCounter0);
    if (frame != g_simFrame) {
        g_simFrame = frame;
        simulate(headModel);
    }
    l_packet.model = headModel;
    dComIfGd_setList();
    j3dSys.getDrawBuffer(J3DSysDrawBuf_Xlu)->entryImm(&l_packet, 0);
}

}  // namespace dusk::linkhair
