// Fur for Link's wolf form (see wolf_fur.h). The same shell technique as Epona's coat:
// the body is drawn again in layers pushed out along its normals, each cutting a hair /
// tuft pattern higher up, coloured and lit exactly like the original body material.

#include "dusk/wolf_fur.h"

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
#include "dusk/settings.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dusk::wolffur {
namespace {

#include "dusk/wolf_data.inc"

void load_material_lighting(J3DMaterial* mat, bool vertexAlbedo) {
    J3DColorBlock* cb = mat != nullptr ? mat->getColorBlock() : nullptr;
    J3DColorChan* ch = cb != nullptr ? cb->getColorChan(0) : nullptr;
    GXSetNumChans(1);
    if (ch == nullptr) {
        GXSetChanCtrl(GX_COLOR0A0, GX_DISABLE, GX_SRC_REG, vertexAlbedo ? GX_SRC_VTX : GX_SRC_REG,
                      GX_LIGHT_NULL, GX_DF_NONE, GX_AF_NONE);
        GXSetChanMatColor(GX_COLOR0A0, GXColor{255, 255, 255, 255});
        return;
    }
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
    if (J3DGXColor* mc = cb->getMatColor(0)) {
        GXSetChanMatColor(GX_COLOR0A0, vertexAlbedo ? GXColor{255, 255, 255, 255} : GXColor(*mc));
    }
    GXSetChanCtrl(GX_COLOR0, ch->getEnable() ? GX_ENABLE : GX_DISABLE, GXColorSrc(ch->getAmbSrc()),
                  vertexAlbedo ? GX_SRC_VTX : GXColorSrc(ch->getMatSrc()), ch->getLightMask(),
                  GXDiffuseFn(ch->getDiffuseFn()), GXAttnFn(ch->getAttnFn()));
    GXSetChanCtrl(GX_ALPHA0, GX_DISABLE, GX_SRC_REG, GX_SRC_REG, GX_LIGHT_NULL, GX_DF_NONE,
                  GX_AF_NONE);
}

// Epona's materials all combine as (C0 + K0*(1-RAS) + TEX*RAS) x4 (hs.bmd MAT3), with C0
// and K0 set by the game each frame; reproduce that in `stage` with our own texture.
void load_material_tev(J3DMaterial* mat, GXTevStageID stage) {
    J3DTevBlock* tb = mat != nullptr ? mat->getTevBlock() : nullptr;
    GXColorS10 c0{0, 0, 0, 255};
    GXColor k0{0, 0, 0, 255};
    if (tb != nullptr) {
        if (J3DGXColorS10* c = tb->getTevColor(0)) c0 = *c;
        if (J3DGXColor* k = tb->getTevKColor(0)) k0 = *k;
    }
    GXSetTevColorS10(GX_TEVREG0, c0);
    GXSetTevKColor(GX_KCOLOR0, k0);
    GXSetTevKColorSel(stage, GX_TEV_KCSEL_K0);
    GXSetTevColorIn(stage, GX_CC_KONST, GX_CC_TEXC, GX_CC_RASC, GX_CC_C0);
    GXSetTevColorOp(stage, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_4, GX_TRUE, GX_TEVPREV);
}

// Put back what J3D expects to find (reinitGX leaves the konst selections alone).
void restore_gx() {
    for (int i = 0; i < GX_MAX_TEVSTAGE; i++) {
        GXSetTevKColorSel(GXTevStageID(i), GX_TEV_KCSEL_1_4);
        GXSetTevKAlphaSel(GXTevStageID(i), GX_TEV_KASEL_1);
    }
    J3DShape::resetVcdVatCache();
    j3dSys.reinitGX();
}

constexpr int kMaxShells = 26;

struct FurStyle {
    int level = 0;
    int shells = 8;
    f32 length = 1.6f;   // game units (1 unit ~ 1 cm)
    f32 density = 34.0f; // pattern repeats across the body texture
    f32 rootShade = 0.78f;
    f32 tipShade = 1.0f;
    bool tufts = false;
};

FurStyle fur_style() {
    const int v = std::clamp(dusk::getSettings().game.wolfFurLength.getValue(), 1, 10);
    FurStyle f;
    f.level = v;
    const f32 t = (v - 1) / 9.0f;  // 0 short .. 1 very fluffy
    f.shells = 8 + int(std::round(18.0f * t));
    f.length = 1.6f + 7.4f * t * t + 2.0f * t;
    f.density = 34.0f / (1.0f + 1.6f * t);
    f.rootShade = 0.80f - 0.20f * t;  // deep coat a little darker between the tufts
    f.tipShade = 1.0f;                 // tips at the body's own brightness
    f.tufts = v >= 3;
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
        g_shellPos[k].resize(kWolfPosCount);
        // Shells packed closer near the skin, where the coat is densest.
        const f32 u = f32(k + 1) / f.shells;
        // Start a little off the skin so the body can never win the depth test.
        const f32 h0 = std::min(0.6f, f.length * 0.3f);
        const f32 h = h0 + (f.length - h0) * (0.35f * u + 0.65f * u * u);
        for (int i = 0; i < kWolfPosCount; i++) {
            const f32 off = kWolfFur[i] > 0 ? h : (kWolfFur[i] < 0 ? -1.5f : 0.0f);
            g_shellPos[k][i].x = base[i].x + kWolfNormals[i][0] * off;
            g_shellPos[k][i].y = base[i].y + kWolfNormals[i][1] * off;
            g_shellPos[k][i].z = base[i].z + kWolfNormals[i][2] * off;
        }
    }
    g_shellLevel = f.level;
}

class FurPacket : public J3DPacket {
public:
    void draw() override;
    J3DModel* model = nullptr;
    FurStyle style;
};
FurPacket l_furPackets[4];
u32 l_furNext = 0;

void FurPacket::draw() {
    if (model == nullptr || g_shellLevel != style.level || !g_noiseReady) {
        return;
    }
    J3DModelData* data = model->getModelData();
    J3DShapePacket* sp = model->getShapePacket(kWolfBodyShape);
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
    load_material_lighting(mat, false);
    GXSetNumTevStages(2);
    // The body's own material formula and lighting, then the depth shade in C1.
    GXSetTevOrder(GX_TEVSTAGE0, GX_TEXCOORD0, GX_TEXMAP0, GX_COLOR0A0);
    load_material_tev(mat, GX_TEVSTAGE0);
    GXSetTevColorIn(GX_TEVSTAGE1, GX_CC_ZERO, GX_CC_CPREV, GX_CC_C1, GX_CC_ZERO);
    GXSetTevAlphaIn(GX_TEVSTAGE0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST);
    GXSetTevAlphaOp(GX_TEVSTAGE0, GX_TEV_ADD, GX_TB_ZERO, GX_CS_SCALE_1, GX_TRUE, GX_TEVPREV);
    GXSetTevOrder(GX_TEVSTAGE1, GX_TEXCOORD1, GX_TEXMAP1, GX_COLOR_NULL);
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
        // The TEV's multiplier input only takes 0..255 (higher values wrap to black).
        const s16 sv = s16(std::clamp(255.0f * shade, 0.0f, 255.0f));
        GXColorS10 c{sv, sv, sv, 255};
        GXSetTevColorS10(GX_TEVREG1, c);
        const f32 cut = style.tufts ? 12.0f + 215.0f * t : 40.0f + 200.0f * t;
        if (k == 0) {
            // The first layer is a solid undercoat hugging the skin, so nothing of the
            // original surface ever shows between the hairs. Fully opaque, alpha 1.
            GXSetTevKAlphaSel(GX_TEVSTAGE1, GX_TEV_KASEL_1);
            GXSetTevAlphaIn(GX_TEVSTAGE1, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_KONST);
            GXSetAlphaCompare(GX_ALWAYS, 0, GX_AOP_AND, GX_ALWAYS, 0);
        } else {
            GXSetTevAlphaIn(GX_TEVSTAGE1, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_TEXA);
            GXSetAlphaCompare(GX_GREATER, u8(cut), GX_AOP_AND, GX_ALWAYS, 0);
        }

        sp->prepareDraw();
        j3dSys.setVtxPos(g_shellPos[k].data(), kWolfPosCount);
        shape->draw();
    }
    restore_gx();
}

bool model_matches(J3DModel* model) {
    J3DModelData* data = model->getModelData();
    return data->getJointNum() == 40 && data->getVertexData().getVtxNum() == kWolfInfVtx &&
           data->getShapeNum() > kWolfBodyShape;
}

}  // namespace

void draw(J3DModel* model) {
    if (model == nullptr || !dusk::getSettings().game.wolfFur.getValue() || !model_matches(model)) {
        return;
    }
    if (!g_noiseReady) build_noise();
    const FurStyle style = fur_style();
    if (g_shellLevel != style.level) build_shells(model->getModelData(), style);
    FurPacket& pk = l_furPackets[l_furNext++ & 3];
    pk.model = model;
    pk.style = style;
    J3DDrawBuffer* buf = j3dSys.getDrawBuffer(J3DSysDrawBuf_Xlu);
    if (buf != nullptr) buf->entryImm(&pk, 0);
}

}  // namespace dusk::wolffur
