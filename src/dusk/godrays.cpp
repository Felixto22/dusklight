// Light shafts (see godrays.h).

#include "dusk/godrays.h"

#include "JSystem/J3DGraphBase/J3DSys.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "f_op/f_op_camera_mng.h"
#include "dusk/settings.h"

#include <aurora/gfx.hpp>
#include <dolphin/gx/GXAurora.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>

namespace dusk::godrays {
namespace {

struct Params {
    f32 sun[4];    // sun position in uv (xy), intensity, aspect
    f32 color[4];  // rgb, decay
    f32 size[4];   // width, height, -, -
};
static_assert(sizeof(Params) == 48);

struct Payload {
    WGPUTextureView depth;
    u32 uniOffset;
    u32 width, height;
};

constexpr const char* kShader = R"(
struct P { sun: vec4f, color: vec4f, size: vec4f };
@group(0) @binding(0) var<uniform> u: P;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(p[vi], 0.0, 1.0);
}

fn sky_at(uv: vec2f) -> f32 {
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 0.0; }
    let sz = vec2i(textureDimensions(depth_tex));
    let c = clamp(vec2i(uv * vec2f(sz)), vec2i(0), sz - vec2i(1));
    return select(0.0, 1.0, textureLoad(depth_tex, c, 0).r <= 1e-6);
}

@fragment fn fs_main(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let uv = frag.xy / u.size.xy;
    let to_sun = u.sun.xy - uv;
    let n = 28;
    let jit = fract(52.9829189 * fract(dot(frag.xy, vec2f(0.06711056, 0.00583715))));
    var acc = 0.0;
    var w = 1.0;
    var wsum = 0.0;
    for (var i = 0; i < n; i++) {
        let f = (f32(i) + jit) / f32(n);
        acc += sky_at(uv + to_sun * f) * w;
        wsum += w;
        w *= u.color.w;
    }
    let d = length(to_sun * vec2f(u.sun.w, 1.0));
    let fall = 1.0 / (1.0 + d * d * 6.0);
    let rays = acc / wsum * fall * u.sun.z;
    return vec4f(u.color.rgb * rays, 0.0);
}
)";

struct Gpu {
    std::mutex mutex;
    u64 layoutKey = ~0ull;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroupLayout bgl;
};
Gpu g_gpu;
aurora::gfx::DrawTypeId g_drawType = aurora::gfx::InvalidDrawType;

void build_pipeline(const aurora::gfx::DrawContext& ctx) {
    const auto& layout = ctx.layout;
    if (g_gpu.pipeline && g_gpu.layoutKey == layout.key) return;
    const wgpu::ShaderSourceWGSL src{wgpu::ShaderSourceWGSL::Init{.code = kShader}};
    const wgpu::ShaderModuleDescriptor md{.nextInChain = &src, .label = "Felix light shafts"};
    const auto module = ctx.device.CreateShaderModule(&md);
    if (!g_gpu.bgl) {
        const wgpu::BindGroupLayoutEntry e[2] = {
            {.binding = 0,
             .visibility = wgpu::ShaderStage::Fragment,
             .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(Params)}},
            {.binding = 1,
             .visibility = wgpu::ShaderStage::Fragment,
             .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                         .viewDimension = wgpu::TextureViewDimension::e2D}},
        };
        const wgpu::BindGroupLayoutDescriptor bd{.label = "Felix light shafts", .entryCount = 2, .entries = e};
        g_gpu.bgl = ctx.device.CreateBindGroupLayout(&bd);
    }
    const wgpu::PipelineLayoutDescriptor pld{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_gpu.bgl};
    const auto pl = ctx.device.CreatePipelineLayout(&pld);
    // Additive: light is added to the scene.
    const wgpu::BlendState blend{
        .color = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::One,
                  .dstFactor = wgpu::BlendFactor::One},
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
        .depthCompare = wgpu::CompareFunction::Always,
    };
    const wgpu::RenderPipelineDescriptor pd{
        .label = "Felix light shafts",
        .layout = pl,
        .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main"},
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
    if (payloadSize != sizeof(Payload)) return;
    Payload p;
    std::memcpy(&p, payload, sizeof(p));
    if (p.depth == nullptr) return;
    std::lock_guard lock{g_gpu.mutex};
    build_pipeline(ctx);
    if (!g_gpu.pipeline) return;
    const wgpu::TextureView depth{p.depth};
    const wgpu::BindGroupEntry e[2] = {
        {.binding = 0, .buffer = ctx.uniformBuffer, .offset = p.uniOffset, .size = sizeof(Params)},
        {.binding = 1, .textureView = depth},
    };
    const wgpu::BindGroupDescriptor bd{.layout = g_gpu.bgl, .entryCount = 2, .entries = e};
    const auto bg = ctx.device.CreateBindGroup(&bd);
    pass.SetPipeline(g_gpu.pipeline);
    pass.SetBindGroup(0, bg);
    pass.SetViewport(0.0f, 0.0f, f32(p.width), f32(p.height), 0.0f, 1.0f);
    pass.SetScissorRect(0, 0, p.width, p.height);
    pass.Draw(3);
}

}  // namespace

void apply() {
    const auto& game = dusk::getSettings().game;
    if (!game.godRays.getValue() || aurora::gfx::is_offscreen()) return;
    camera_process_class* cam = dComIfGp_getCamera(0);
    if (cam == NULL) return;
    dScnKy_env_light_c* env = dKy_getEnvlight();

    // Sun on screen: view-projection as aurora draws it.
    const Mtx44& P = cam->view.projMtx;
    MtxP V = j3dSys.getViewMtx();
    const f32 s[3] = {env->sun_pos.x, env->sun_pos.y, env->sun_pos.z};
    f32 vs[3];
    for (int r = 0; r < 3; r++) vs[r] = V[r][0] * s[0] + V[r][1] * s[1] + V[r][2] * s[2] + V[r][3];
    if (vs[2] >= -1.0f) return;  // behind the camera
    const f32 cx = P[0][0] * vs[0] + P[0][2] * vs[2];
    const f32 cy = P[1][1] * vs[1] + P[1][2] * vs[2];
    const f32 w = -vs[2];
    const f32 sx = (cx / w) * 0.5f + 0.5f, sy = 0.5f - (cy / w) * 0.5f;
    // Fade out as the sun leaves the screen, and with how directly we look at it.
    const f32 edge = std::max(std::max(-sx, sx - 1.0f), std::max(-sy, sy - 1.0f));
    const f32 onScreen = std::clamp(1.0f - edge / 0.6f, 0.0f, 1.0f);
    const f32 len = std::sqrt(vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2]);
    const f32 facing = std::clamp((-vs[2] / std::max(len, 1.0f) - 0.3f) / 0.5f, 0.0f, 1.0f);

    const f32 t = env->getDaytime();
    const bool day = t >= 90.0f && t < 285.0f;
    const bool lowSun = t < 120.0f || t > 255.0f;
    f32 col[3] = {1.0f, 0.93f, 0.78f};
    if (!day) {
        col[0] = 0.45f; col[1] = 0.55f; col[2] = 0.85f;
    } else if (lowSun) {
        col[0] = 1.0f; col[1] = 0.68f; col[2] = 0.38f;
    }
    f32 intensity = onScreen * facing * (day ? 0.55f : 0.18f) *
                    std::clamp(game.godRaysStrength.getValue(), 0, 300) / 100.0f;
    if (dKy_darkworld_check()) intensity *= 0.4f;
    if (intensity <= 0.002f) return;

    if (g_drawType == aurora::gfx::InvalidDrawType) {
        g_drawType = aurora::gfx::register_draw_type({.label = "Felix light shafts", .draw = draw_cb});
        if (g_drawType == aurora::gfx::InvalidDrawType) return;
    }
    AuroraGXSync();
    aurora::gfx::ResolvedTargets targets;
    if (!aurora::gfx::resolve_pass({.color = false, .depth = true, .normal = false}, targets) ||
        !targets.depth || targets.width == 0)
    {
        return;
    }
    const Params prm{
        .sun = {sx, sy, intensity, f32(targets.width) / f32(targets.height)},
        .color = {col[0], col[1], col[2], 0.965f},
        .size = {f32(targets.width), f32(targets.height), 0.0f, 0.0f},
    };
    const auto ur = aurora::gfx::push_uniform(reinterpret_cast<const uint8_t*>(&prm), sizeof(prm));
    if (ur.size == 0) return;
    const Payload pl{.depth = targets.depth.Get(), .uniOffset = ur.offset, .width = targets.width,
                     .height = targets.height};
    aurora::gfx::push_custom_draw(g_drawType, &pl, sizeof(pl));
}

}  // namespace dusk::godrays
