// Screen-space ambient occlusion (see ssao.h).

#include "dusk/ssao.h"

#include "dusk/settings.h"

#include <aurora/gfx.hpp>
#include <dolphin/gx/GXAurora.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace dusk::ssao {
namespace {

struct Params {
    float p00, p11, p02, p12;
    float p22, p23, width, height;
    float radius, strength, fadeDist, pad;
};
static_assert(sizeof(Params) == 48);

struct Payload {
    WGPUTextureView depth;
    uint32_t uniformOffset;
    uint32_t uniformSize;
    uint32_t width, height;
};

constexpr const char* kShader = R"(
struct Params {
    p00: f32, p11: f32, p02: f32, p12: f32,
    p22: f32, p23: f32, width: f32, height: f32,
    radius: f32, strength: f32, fade_dist: f32, pad: f32,
};
@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(p[vi], 0.0, 1.0);
}

fn depth_at(px: vec2i) -> f32 {
    let sz = vec2i(textureDimensions(depth_tex));
    return textureLoad(depth_tex, clamp(px, vec2i(0), sz - vec2i(1)), 0).r;
}

// View-space position from the (reversed-Z) depth: d = (p22*z + p23) / z.
fn view_pos(px: vec2i, d: f32) -> vec3f {
    let z = u.p23 / (d - u.p22);
    let uv = (vec2f(px) + 0.5) / vec2f(u.width, u.height);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3f((-z) * (ndc.x + u.p02) / u.p00, (-z) * (ndc.y + u.p12) / u.p11, z);
}

@fragment fn fs_main(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let px = vec2i(frag.xy);
    let d = depth_at(px);
    if (d <= 1e-6) { return vec4f(1.0); }  // sky / far plane
    let p = view_pos(px, d);

    // Normal from the depth of the neighbours (the flatter side, so edges stay clean).
    let pr = view_pos(px + vec2i(1, 0), depth_at(px + vec2i(1, 0)));
    let pl = view_pos(px - vec2i(1, 0), depth_at(px - vec2i(1, 0)));
    let pu = view_pos(px + vec2i(0, 1), depth_at(px + vec2i(0, 1)));
    let pd = view_pos(px - vec2i(0, 1), depth_at(px - vec2i(0, 1)));
    let dx = select(p - pl, pr - p, abs(pr.z - p.z) < abs(p.z - pl.z));
    let dy = select(p - pd, pu - p, abs(pu.z - p.z) < abs(p.z - pd.z));
    var n = normalize(cross(dx, dy));
    if (dot(n, -p) < 0.0) { n = -n; }

    let dist = -p.z;
    let r_px = clamp(u.radius * u.p00 * 0.5 * u.width / dist, 2.0, 0.08 * u.width);
    let noise = fract(52.9829189 * fract(dot(frag.xy, vec2f(0.06711056, 0.00583715))));
    let r2 = u.radius * u.radius;
    var occ = 0.0;
    let N = 12;
    for (var i = 0; i < N; i++) {
        let fi = f32(i);
        let ang = (fi + noise) * 2.39996323;
        let rr = sqrt((fi + 0.5) / f32(N)) * r_px;
        let off = vec2i(vec2f(cos(ang), sin(ang)) * rr);
        let q = px + off;
        let dq = depth_at(q);
        if (dq <= 1e-6) { continue; }
        let s = view_pos(q, dq);
        let v = s - p;
        let l2 = dot(v, v);
        if (l2 < 1e-4) { continue; }
        let cosv = dot(v, n) * inverseSqrt(l2);
        let falloff = clamp(1.0 - l2 / r2, 0.0, 1.0);
        occ += max(0.0, cosv - 0.15) * falloff;
    }
    var ao = 1.0 - u.strength * occ / f32(N) * 1.6;
    let fade = 1.0 - smoothstep(u.fade_dist * 0.5, u.fade_dist, dist);
    ao = mix(1.0, clamp(ao, 0.3, 1.0), fade);
    return vec4f(ao, ao, ao, 1.0);
}
)";

struct GpuState {
    std::mutex mutex;
    uint64_t layoutKey = ~0ull;
    wgpu::RenderPipeline pipeline;
    wgpu::BindGroupLayout bgl;
};
GpuState g_gpu;
aurora::gfx::DrawTypeId g_drawType = aurora::gfx::InvalidDrawType;

void build_pipeline(const aurora::gfx::DrawContext& ctx) {
    const auto& layout = ctx.layout;
    if (g_gpu.pipeline && g_gpu.layoutKey == layout.key) return;
    const wgpu::ShaderSourceWGSL src{wgpu::ShaderSourceWGSL::Init{.code = kShader}};
    const wgpu::ShaderModuleDescriptor md{.nextInChain = &src, .label = "Felix SSAO"};
    const auto module = ctx.device.CreateShaderModule(&md);

    if (!g_gpu.bgl) {
        const wgpu::BindGroupLayoutEntry entries[2] = {
            {.binding = 0,
             .visibility = wgpu::ShaderStage::Fragment,
             .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(Params)}},
            {.binding = 1,
             .visibility = wgpu::ShaderStage::Fragment,
             .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                         .viewDimension = wgpu::TextureViewDimension::e2D}},
        };
        const wgpu::BindGroupLayoutDescriptor bd{.label = "Felix SSAO", .entryCount = 2, .entries = entries};
        g_gpu.bgl = ctx.device.CreateBindGroupLayout(&bd);
    }
    const wgpu::PipelineLayoutDescriptor pld{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_gpu.bgl};
    const auto pl = ctx.device.CreatePipelineLayout(&pld);

    // Multiply the scene colour by the occlusion; leave alpha and other targets alone.
    const wgpu::BlendState blend{
        .color = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::Dst,
                  .dstFactor = wgpu::BlendFactor::Zero},
        .alpha = {.operation = wgpu::BlendOperation::Add,
                  .srcFactor = wgpu::BlendFactor::Zero,
                  .dstFactor = wgpu::BlendFactor::One},
    };
    std::array<wgpu::ColorTargetState, aurora::gfx::MaxColorAttachments> targets{};
    for (uint32_t i = 0; i < layout.colorAttachmentCount; i++) {
        targets[i] = {.format = layout.colorAttachments[i].format, .writeMask = wgpu::ColorWriteMask::None};
    }
    targets[aurora::gfx::SceneColorAttachmentIndex].blend = &blend;
    targets[aurora::gfx::SceneColorAttachmentIndex].writeMask = wgpu::ColorWriteMask::Red |
                                                                 wgpu::ColorWriteMask::Green |
                                                                 wgpu::ColorWriteMask::Blue;
    const wgpu::FragmentState fs{
        .module = module, .entryPoint = "fs_main", .targetCount = layout.colorAttachmentCount,
        .targets = targets.data()};
    const wgpu::DepthStencilState ds{
        .format = layout.depthStencilFormat,
        .depthWriteEnabled = false,
        .depthCompare = wgpu::CompareFunction::Always,
    };
    const wgpu::RenderPipelineDescriptor pd{
        .label = "Felix SSAO",
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
    Payload pl;
    std::memcpy(&pl, payload, sizeof(pl));
    if (pl.depth == nullptr) return;
    std::lock_guard lock{g_gpu.mutex};
    build_pipeline(ctx);
    if (!g_gpu.pipeline) return;
    const wgpu::TextureView depth{pl.depth};
    const wgpu::BindGroupEntry entries[2] = {
        {.binding = 0, .buffer = ctx.uniformBuffer, .offset = pl.uniformOffset, .size = sizeof(Params)},
        {.binding = 1, .textureView = depth},
    };
    const wgpu::BindGroupDescriptor bd{.layout = g_gpu.bgl, .entryCount = 2, .entries = entries};
    const auto bg = ctx.device.CreateBindGroup(&bd);
    pass.SetPipeline(g_gpu.pipeline);
    pass.SetBindGroup(0, bg);
    pass.SetViewport(0.0f, 0.0f, float(pl.width), float(pl.height), 0.0f, 1.0f);
    pass.SetScissorRect(0, 0, pl.width, pl.height);
    pass.Draw(3);
}

}  // namespace

void apply(const Mtx44 proj) {
    const auto& game = dusk::getSettings().game;
    if (!game.ambientOcclusion.getValue()) return;
    if (aurora::gfx::is_offscreen()) return;
    if (g_drawType == aurora::gfx::InvalidDrawType) {
        g_drawType = aurora::gfx::register_draw_type({.label = "Felix SSAO", .draw = draw_cb});
        if (g_drawType == aurora::gfx::InvalidDrawType) return;
    }
    AuroraGXSync();
    aurora::gfx::ResolvedTargets targets;
    if (!aurora::gfx::resolve_pass({.color = false, .depth = true, .normal = false}, targets) ||
        !targets.depth || targets.width == 0 || targets.height == 0)
    {
        return;
    }
    const Params prm{
        .p00 = proj[0][0], .p11 = proj[1][1], .p02 = proj[0][2], .p12 = proj[1][2],
        .p22 = proj[2][2], .p23 = proj[2][3],
        .width = float(targets.width), .height = float(targets.height),
        .radius = 65.0f,  // game units (~65 cm)
        .strength = std::clamp(game.ambientOcclusionStrength.getValue(), 10, 300) / 100.0f,
        .fadeDist = 9000.0f,
        .pad = 0.0f,
    };
    const auto range = aurora::gfx::push_uniform(reinterpret_cast<const uint8_t*>(&prm), sizeof(prm));
    if (range.size == 0) return;
    const Payload pl{
        .depth = targets.depth.Get(),
        .uniformOffset = range.offset,
        .uniformSize = range.size,
        .width = targets.width,
        .height = targets.height,
    };
    aurora::gfx::push_custom_draw(g_drawType, &pl, sizeof(pl));
}

}  // namespace dusk::ssao
