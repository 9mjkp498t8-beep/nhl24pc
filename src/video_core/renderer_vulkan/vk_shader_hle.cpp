// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {

static constexpr u64 COPY_SHADER_HASH = 0xfefebf9f;

static bool ExecuteCopyShaderHLE(const Shader::Info& info, const AmdGpu::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();

    // Copy shader defines three formatted buffers as inputs: control, source, and destination.
    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    static constexpr vk::DeviceSize MaxDistanceForMerge = 64_MB;
    u32 batch_start = 0;
    u32 batch_end = 0;

    while (batch_end < copies.size()) {
        // Place first copy into the current batch
        const auto& copy = copies[batch_start];
        auto src_offset_min = copy.srcOffset;
        auto src_offset_max = copy.srcOffset + copy.size;
        auto dst_offset_min = copy.dstOffset;
        auto dst_offset_max = copy.dstOffset + copy.size;

        for (++batch_end; batch_end < copies.size(); batch_end++) {
            // Compute new src and dst bounds if we were to batch this copy
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            auto new_src_offset_min = std::min(src_offset_min, src_offset);
            auto new_src_offset_max = std::max(src_offset_max, src_offset + size);
            if (new_src_offset_max - new_src_offset_min > MaxDistanceForMerge) {
                break;
            }

            auto new_dst_offset_min = std::min(dst_offset_min, dst_offset);
            auto new_dst_offset_max = std::max(dst_offset_max, dst_offset + size);
            if (new_dst_offset_max - new_dst_offset_min > MaxDistanceForMerge) {
                break;
            }

            // We can batch this copy
            src_offset_min = new_src_offset_min;
            src_offset_max = new_src_offset_max;
            dst_offset_min = new_dst_offset_min;
            dst_offset_max = new_dst_offset_max;
        }

        // Obtain buffers for the total source and destination ranges.
        const auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_offset_min, src_offset_max - src_offset_min, false);
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_offset_min, dst_offset_max - dst_offset_min, true);

        // Apply found buffer base.
        const auto vk_copies = std::span{copies}.subspan(batch_start, batch_end - batch_start);
        for (auto& copy : vk_copies) {
            copy.srcOffset = copy.srcOffset - src_offset_min + src_buf_offset;
            copy.dstOffset = copy.dstOffset - dst_offset_min + dst_buf_offset;
        }

        // Execute buffer copies.
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}",
                  src_offset_max - src_offset_min, dst_offset_max - dst_offset_min);
        runtime.CopyBuffer(src_buf, dst_buf, vk_copies);
        batch_start = batch_end;
    }

    return true;
}

// nhlport: NHL 24 in PS4 Pro mode renders a 1600x1800-class frame and rebuilds the 3200x1800 image
// with a checkerboard resolve compute shader. Our translation of that shader does not produce a
// usable image, so replace it with a filtered 2x horizontal upscale of the lit image into its
// colour output. NHLPORT_CB_UPSCALE=0 runs the game's shader instead.
static constexpr u64 NHL24_CHECKERBOARD_RESOLVE_HASH = 0x54d0feeb;

static bool ExecuteCheckerboardUpscaleHLE(const Shader::Info& info, Rasterizer& rasterizer) {
    static const bool enabled = [] {
        const char* e = std::getenv("NHLPORT_CB_UPSCALE");
        return !(e && e[0] == '0');
    }();
    if (!enabled) {
        return false;
    }

    // The lit image is the only read-only R11G11B10 image, the colour output the only written one.
    const Shader::ImageResource* src_res{};
    const Shader::ImageResource* dst_res{};
    AmdGpu::Image src_sharp{};
    AmdGpu::Image dst_sharp{};
    for (const auto& res : info.images) {
        const auto sharp = res.GetSharp(info);
        if (sharp.Address() == 0 || sharp.GetDataFmt() != AmdGpu::DataFormat::Format10_11_11) {
            continue;
        }
        if (res.is_written && !dst_res) {
            dst_res = &res;
            dst_sharp = sharp;
        } else if (!res.is_written && !src_res) {
            src_res = &res;
            src_sharp = sharp;
        }
    }
    if (!src_res || !dst_res) {
        return false;
    }

    auto& texture_cache = rasterizer.GetTextureCache();
    VideoCore::TextureCache::ImageDesc src_desc{src_sharp, *src_res};
    VideoCore::TextureCache::ImageDesc dst_desc{dst_sharp, *dst_res};
    const auto src_id = texture_cache.FindImage(src_desc);
    const auto dst_id = texture_cache.FindImage(dst_desc);
    texture_cache.FindTexture(src_id, src_desc);
    texture_cache.FindTexture(dst_id, dst_desc);
    auto& src = texture_cache.GetImage(src_id);
    auto& dst = texture_cache.GetImage(dst_id);

    const u32 src_w = src.info.size.width;
    const u32 src_h = src.info.size.height;
    const u32 dst_w = dst.info.size.width;
    const u32 dst_h = dst.info.size.height;
    if (src.info.num_samples != 1 || dst.info.num_samples != 1 || dst_w != src_w * 2 ||
        dst_h != src_h) {
        return false;
    }

    auto& runtime = rasterizer.GetRuntime();
    auto& scheduler = rasterizer.GetScheduler();
    scheduler.EndRendering();
    bool needs_flush =
        runtime.Transit(&src, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead);
    needs_flush |=
        runtime.Transit(&dst, vk::ImageLayout::eTransferDstOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        runtime.FlushBarriers();
    }

    const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    const vk::ImageBlit region{
        .srcSubresource = layers,
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{s32(src_w), s32(src_h), 1}},
        .dstSubresource = layers,
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0}, vk::Offset3D{s32(dst_w), s32(dst_h), 1}},
    };
    scheduler.CommandBuffer().blitImage(src.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        dst.GetImage(), vk::ImageLayout::eTransferDstOptimal,
                                        region, vk::Filter::eLinear);
    dst.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool ExecuteShaderHLE(const Shader::Info& info, const AmdGpu::Regs& regs,
                      const AmdGpu::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    switch (info.pgm_hash) {
    case COPY_SHADER_HASH:
        return ExecuteCopyShaderHLE(info, cs_program, rasterizer);
    case NHL24_CHECKERBOARD_RESOLVE_HASH:
        return ExecuteCheckerboardUpscaleHLE(info, rasterizer);
    default:
        return false;
    }
}

} // namespace Vulkan
