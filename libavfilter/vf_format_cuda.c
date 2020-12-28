/*
 * Copyright (c) 2020 Yaroslav Pogrebnyak <yyyaroslav@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file format_cuda filter
 */

#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"

#include "avfilter.h"
#include "framesync.h"
#include "internal.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, ctx->hwctx->internal->cuda_dl, x)
#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

#define BLOCK_X 32
#define BLOCK_Y 16

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_P010LE,
    AV_PIX_FMT_NONE,
};

/**
 * FormatCUDAContext
 */
typedef struct FormatCUDAContext {
    const AVClass      *class;

    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    char *out_format_str;

    AVBufferRef *device_ref;
    AVBufferRef *frames_ctx;
    AVCUDADeviceContext *hwctx;

    CUmodule cu_module;
    CUfunction cu_func;
    CUstream cu_stream;

} FormatCUDAContext;

/**
 * Helper to find out if provided format is supported by filter
 */
static int format_is_supported(const enum AVPixelFormat formats[], enum AVPixelFormat fmt)
{
    for (int i = 0; formats[i] != AV_PIX_FMT_NONE; i++)
        if (formats[i] == fmt)
            return 1;
    return 0;
}

/**
 * Init filter
 */
static av_cold int format_cuda_init(AVFilterContext *ctx)
{
    FormatCUDAContext *s = ctx->priv;

    s->out_format = av_get_pix_fmt(s->out_format_str);
    if (s->out_format == AV_PIX_FMT_NONE) {
        av_log(ctx, AV_LOG_ERROR, "Unrecognized pixel format: %s\n", s->out_format_str);
        return AVERROR(EINVAL);
    }

    return 0;
}


/**
 * Call kernel for a plane
 */
static int format_cuda_kernel(FormatCUDAContext *ctx,
        uint8_t* in_data, int in_linesize,
        uint8_t *out_data, int out_linesize,
        int width, int height, int x_factor) {

    void* kernel_args[] = {
        &in_data, &in_linesize,
        &out_data, &out_linesize,
        &width, &height, &x_factor
    };

    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;

    return CHECK_CU(cu->cuLaunchKernel(
        ctx->cu_func,
        DIV_UP(width / x_factor, BLOCK_X),
        DIV_UP(height, BLOCK_Y),
        1, BLOCK_X, BLOCK_Y, 1, 0, ctx->cu_stream, kernel_args, NULL));
}

/**
 * Filter frame
 */
static int format_cuda_filter_frame(AVFilterLink *link, AVFrame *in) {

    int ret;
    AVFrame *out = NULL;

    AVFilterContext             *avctx = link->dst;
    FormatCUDAContext            *ctx = avctx->priv;
    AVFilterLink             *outlink = avctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)outlink->hw_frames_ctx->data;

    CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
    CUcontext dummy, cuda_ctx = ctx->hwctx->cuda_ctx;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = av_hwframe_get_buffer(ctx->frames_ctx, out, 0);
    if (ret < 0) {
        goto fail;
    }

/*
    av_log(ctx, AV_LOG_ERROR, "%d %d format: %s %s linesize %d %d %d\n",
               in->width, in->height, av_get_pix_fmt_name(in->format),
               av_get_pix_fmt_name(  ((AVHWFramesContext*)in->hw_frames_ctx->data)->sw_format  ),
               in->linesize[0], in->linesize[1], in->linesize[2]);

    av_log(ctx, AV_LOG_ERROR, "%d %d format: %s %s linesize %d %d %d \n",
               out->width, out->height, av_get_pix_fmt_name(out->format),
               av_get_pix_fmt_name(  ((AVHWFramesContext*)out->hw_frames_ctx->data)->sw_format  ),
               out->linesize[0], out->linesize[1], out->linesize[2]);
*/

/*
    av_log(ctx, AV_LOG_ERROR, ">>> %s  %s %s %s %s %s %s bits_per_pixel(%d) \n",
	    av_get_pix_fmt_name (in->format),
        av_get_pix_fmt_name(  ((AVHWFramesContext*)in->hw_frames_ctx->data)->sw_format  ),
        av_color_primaries_name(in->color_primaries), 
        av_color_transfer_name(in->color_trc), 
        av_color_space_name(in->colorspace), 
        av_color_range_name(in->color_range), 
        av_chroma_location_name (in->chroma_location),
        av_get_bits_per_pixel (  av_pix_fmt_desc_get( ((AVHWFramesContext*)in->hw_frames_ctx->data)->sw_format) )
    );
*/

    out->width = in->width;
    out->height = in->height;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0) {
        goto fail;
    }

    switch(ctx->out_format) {

    //
    // p010le -> yuv420p
    //
    case AV_PIX_FMT_YUV420P:

        // Y plane
        format_cuda_kernel(ctx,
            in->data[0], in->linesize[0] / 2,
            out->data[0], out->linesize[0],
            in->width, in->height, 1);

        // U/V -> U plane
        format_cuda_kernel(ctx,
            in->data[1], in->linesize[1] / 2,
            out->data[1], out->linesize[1],
            in->width, in->height / 2, 2);

        // U/V -> V plane
        format_cuda_kernel(ctx,
            in->data[1]+2, in->linesize[1] / 2,
            out->data[2], out->linesize[2],
            in->width, in->height / 2, 2);
        break;

    //
    // p010le -> nv12
    //
    case AV_PIX_FMT_NV12:

       // Y plane
        format_cuda_kernel(ctx,
            in->data[0], in->linesize[0] / 2,
            out->data[0], out->linesize[0],
            in->width, in->height, 1);

        // U/V -> U/V plane
        format_cuda_kernel(ctx,
            in->data[1], in->linesize[1] / 2,
            out->data[1], out->linesize[1],
            in->width, in->height / 2, 1);

        break;

    //
    // Can't convert
    //
    default:
        ret = AVERROR(ENOSYS);
        av_log(ctx, AV_LOG_ERROR, "Unsupportd output pixel format: %s\n", ctx->out_format_str);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0) {
        goto fail;
    }

    ret = av_frame_copy_props(out, in);
    if (ret < 0) {
        goto fail;
    }

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);

fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

/**
 * Uninitialize format_cuda
 */
static av_cold void format_cuda_uninit(AVFilterContext *avctx)
{
    FormatCUDAContext* ctx = avctx->priv;

    if (ctx->hwctx && ctx->cu_module) {
        CUcontext dummy;
        CudaFunctions *cu = ctx->hwctx->internal->cuda_dl;
        CHECK_CU(cu->cuCtxPushCurrent(ctx->hwctx->cuda_ctx));
        CHECK_CU(cu->cuModuleUnload(ctx->cu_module));
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }

    av_buffer_unref(&ctx->frames_ctx);
}



/**
 * Query formats
 */
static int format_cuda_query_formats(AVFilterContext *avctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };

    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);
    return ff_set_common_formats(avctx, pix_fmts);
}

/**
 * Configure output
 */
static int format_cuda_config_output(AVFilterLink *outlink)
{

    extern char vf_format_cuda_ptx[];

    int err;
    AVFilterContext* avctx = outlink->src;
    FormatCUDAContext* ctx = avctx->priv;

    AVFilterLink *inlink = avctx->inputs[0];
    AVHWFramesContext  *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;

    CUcontext dummy, cuda_ctx;
    CudaFunctions *cu;

    AVHWFramesContext *out_ctx = NULL;

    // check main input formats

    if (!frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on main input\n");
        return AVERROR(EINVAL);
    }

    ctx->in_format = frames_ctx->sw_format;
    if (!format_is_supported(supported_formats, ctx->in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(ctx->in_format));
        return AVERROR(ENOSYS);
    }

    // initialize

    ctx->hwctx = frames_ctx->device_ctx->hwctx;
    cuda_ctx = ctx->hwctx->cuda_ctx;

    ctx->cu_stream = ctx->hwctx->stream;
    ctx->device_ref = ((AVHWFramesContext*)inlink->hw_frames_ctx->data)->device_ref;


    ctx->frames_ctx = av_hwframe_ctx_alloc(ctx->device_ref);
    out_ctx = (AVHWFramesContext*)ctx->frames_ctx->data;
    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = ctx->out_format;
    out_ctx->width     = FFALIGN(frames_ctx->width,  32);
    out_ctx->height    = FFALIGN(frames_ctx->height, 32);
    av_hwframe_ctx_init(ctx->frames_ctx);

    outlink->hw_frames_ctx = av_buffer_ref(ctx->frames_ctx);
    // load functions

    cu = ctx->hwctx->internal->cuda_dl;

    err = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (err < 0) {
        return err;
    }

    err = CHECK_CU(cu->cuModuleLoadData(&ctx->cu_module, vf_format_cuda_ptx));
    if (err < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return err;
    }

    err = CHECK_CU(cu->cuModuleGetFunction(&ctx->cu_func, ctx->cu_module, "Format_Cuda_Short_To_Char"));
    if (err < 0) {
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        return err;
    }

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));


    return 0;
}


#define OFFSET(x) offsetof(FormatCUDAContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption format_cuda_options[] = {
    { "format", "Output format",   OFFSET(out_format_str), AV_OPT_TYPE_STRING, { .str = "nv12" }, .flags = FLAGS },
    { NULL },
};

static const AVClass format_cuda_class = {
    .class_name = "format_cuda",
    .item_name  = av_default_item_name,
    .option     = format_cuda_options,
    .version    = LIBAVUTIL_VERSION_INT,
};


static const AVFilterPad format_cuda_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = format_cuda_filter_frame,
    },
    { NULL }
};

static const AVFilterPad format_cuda_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = &format_cuda_config_output,
    },
    { NULL }
};

AVFilter ff_vf_format_cuda = {
    .name            = "format_cuda",
    .description     = NULL_IF_CONFIG_SMALL("Change pixel format on cuda"),
    .priv_size       = sizeof(FormatCUDAContext),
    .priv_class      = &format_cuda_class,
    .init            = &format_cuda_init,
    .uninit          = &format_cuda_uninit,
    .query_formats   = &format_cuda_query_formats,
    .inputs          = format_cuda_inputs,
    .outputs         = format_cuda_outputs,
    .flags_internal  = FF_FILTER_FLAG_HWFRAME_AWARE,
};
