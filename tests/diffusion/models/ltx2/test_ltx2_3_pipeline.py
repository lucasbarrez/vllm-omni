# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for LTX-2.3 pipeline integration.

These tests verify:
- Pipeline is properly registered in the diffusion registry
- Post-process function is registered
- Cache-DiT enablers are registered
- Pipeline does NOT inherit from LTX2Pipeline
- Vocoder sample rate detection logic
- Re-export module works correctly
"""

import json
import os
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_ltx23_pipeline(sequence_parallel_size: int = 1):
    from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

    pipeline = object.__new__(LTX23Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae_temporal_compression_ratio = 4
    pipeline.audio_vae_mel_compression_ratio = 4
    pipeline.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=sequence_parallel_size))
    pipeline.audio_vae = SimpleNamespace(
        latents_mean=torch.tensor(0.0),
        latents_std=torch.tensor(1.0),
    )
    return pipeline


def _make_ltx23_request_pipe(cls):
    pipe = object.__new__(cls)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.tokenizer_max_length = 99
    return pipe


def _resolve_request_inputs_for_test(pipe, req):
    return pipe._resolve_request_inputs(
        req,
        prompt=None,
        negative_prompt=None,
        height=None,
        width=None,
        num_frames=None,
        frame_rate=None,
        num_inference_steps=None,
        timesteps=None,
        guidance_scale=4.0,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        audio_latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        decode_timestep=0.0,
        decode_noise_scale=None,
        output_type="np",
        max_sequence_length=None,
    )


class TestLTX23RequestParsing:
    def test_t2v_and_i2v_share_request_input_resolution(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline, LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        prompt_embeds = torch.tensor([[1.0, 2.0]])
        negative_prompt_embeds = torch.tensor([[3.0, 4.0]])
        prompt_attention_mask = torch.tensor([True, False])
        negative_attention_mask = torch.tensor([False, True])
        video_latents = torch.ones(1, 2, 3)
        audio_latents = torch.zeros(1, 2, 3)

        req = OmniDiffusionRequest(
            prompts=[
                {
                    "prompt": "shared prompt",
                    "negative_prompt": "shared negative",
                    "additional_information": {
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds": negative_prompt_embeds,
                        "attention_mask": prompt_attention_mask,
                        "negative_attention_mask": negative_attention_mask,
                    },
                }
            ],
            sampling_params=OmniDiffusionSamplingParams(
                height=384,
                width=512,
                num_frames=25,
                frame_rate=12.5,
                num_inference_steps=1,
                num_outputs_per_prompt=2,
                guidance_scale=6.0,
                seed=123,
                latents=video_latents,
                extra_args={"audio_latents": audio_latents},
                decode_timestep=[0.1],
                decode_noise_scale=[0.2],
                output_type="latent",
                max_sequence_length=17,
            ),
            request_id="ltx23-shared-request-inputs",
        )

        resolved_t2v = _resolve_request_inputs_for_test(
            _make_ltx23_request_pipe(LTX23Pipeline),
            req,
        )
        resolved_i2v = _resolve_request_inputs_for_test(
            _make_ltx23_request_pipe(LTX23ImageToVideoPipeline),
            req,
        )

        assert resolved_i2v.prompt == resolved_t2v.prompt == ["shared prompt"]
        assert resolved_i2v.negative_prompt == resolved_t2v.negative_prompt == ["shared negative"]
        assert resolved_i2v.height == resolved_t2v.height == 384
        assert resolved_i2v.width == resolved_t2v.width == 512
        assert resolved_i2v.num_frames == resolved_t2v.num_frames == 25
        assert resolved_i2v.frame_rate == resolved_t2v.frame_rate == 12.5
        assert resolved_i2v.num_inference_steps == resolved_t2v.num_inference_steps == 2
        assert resolved_i2v.guidance_scale == resolved_t2v.guidance_scale == 6.0
        assert resolved_i2v.num_videos_per_prompt == resolved_t2v.num_videos_per_prompt == 2
        assert resolved_i2v.generator.initial_seed() == resolved_t2v.generator.initial_seed() == 123
        assert resolved_i2v.decode_timestep == resolved_t2v.decode_timestep == [0.1]
        assert resolved_i2v.decode_noise_scale == resolved_t2v.decode_noise_scale == [0.2]
        assert resolved_i2v.output_type == resolved_t2v.output_type == "latent"
        assert resolved_i2v.max_sequence_length == resolved_t2v.max_sequence_length == 17
        torch.testing.assert_close(resolved_i2v.latents, video_latents)
        torch.testing.assert_close(resolved_t2v.latents, video_latents)
        torch.testing.assert_close(resolved_i2v.audio_latents, audio_latents)
        torch.testing.assert_close(resolved_t2v.audio_latents, audio_latents)
        torch.testing.assert_close(resolved_i2v.prompt_embeds, torch.stack([prompt_embeds]))
        torch.testing.assert_close(resolved_t2v.prompt_embeds, torch.stack([prompt_embeds]))
        torch.testing.assert_close(resolved_i2v.negative_prompt_embeds, torch.stack([negative_prompt_embeds]))
        torch.testing.assert_close(resolved_t2v.negative_prompt_embeds, torch.stack([negative_prompt_embeds]))
        torch.testing.assert_close(resolved_i2v.prompt_attention_mask, torch.stack([prompt_attention_mask]))
        torch.testing.assert_close(resolved_t2v.prompt_attention_mask, torch.stack([prompt_attention_mask]))
        torch.testing.assert_close(
            resolved_i2v.negative_prompt_attention_mask,
            torch.stack([negative_attention_mask]),
        )
        torch.testing.assert_close(
            resolved_t2v.negative_prompt_attention_mask,
            torch.stack([negative_attention_mask]),
        )

    def test_request_input_resolution_rejects_mixed_precomputed_prompt_fields(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = _make_ltx23_request_pipe(LTX23Pipeline)
        req = OmniDiffusionRequest(
            prompts=[
                {
                    "prompt": "with embeds",
                    "additional_information": {"prompt_embeds": torch.tensor([[1.0]])},
                },
                {"prompt": "without embeds"},
            ],
            sampling_params=OmniDiffusionSamplingParams(
                height=384,
                width=512,
                num_frames=25,
                num_inference_steps=2,
            ),
            request_id="ltx23-mixed-precomputed-fields",
        )

        with pytest.raises(ValueError, match="prompt_embeds.*every prompt"):
            _resolve_request_inputs_for_test(pipe, req)


class TestPipelineIndependence:
    """Verify LTX23Pipeline is fully independent from LTX2Pipeline."""

    def test_ltx23_pipeline_does_not_inherit_from_ltx2(self):
        """LTX23Pipeline must NOT inherit from LTX2Pipeline."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert not issubclass(LTX23Pipeline, LTX2Pipeline), (
            "LTX23Pipeline should be fully independent and not inherit from LTX2Pipeline"
        )

    def test_ltx23_pipeline_is_nn_module(self):
        """LTX23Pipeline must be an nn.Module."""
        import torch.nn as nn

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert issubclass(LTX23Pipeline, nn.Module)

    def test_ltx23_pipeline_has_progress_bar(self):
        """LTX23Pipeline must mix in ProgressBarMixin."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

        assert issubclass(LTX23Pipeline, ProgressBarMixin)

    def test_ltx23_pipeline_has_cfg_parallel_mixin(self):
        """LTX23Pipeline must use the shared CFG parallel implementation."""
        from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        assert issubclass(LTX23Pipeline, CFGParallelMixin)

    def test_ltx23_pipeline_declares_offload_components(self):
        """LTX23Pipeline must expose LTX-2.3-specific modules to offload discovery."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

        pipe = object.__new__(LTX23Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer = torch.nn.Linear(1, 1)
        pipe.text_encoder = torch.nn.Linear(1, 1)
        pipe.connectors = torch.nn.Linear(1, 1)
        pipe.vae = torch.nn.Linear(1, 1)
        pipe.audio_vae = torch.nn.Linear(1, 1)
        pipe.vocoder = torch.nn.Linear(1, 1)

        modules = ModuleDiscovery.discover(pipe)

        assert LTX23Pipeline._dit_modules == ["transformer"]
        assert LTX23Pipeline._encoder_modules == ["text_encoder", "connectors"]
        assert LTX23Pipeline._vae_modules == ["vae", "audio_vae"]
        assert LTX23Pipeline._resident_modules == ["vocoder"]
        assert modules.dit_names == ["transformer"]
        assert modules.encoder_names == ["text_encoder", "connectors"]
        assert modules.resident_names == ["vocoder"]
        assert len(modules.vaes) == 2

    def test_ltx23_pipeline_has_diffusion_pipeline_profiler_mixin(self):
        """LTX23Pipeline must support lightweight stage timing."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline
        from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin

        assert issubclass(LTX23Pipeline, DiffusionPipelineProfilerMixin)


class TestLTX23ImageToVideoPipeline:
    def test_ltx23_i2v_pipeline_reuses_ltx23_semantics(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline, LTX23Pipeline

        assert issubclass(LTX23ImageToVideoPipeline, LTX23Pipeline)
        assert LTX23ImageToVideoPipeline.support_image_input is True

    def test_ltx23_i2v_rejects_multi_image_prompt_list(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        image = object()

        assert LTX23ImageToVideoPipeline._resolve_single_prompt_image([image]) is image
        with pytest.raises(ValueError, match="exactly one image per prompt"):
            LTX23ImageToVideoPipeline._resolve_single_prompt_image([object(), object()])

    def test_ltx23_i2v_additional_image_resolution_is_tensor_safe(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        image = torch.zeros(1, 3, 4, 4)
        additional = {
            "preprocessed_image": None,
            "pixel_values": image,
            "image": torch.ones_like(image),
        }

        assert LTX23ImageToVideoPipeline._resolve_additional_image(additional) is image

    def test_ltx23_i2v_packed_latents_noise_preserves_conditioning_frame(self, monkeypatch):
        import vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 as ltx23
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae_spatial_compression_ratio = 1
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        def fake_randn_tensor(shape, generator=None, device=None, dtype=None):
            return torch.ones(shape, device=device, dtype=dtype)

        monkeypatch.setattr(ltx23, "randn_tensor", fake_randn_tensor)

        latents = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])

        out, conditioning_mask = pipe.prepare_image_to_video_latents(
            image=None,
            batch_size=1,
            num_channels_latents=2,
            height=1,
            width=1,
            num_frames=3,
            noise_scale=1.0,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        torch.testing.assert_close(conditioning_mask, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(out[:, :1], latents[:, :1])
        torch.testing.assert_close(out[:, 1:], torch.ones_like(out[:, 1:]))

    def test_ltx23_i2v_video_step_preserves_conditioning_frame(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        torch.nn.Module.__init__(pipe)
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        class FakeScheduler:
            def step(self, noise_pred, t, latents, return_dict=False):
                return (latents + noise_pred + t,)

        pipe.scheduler = FakeScheduler()
        latents = torch.tensor([[[1.0], [2.0], [3.0]]])
        noise_pred = torch.full_like(latents, 10.0)

        out = pipe._step_video_latents_i2v(
            noise_pred,
            latents,
            torch.tensor(0.5),
            latent_num_frames=3,
            latent_height=1,
            latent_width=1,
        )

        torch.testing.assert_close(out[:, :1], latents[:, :1])
        torch.testing.assert_close(out[:, 1:], latents[:, 1:] + noise_pred[:, 1:] + 0.5)

    def test_i2v_build_video_timestep_masks_first_frame(self):
        """I2V's video-timestep hook zeros the conditioned tokens (first frame)."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        # I2V mask is binary (1 at first-frame tokens, 0 elsewhere).
        pipe._conditioning_mask = torch.tensor([[1.0, 0.0, 0.0]])

        ts = torch.tensor([7.0])
        out = pipe._build_video_timestep(ts)

        torch.testing.assert_close(out, torch.tensor([[0.0, 7.0, 7.0]]))

    def test_i2v_build_video_timestep_handles_cfg_batch_doubling(self):
        """When CFG doubles the batch (non-CFG-parallel path), the hook
        broadcasts the mask along the batch dim."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        pipe._conditioning_mask = torch.tensor([[1.0, 0.0]])

        ts = torch.tensor([7.0, 7.0])
        out = pipe._build_video_timestep(ts)

        torch.testing.assert_close(out, torch.tensor([[0.0, 7.0], [0.0, 7.0]]))

    def test_i2v_prepare_latents_adapter_stashes_mask(self, monkeypatch):
        """The base-compatible prepare_latents adapter pulls image from
        self._pending_image, calls prepare_image_to_video_latents, and stashes
        the mask on self._conditioning_mask."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline

        pipe = object.__new__(LTX23ImageToVideoPipeline)
        pipe._pending_image = "stand-in-image"

        seen: dict[str, Any] = {}
        sentinel_latents = torch.zeros(1, 3, 4)
        sentinel_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        def fake_prepare(self, *, image, **kwargs):
            seen["image"] = image
            seen["kwargs"] = kwargs
            return sentinel_latents, sentinel_mask

        monkeypatch.setattr(
            LTX23ImageToVideoPipeline, "prepare_image_to_video_latents", fake_prepare
        )

        out = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=2,
            height=8,
            width=8,
            num_frames=3,
        )

        assert seen["image"] == "stand-in-image"
        assert out is sentinel_latents
        assert pipe._conditioning_mask is sentinel_mask

    def test_i2v_prepare_image_to_video_latents_casts_image_to_vae_dtype(self, monkeypatch):
        """The image must be cast to vae.dtype before vae.encode.

        Regression: passing the latent dtype (typically float32) here trips
        the VAE conv layers with "Input type (float) and bias type
        (c10::BFloat16) should be the same" at engine warmup.
        """
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2_3 as ltx23

        pipe = object.__new__(ltx23.LTX23ImageToVideoPipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        seen_encode_dtypes: list[torch.dtype] = []

        def fake_encode(x):
            seen_encode_dtypes.append(x.dtype)
            return SimpleNamespace(
                latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, 4, 1, 1, 1, dtype=x.dtype))
            )

        pipe.vae = SimpleNamespace(
            dtype=torch.bfloat16,
            encode=fake_encode,
            latents_mean=torch.zeros(4),
            latents_std=torch.ones(4),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        monkeypatch.setattr(ltx23, "retrieve_latents", lambda enc, generator, sample_mode: enc.latent_dist.mode())

        image = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        latents, mask = pipe.prepare_image_to_video_latents(
            image=image,
            batch_size=1,
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert seen_encode_dtypes == [torch.bfloat16], (
            f"vae.encode must receive vae.dtype, got {seen_encode_dtypes}"
        )
        # Latents themselves come back in the requested latent dtype.
        assert latents.dtype == torch.float32


class TestLTX23DecodeConditioning:
    def test_decode_conditioning_expands_per_prompt_values_to_effective_batch(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _expand_per_prompt_decode_value

        assert _expand_per_prompt_decode_value(
            [0.1, 0.2],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.1, 0.2, 0.2]
        assert _expand_per_prompt_decode_value(
            [0.3],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.3, 0.3, 0.3, 0.3]
        assert _expand_per_prompt_decode_value(
            [0.1, 0.2, 0.3, 0.4],
            prompt_batch_size=2,
            effective_batch_size=4,
            field_name="decode_timestep",
        ) == [0.1, 0.2, 0.3, 0.4]

    def test_decode_conditioning_rejects_ambiguous_lengths(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _expand_per_prompt_decode_value

        with pytest.raises(ValueError, match="decode_timestep"):
            _expand_per_prompt_decode_value(
                [0.1, 0.2, 0.3],
                prompt_batch_size=2,
                effective_batch_size=4,
                field_name="decode_timestep",
            )


class TestLTX23VaeDecodeParallel:
    """Test LTX-2.3 video VAE tiled parallel helpers without loading weights."""

    def test_ltx23_video_vae_is_distributed_tile_only_class(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

        assert issubclass(DistributedAutoencoderKLLTX2Video, DistributedVaeMixin)
        assert not hasattr(DistributedAutoencoderKLLTX2Video, "patch_split")

    def test_ltx23_video_vae_tile_split_uses_native_ltx23_tile_geometry(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )

        vae = SimpleNamespace(
            spatial_compression_ratio=32,
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=448,
            tile_sample_stride_width=448,
            temporal_compression_ratio=8,
            dtype=torch.float32,
        )

        z = torch.zeros(1, 2, 5, 16, 24)
        tasks, grid_spec = DistributedAutoencoderKLLTX2Video.tile_split(vae, z)

        assert grid_spec.grid_shape == (2, 2)
        assert grid_spec.split_dims == (3, 4)
        assert grid_spec.tile_spec["sample_height"] == 512
        assert grid_spec.tile_spec["sample_width"] == 768
        assert grid_spec.tile_spec["blend_height"] == 64
        assert grid_spec.tile_spec["blend_width"] == 64
        assert grid_spec.tile_spec["max_tile_output_shape"] == (1, 3, 33, 512, 512)
        assert grid_spec.tile_spec["tile_output_shapes"] == {
            0: (1, 3, 33, 512, 512),
            1: (1, 3, 33, 512, 320),
            2: (1, 3, 33, 64, 512),
            3: (1, 3, 33, 64, 320),
        }
        assert [task.grid_coord for task in tasks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert [tuple(task.tensor.shape) for task in tasks] == [
            (1, 2, 5, 16, 16),
            (1, 2, 5, 16, 10),
            (1, 2, 5, 2, 16),
            (1, 2, 5, 2, 10),
        ]
        assert [task.workload for task in tasks] == [5 * 16 * 16, 5 * 16 * 10, 5 * 2 * 16, 5 * 2 * 10]

    def test_ltx23_video_vae_tile_merge_blends_and_crops_like_tiled_decode(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
            DistributedAutoencoderKLLTX2Video,
        )
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import GridSpec

        class FakeVae:
            def __init__(self):
                self.blend_calls = []

            def clear_cache(self):
                pass

            def blend_v(self, _previous, current, blend_height):
                self.blend_calls.append(("v", blend_height))
                return current

            def blend_h(self, _previous, current, blend_width):
                self.blend_calls.append(("h", blend_width))
                return current

        fake_vae = FakeVae()
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "sample_height": 10,
                "sample_width": 10,
                "blend_height": 1,
                "blend_width": 2,
                "tile_sample_stride_height": 5,
                "tile_sample_stride_width": 5,
            },
        )
        tiles = {
            (0, 0): torch.full((1, 3, 2, 6, 6), 1.0),
            (0, 1): torch.full((1, 3, 2, 6, 6), 2.0),
            (1, 0): torch.full((1, 3, 2, 6, 6), 3.0),
            (1, 1): torch.full((1, 3, 2, 6, 6), 4.0),
        }

        merged = DistributedAutoencoderKLLTX2Video.tile_merge(fake_vae, tiles, grid_spec)

        assert merged.shape == (1, 3, 2, 10, 10)
        assert fake_vae.blend_calls == [("h", 2), ("v", 1), ("v", 1), ("h", 2)]
        torch.testing.assert_close(merged[:, :, :, :5, :5], torch.ones(1, 3, 2, 5, 5))
        torch.testing.assert_close(merged[:, :, :, :5, 5:], torch.full((1, 3, 2, 5, 5), 2.0))
        torch.testing.assert_close(merged[:, :, :, 5:, :5], torch.full((1, 3, 2, 5, 5), 3.0))
        torch.testing.assert_close(merged[:, :, :, 5:, 5:], torch.full((1, 3, 2, 5, 5), 4.0))

    def test_ltx23_video_vae_tiled_decode_dispatches_to_tile_operator(self):
        from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_ltx2

        z = torch.zeros(1, 2, 1, 16, 24)
        expected = torch.ones(1, 3, 1, 512, 768)
        seen = {}

        class FakeExecutor:
            def execute(self, tensor, operator, broadcast_result=True):
                seen["tensor"] = tensor
                seen["operator"] = operator
                seen["broadcast_result"] = broadcast_result
                return expected

        vae = SimpleNamespace(distributed_executor=FakeExecutor(), is_distributed_enabled=lambda: True)
        vae.tile_split = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_split.__get__(vae)
        vae.tile_exec = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_exec.__get__(vae)
        vae.tile_merge = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tile_merge.__get__(vae)

        output = autoencoder_kl_ltx2.DistributedAutoencoderKLLTX2Video.tiled_decode(
            vae,
            z,
            temb=torch.tensor(0.5),
            return_dict=False,
        )

        assert len(output) == 1
        assert output[0] is expected
        assert seen["tensor"] is z
        assert seen["broadcast_result"] is False
        assert seen["operator"].split.__name__ == "tile_split"
        assert seen["operator"].merge.__name__ == "tile_merge"

    def test_ltx23_vae_executor_gathers_known_tile_shapes_and_returns_empty_on_non_rank0(self):
        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import LTX2VaeExecutor
        from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
            DistributedOperator,
            GridSpec,
            TileTask,
        )

        z = torch.zeros(1, 1, 1, 1, 1)
        tile_output_shapes = {
            0: (1, 1, 1, 2, 2),
            1: (1, 1, 1, 2, 1),
            2: (1, 1, 1, 1, 2),
            3: (1, 1, 1, 1, 1),
        }
        tasks = [
            TileTask(0, (0, 0), z, workload=4),
            TileTask(1, (0, 1), z, workload=2),
            TileTask(2, (1, 0), z, workload=2),
            TileTask(3, (1, 1), z, workload=1),
        ]
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(2, 2),
            tile_spec={
                "max_tile_output_shape": (1, 1, 1, 2, 2),
                "tile_output_shapes": tile_output_shapes,
            },
            output_dtype=torch.float32,
        )
        seen = {}

        def exec_tile(task):
            return torch.full(tile_output_shapes[task.tile_id], float(task.tile_id + 1))

        def merge_tiles(coord_tensor_map, passed_grid_spec):
            seen["merged_shapes"] = {coord: tuple(tile.shape) for coord, tile in coord_tensor_map.items()}
            assert passed_grid_spec is grid_spec
            return torch.stack(
                [
                    coord_tensor_map[(0, 0)].flatten()[0],
                    coord_tensor_map[(0, 1)].flatten()[0],
                    coord_tensor_map[(1, 0)].flatten()[0],
                    coord_tensor_map[(1, 1)].flatten()[0],
                ]
            )

        operator = DistributedOperator(split=lambda _z: (tasks, grid_spec), exec=exec_tile, merge=merge_tiles)

        rank0_executor = object.__new__(LTX2VaeExecutor)
        rank0_executor.parallel_size = 2
        rank0_executor.world_size = 2
        rank0_executor.rank = 0

        def gather_rank0(local_tile_tensor):
            assigned = rank0_executor._balance_tasks(tasks, 2)
            rank1_results = [(task.tile_id, exec_tile(task)) for task in assigned[1]]
            rank1_tile_tensor = rank0_executor._pack_local_tiles_without_meta(
                rank1_results,
                list(local_tile_tensor.shape),
                z.device,
                torch.float32,
            )
            seen["rank0_gather_shape"] = tuple(local_tile_tensor.shape)
            return [local_tile_tensor, rank1_tile_tensor]

        def fail_final_sync(*_args, **_kwargs):
            raise AssertionError("broadcast_result=False should not sync the final result")

        rank0_executor.gather_tensors = gather_rank0
        rank0_executor._sync_final_result = fail_final_sync

        rank0_result = rank0_executor.execute(z, operator, broadcast_result=False)

        torch.testing.assert_close(rank0_result, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert seen["rank0_gather_shape"] == (2, 1, 1, 1, 2, 2)
        assert seen["merged_shapes"] == {
            (0, 0): (1, 1, 1, 2, 2),
            (0, 1): (1, 1, 1, 2, 1),
            (1, 0): (1, 1, 1, 1, 2),
            (1, 1): (1, 1, 1, 1, 1),
        }

        non_rank0_executor = object.__new__(LTX2VaeExecutor)
        non_rank0_executor.parallel_size = 2
        non_rank0_executor.world_size = 2
        non_rank0_executor.rank = 1

        def gather_rank1(local_tile_tensor):
            seen["rank1_gather_shape"] = tuple(local_tile_tensor.shape)
            return None

        def fail_non_rank0_merge(*_args, **_kwargs):
            raise AssertionError("non-rank0 should not merge gathered tiles")

        non_rank0_executor.gather_tensors = gather_rank1
        non_rank0_executor._sync_final_result = fail_final_sync

        empty_result = non_rank0_executor.execute(
            z,
            DistributedOperator(
                split=lambda _z: (tasks, grid_spec),
                exec=exec_tile,
                merge=fail_non_rank0_merge,
            ),
            broadcast_result=False,
        )

        assert tuple(empty_result.shape) == (0,)
        assert seen["rank1_gather_shape"] == (2, 1, 1, 1, 2, 2)


class TestCFGParallelHelpers:
    """Test LTX-2.3 CFG helper math without loading model weights."""

    def test_combine_cfg_noise_matches_x0_space_formula(self):
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline

        pipe = object.__new__(LTX23Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        video_combined, audio_combined = pipe.combine_cfg_noise(
            (video_pos, audio_pos),
            (video_neg, audio_neg),
            scale,
            video_latents=video_sample,
            audio_latents=audio_sample,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

        x0_video_cond = video_sample - video_pos * video_sigma
        x0_video_uncond = video_sample - video_neg * video_sigma
        x0_video_guided = x0_video_cond + (scale - 1) * (x0_video_cond - x0_video_uncond)
        expected_video = (video_sample - x0_video_guided) / video_sigma

        x0_audio_cond = audio_sample - audio_pos * audio_sigma
        x0_audio_uncond = audio_sample - audio_neg * audio_sigma
        x0_audio_guided = x0_audio_cond + (scale - 1) * (x0_audio_cond - x0_audio_uncond)
        expected_audio = (audio_sample - x0_audio_guided) / audio_sigma
        assert torch.allclose(video_combined, expected_video)
        assert torch.allclose(audio_combined, expected_audio)

    def test_two_rank_cfg_parallel_smoke_uses_rank_local_branch_and_x0_formula(self, monkeypatch):
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2_3 as ltx23

        pipe = object.__new__(ltx23.LTX23Pipeline)
        video_sample = torch.tensor([[[1.0, -2.0]]])
        audio_sample = torch.tensor([[[0.5, 3.0, -1.0]]])
        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2, 0.3]]])
        audio_neg = torch.tensor([[[0.1, 0.4, -0.5]]])
        video_sigma = torch.tensor(0.25)
        audio_sigma = torch.tensor(0.5)
        scale = 4.0

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if tensor.shape == video_pos.shape:
                    return [video_pos, video_neg]
                return [audio_pos, audio_neg]

        monkeypatch.setattr(ltx23, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx23, "get_cfg_group", lambda: FakeCfgGroup())

        expected_video = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            video_sample,
            video_pos,
            video_neg,
            video_sigma,
            scale,
        )
        expected_audio = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            audio_sample,
            audio_pos,
            audio_neg,
            audio_sigma,
            scale,
        )

        for rank, expected_branch in ((0, "positive"), (1, "negative")):
            calls = []
            monkeypatch.setattr(ltx23, "get_classifier_free_guidance_rank", lambda rank=rank: rank)

            def fake_predict_noise(**kwargs):
                calls.append(kwargs["branch"])
                if kwargs["branch"] == "positive":
                    return video_pos, audio_pos
                return video_neg, audio_neg

            object.__setattr__(pipe, "predict_noise", fake_predict_noise)
            video_combined, audio_combined = pipe.predict_noise_with_parallel_cfg(
                true_cfg_scale=scale,
                positive_kwargs={"branch": "positive"},
                negative_kwargs={"branch": "negative"},
                cfg_normalize=False,
                video_latents=video_sample,
                audio_latents=audio_sample,
                video_sigma=video_sigma,
                audio_sigma=audio_sigma,
            )

            assert calls == [expected_branch]
            torch.testing.assert_close(video_combined, expected_video)
            torch.testing.assert_close(audio_combined, expected_audio)

        assert "_cfg_video_latents" not in pipe.__dict__
        assert "_cfg_audio_latents" not in pipe.__dict__


class TestCFGParallelForwardPath:
    """Test the LTX-2.3 CFG-parallel denoising path without loading model weights."""

    @pytest.mark.parametrize(("cfg_rank", "expected_prompt_value"), [(0, 1.0), (1, 0.0)])
    def test_forward_cfg_parallel_steps_video_and_audio_scheduler(
        self,
        monkeypatch,
        cfg_rank,
        expected_prompt_value,
    ):
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2_3 as ltx23
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        pipe = object.__new__(ltx23.LTX23Pipeline)
        torch.nn.Module.__init__(pipe)
        pipe.device = torch.device("cpu")
        pipe.tokenizer_max_length = 1
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1
        pipe.audio_sampling_rate = 1
        pipe.audio_hop_length = 1
        pipe.audio_vae_temporal_compression_ratio = 1
        pipe.audio_vae_mel_compression_ratio = 1
        pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
        pipe.tokenizer = SimpleNamespace(padding_side="left")
        pipe.vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.audio_vae = SimpleNamespace(
            latents_mean=torch.zeros(2),
            latents_std=torch.ones(2),
            config=SimpleNamespace(mel_bins=2, latent_channels=1),
        )

        video_pos = torch.tensor([[[0.2, -0.3]]])
        video_neg = torch.tensor([[[-0.4, 0.1]]])
        audio_pos = torch.tensor([[[0.7, -0.2]]])
        audio_neg = torch.tensor([[[0.1, 0.4]]])

        class FakeCfgGroup:
            def all_gather(self, tensor, separate_tensors=True):
                assert separate_tensors
                if torch.equal(tensor, video_pos) or torch.equal(tensor, video_neg):
                    return [video_pos, video_neg]
                if torch.equal(tensor, audio_pos) or torch.equal(tensor, audio_neg):
                    return [audio_pos, audio_neg]
                raise AssertionError(f"Unexpected gathered tensor: {tensor}")

        monkeypatch.setattr(ltx23, "get_classifier_free_guidance_world_size", lambda: 2)
        monkeypatch.setattr(ltx23, "get_classifier_free_guidance_rank", lambda: cfg_rank)
        monkeypatch.setattr(ltx23, "get_cfg_group", lambda: FakeCfgGroup())

        def fake_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas=None, mu=None):
            scheduler.sigmas = torch.tensor([0.25, 0.25], device=device)
            return torch.tensor([1.0, 0.5], device=device), 2

        monkeypatch.setattr(ltx23, "retrieve_timesteps", fake_retrieve_timesteps)

        class FakeScheduler:
            def __init__(self, name="video", calls=None):
                self.name = name
                self.calls = [] if calls is None else calls
                self.config = {
                    "max_image_seq_len": 4096,
                    "base_image_seq_len": 1024,
                    "base_shift": 0.95,
                    "max_shift": 2.05,
                }
                self.sigmas = torch.tensor([0.25, 0.25])

            def __deepcopy__(self, memo):
                return FakeScheduler("audio", self.calls)

            def step(self, noise_pred, t, latents, return_dict=False, generator=None):
                self.calls.append((self.name, noise_pred.clone(), t.clone(), latents.clone()))
                return (latents - noise_pred,)

        class FakeConnectors:
            def to(self, device):
                return self

            def __call__(self, prompt_embeds, prompt_attention_mask, padding_side):
                assert padding_side == "left"
                assert prompt_embeds.shape[0] == 2
                return prompt_embeds, prompt_embeds, prompt_attention_mask

        class FakeRope:
            def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps):
                return torch.zeros(batch_size, num_frames * height * width, 3, device=device)

            def prepare_audio_coords(self, batch_size, num_frames, device):
                return torch.zeros(batch_size, num_frames, 1, device=device)

        class FakeTransformer:
            def __init__(self):
                self.config = SimpleNamespace(in_channels=2)
                self.rope = FakeRope()
                self.audio_rope = FakeRope()
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                expected_prompt = torch.full((1, 1, 1), expected_prompt_value)
                torch.testing.assert_close(kwargs["encoder_hidden_states"], expected_prompt)
                torch.testing.assert_close(kwargs["audio_encoder_hidden_states"], expected_prompt)
                assert kwargs["hidden_states"].shape == (1, 1, 2)
                assert kwargs["audio_hidden_states"].shape == (1, 1, 2)
                # Regression guard: audio_timestep must be a scalar (B,), not the
                # per-token video timestep. The transformer would otherwise fall
                # back to `timestep` which the I2V hook makes per-token.
                assert "audio_timestep" in kwargs, "base forward must pass audio_timestep explicitly"
                assert kwargs["audio_timestep"].ndim == 1, (
                    f"audio_timestep must be scalar (B,), got shape {tuple(kwargs['audio_timestep'].shape)}"
                )
                if cfg_rank == 0:
                    return video_pos, audio_pos
                return video_neg, audio_neg

        class DummyProgress:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self):
                pass

        pipe.scheduler = FakeScheduler()
        pipe.connectors = FakeConnectors()
        pipe.transformer = FakeTransformer()
        object.__setattr__(pipe, "progress_bar", lambda total: DummyProgress())

        def fake_encode_prompt(**kwargs):
            return (
                torch.ones(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, 1),
                torch.ones(1, 1, dtype=torch.bool),
            )

        object.__setattr__(pipe, "encode_prompt", fake_encode_prompt)

        video_latents = torch.tensor([[[1.0, -2.0]]])
        audio_latents = torch.tensor([[[0.5, 3.0]]])
        req = OmniDiffusionRequest(
            prompts=[{"prompt": "prompt", "negative_prompt": "negative"}],
            sampling_params=OmniDiffusionSamplingParams(
                height=32,
                width=32,
                num_frames=1,
                frame_rate=1.0,
                num_inference_steps=2,
                guidance_scale=4.0,
                latents=video_latents,
                audio_latents=audio_latents,
                output_type="latent",
            ),
            request_id="ltx23-cfg-parallel-forward-test",
        )

        output = pipe.forward(req)

        expected_video_noise = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            video_latents,
            video_pos,
            video_neg,
            pipe.scheduler.sigmas[0],
            4.0,
        )
        expected_audio_noise = ltx23.LTX23Pipeline._combine_x0_space_cfg(
            audio_latents,
            audio_pos,
            audio_neg,
            pipe.scheduler.sigmas[0],
            4.0,
        )
        scheduler_call_names = [call[0] for call in pipe.scheduler.calls]
        assert scheduler_call_names == ["video", "audio", "video", "audio"]
        assert len(pipe.transformer.calls) == 2
        torch.testing.assert_close(pipe.scheduler.calls[0][1], expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[1][1], expected_audio_noise)
        torch.testing.assert_close(pipe.scheduler.calls[2][1], expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[3][1], expected_audio_noise)
        torch.testing.assert_close(pipe.scheduler.calls[2][3], video_latents - expected_video_noise)
        torch.testing.assert_close(pipe.scheduler.calls[3][3], audio_latents - expected_audio_noise)

        video_out, audio_out = output.output
        torch.testing.assert_close(video_out, (video_latents - 2 * expected_video_noise).reshape(1, 2, 1, 1, 1))
        torch.testing.assert_close(audio_out, (audio_latents - 2 * expected_audio_noise).reshape(1, 1, 1, 2))


class TestRegistryIntegration:
    """Verify all LTX-2.3 pipeline variants are registered."""

    def test_pipeline_models_registered(self):
        """LTX-2.3 pipeline variants must be in _DIFFUSION_MODELS."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_MODELS, f"{name} not found in _DIFFUSION_MODELS"

    def test_pipeline_module_paths(self):
        """Registry entries must point to the correct modules."""
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        # T2V -> pipeline_ltx2_3
        assert _DIFFUSION_MODELS["LTX23Pipeline"] == ("ltx2", "pipeline_ltx2_3", "LTX23Pipeline")

        # I2V -> pipeline_ltx2_3_image2video
        assert _DIFFUSION_MODELS["LTX23ImageToVideoPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3_image2video",
            "LTX23ImageToVideoPipeline",
        )

    def test_post_process_funcs_registered(self):
        """Pipeline variants must map to get_ltx2_post_process_func."""
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in _DIFFUSION_POST_PROCESS_FUNCS, f"{name} not in _DIFFUSION_POST_PROCESS_FUNCS"
            assert _DIFFUSION_POST_PROCESS_FUNCS[name] == "get_ltx2_post_process_func"

    def test_cache_dit_enablers_registered(self):
        """Pipeline variants must be registered in CUSTOM_DIT_ENABLERS."""
        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        expected = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected:
            assert name in CUSTOM_DIT_ENABLERS, f"{name} not in CUSTOM_DIT_ENABLERS"


class TestVocoderSampleRateDetection:
    """Test _detect_vocoder_output_sample_rate logic."""

    def test_detects_48khz_from_config(self):
        """Should detect output_sampling_rate=48000 from vocoder/config.json."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000, "input_sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result == 48000

    def test_returns_none_for_no_output_sr(self):
        """Should return None if vocoder config has no output_sampling_rate."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"sampling_rate": 16000}, f)

            result = _detect_vocoder_output_sample_rate(tmpdir)
            assert result is None

    def test_returns_none_for_missing_directory(self):
        """Should return None if vocoder directory doesn't exist."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _detect_vocoder_output_sample_rate

        result = _detect_vocoder_output_sample_rate("/nonexistent/path")
        assert result is None


class TestPostProcessFunction:
    """Test the post-process function factory."""

    def test_post_process_includes_audio_sample_rate(self):
        """Post-process func should include audio_sample_rate when detected."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import get_ltx2_post_process_func

        with tempfile.TemporaryDirectory() as tmpdir:
            vocoder_dir = os.path.join(tmpdir, "vocoder")
            os.makedirs(vocoder_dir)
            with open(os.path.join(vocoder_dir, "config.json"), "w") as f:
                json.dump({"output_sampling_rate": 48000}, f)

            # Create a minimal od_config mock
            class MockConfig:
                model = tmpdir

            func = get_ltx2_post_process_func(MockConfig())

            video = torch.zeros(1, 3, 4, 64, 64)
            audio = torch.zeros(1, 1, 48000)
            result = func((video, audio))

            assert isinstance(result, dict)
            assert "video" in result
            assert "audio" in result
            assert result["audio_sample_rate"] == 48000

    def test_post_process_without_vocoder_config(self):
        """Post-process func should work without vocoder config (no audio_sample_rate key)."""
        import torch

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import get_ltx2_post_process_func

        class MockConfig:
            model = "/nonexistent/path"

        func = get_ltx2_post_process_func(MockConfig())

        video = torch.zeros(1, 3, 4, 64, 64)
        audio = torch.zeros(1, 1, 16000)
        result = func((video, audio))

        assert isinstance(result, dict)
        assert "video" in result
        assert "audio" in result
        assert "audio_sample_rate" not in result


class TestReExportModule:
    """Test that pipeline_ltx2_3_image2video.py correctly re-exports."""

    def test_i2v_classes_importable(self):
        """I2V classes must be importable from the re-export module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import LTX23ImageToVideoPipeline

        assert LTX23ImageToVideoPipeline is not None

    def test_post_process_func_importable(self):
        """get_ltx2_post_process_func must be importable from re-export module."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import get_ltx2_post_process_func

        assert callable(get_ltx2_post_process_func)

    def test_i2v_classes_are_same_as_direct_import(self):
        """Re-exported classes must be the same objects as direct imports."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoPipeline as Direct
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3_image2video import (
            LTX23ImageToVideoPipeline as ReExported,
        )

        assert Direct is ReExported


class TestInitExports:
    """Test that __init__.py exports all LTX-2.3 classes."""

    def test_all_ltx23_classes_exported(self):
        """All LTX-2.3 pipeline classes must be in the ltx2 package __all__."""
        from vllm_omni.diffusion.models import ltx2

        expected_classes = [
            "LTX23Pipeline",
            "LTX23ImageToVideoPipeline",
        ]
        for name in expected_classes:
            assert hasattr(ltx2, name), f"{name} not exported from ltx2 package"
            assert name in ltx2.__all__, f"{name} not in ltx2.__all__"


class TestAudioLatentSPPadding:
    def test_prepare_audio_latents_pads_generated_dummy_length_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=2)

        latents, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=8,
            num_mel_bins=64,
            audio_latent_length=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert original_num_frames == 1
        assert padded_num_frames == 2
        assert latents.shape == (1, 2, 128)

    def test_prepare_audio_latents_pads_provided_packed_sequence_dim_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(40, dtype=torch.float32).view(1, 10, 4)

        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=8,
            audio_latent_length=10,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 4)
        torch.testing.assert_close(padded[:, :10], latents)
        torch.testing.assert_close(padded[:, 10:], torch.zeros(1, 2, 4))

    def test_prepare_audio_latents_accepts_already_padded_4d_latents_for_sp(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.arange(96, dtype=torch.float32).view(1, 2, 12, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)
        padded, original_num_frames, padded_num_frames = pipeline.prepare_audio_latents(
            batch_size=1,
            num_channels_latents=2,
            num_mel_bins=16,
            audio_latent_length=audio_latent_length,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latents=latents,
        )

        assert audio_latent_length == 10
        assert original_num_frames == 10
        assert padded_num_frames == 12
        assert padded.shape == (1, 12, 8)
        torch.testing.assert_close(padded, pipeline._pack_audio_latents(latents))

    def test_resolve_audio_latent_length_preserves_legacy_4d_shape_inference(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 2, 13, 4)

        audio_latent_length = pipeline._resolve_audio_latent_length(10, latents)

        assert audio_latent_length == 13

    def test_prepare_audio_latents_rejects_incompatible_provided_length(self):
        pipeline = _make_ltx23_pipeline(sequence_parallel_size=4)
        latents = torch.zeros(1, 11, 4)

        with pytest.raises(ValueError, match="incompatible audio frame count"):
            pipeline.prepare_audio_latents(
                batch_size=1,
                num_channels_latents=2,
                num_mel_bins=8,
                audio_latent_length=10,
                dtype=torch.float32,
                device=torch.device("cpu"),
                latents=latents,
            )


class TestLightricksDistilledMixin:
    """Tests for the reusable Lightricks step-distillation mixin.

    Verifies that the mixin alone (composed with any base) injects the
    distilled defaults correctly. This is what guarantees future LTX-2.3 mode
    pipelines (I2V, Condition, ...) can derive a distilled variant in 3 lines.
    """

    def test_mixin_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LightricksDistilledMixin")
        assert "LightricksDistilledMixin" in ltx2.__all__

    @staticmethod
    def _make_sampling_params(**overrides):
        defaults = dict(
            num_inference_steps=None,
            guidance_scale=1.0,
            guidance_scale_provided=False,
            do_classifier_free_guidance=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @classmethod
    def _make_request(cls, *, is_dummy_run: bool = False, **sampling_overrides):
        """Build a request namespace shaped like the production OmniDiffusionRequest.

        The mixin's forward branches on ``req.is_dummy_run()`` to skip sanitize
        on the engine's warmup pass, so tests must provide that callable.
        """
        return SimpleNamespace(
            sampling_params=cls._make_sampling_params(**sampling_overrides),
            is_dummy_run=lambda: is_dummy_run,
        )

    def test_mixin_injects_defaults_on_arbitrary_base(self):
        """Compose the mixin with a stub base; defaults must flow to super().forward."""
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin

        captured: dict = {}

        class _Base:
            def forward(self, req, sigmas=None, num_inference_steps=None, guidance_scale=4.0, **kwargs):
                captured["sigmas"] = sigmas
                captured["num_inference_steps"] = num_inference_steps
                captured["guidance_scale"] = guidance_scale
                return SimpleNamespace(output=("v", "a"))

        class _Composed(LightricksDistilledMixin, _Base):
            pass

        req = self._make_request()
        _Composed().forward(req)

        assert captured["sigmas"] is DISTILLED_SIGMA_VALUES
        assert captured["num_inference_steps"] == 8
        assert captured["guidance_scale"] == 1.0

    def test_mixin_sigmas_kwarg_override_is_respected(self):
        """Caller can swap the sigma schedule (e.g. for ablation), but num_inference_steps
        and guidance_scale kwargs are dropped — they belong to the distilled contract."""
        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin

        captured: dict = {}

        class _Base:
            def forward(self, req, sigmas=None, num_inference_steps=None, guidance_scale=4.0, **kwargs):
                captured["sigmas"] = sigmas
                captured["num_inference_steps"] = num_inference_steps
                captured["guidance_scale"] = guidance_scale
                return SimpleNamespace(output=("v", "a"))

        class _Composed(LightricksDistilledMixin, _Base):
            pass

        req = self._make_request()
        _Composed().forward(req, sigmas=[0.9, 0.4], num_inference_steps=12, guidance_scale=3.5)

        assert captured["sigmas"] == [0.9, 0.4]
        assert captured["num_inference_steps"] == 8
        assert captured["guidance_scale"] == 1.0

    def test_mixin_sanitizes_request_payload_overrides(self, caplog):
        """A user POSTing guidance_scale=4.0 + num_inference_steps=30 via the HTTP layer
        must NOT be able to break the distilled defaults. The base pipeline reads from
        req.sampling_params (winning over forward kwargs), so the mixin scrubs the
        request in place and logs a warning per dropped field."""
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin

        captured: dict = {}

        class _Base:
            def forward(self, req, sigmas=None, num_inference_steps=None, guidance_scale=4.0, **kwargs):
                captured["sigmas"] = sigmas
                captured["num_inference_steps"] = num_inference_steps
                captured["guidance_scale"] = guidance_scale
                captured["req_num_steps"] = req.sampling_params.num_inference_steps
                captured["req_guidance"] = req.sampling_params.guidance_scale
                captured["req_provided"] = req.sampling_params.guidance_scale_provided
                captured["req_cfg"] = req.sampling_params.do_classifier_free_guidance
                return SimpleNamespace(output=("v", "a"))

        class _Composed(LightricksDistilledMixin, _Base):
            pass

        req = self._make_request(
            num_inference_steps=30,
            guidance_scale=4.0,
            guidance_scale_provided=True,
            do_classifier_free_guidance=True,
        )

        with caplog.at_level("WARNING", logger="vllm_omni.diffusion.models.ltx2.distilled_mixin"):
            _Composed().forward(req)

        assert captured["sigmas"] is DISTILLED_SIGMA_VALUES
        assert captured["num_inference_steps"] == 8
        assert captured["guidance_scale"] == 1.0
        # Request was scrubbed in place — base pipeline now reads safe values.
        assert captured["req_num_steps"] == 8
        assert captured["req_guidance"] == 1.0
        assert captured["req_provided"] is False
        assert captured["req_cfg"] is False
        # One warning per dropped field.
        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("num_inference_steps=30" in m for m in warnings)
        assert any("guidance_scale=4.00" in m for m in warnings)

    def test_mixin_skips_warnings_when_request_matches_distilled_contract(self, caplog):
        """No warning should be logged when the user did not override the defaults."""
        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin

        class _Base:
            def forward(self, req, **kwargs):
                return SimpleNamespace(output=("v", "a"))

        class _Composed(LightricksDistilledMixin, _Base):
            pass

        req = self._make_request(num_inference_steps=8)

        with caplog.at_level("WARNING", logger="vllm_omni.diffusion.models.ltx2.distilled_mixin"):
            _Composed().forward(req)

        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_mixin_bypasses_sanitize_for_engine_warmup_dummy_run(self, caplog):
        """The engine's internal dummy warmup pass uses num_inference_steps=1
        and guidance_scale=0.0 on purpose to keep cold-start cheap. The mixin
        must NOT sanitize that path — forcing 8 steps would multiply warmup
        cost without benefit, and the warning would mislead users into
        thinking a client sent num_steps=1."""
        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin

        captured: dict = {}

        class _Base:
            def forward(self, req, sigmas=None, num_inference_steps=None, guidance_scale=4.0, **kwargs):
                captured["sigmas"] = sigmas
                captured["num_inference_steps"] = num_inference_steps
                captured["guidance_scale"] = guidance_scale
                captured["req_num_steps"] = req.sampling_params.num_inference_steps
                captured["req_guidance"] = req.sampling_params.guidance_scale
                return SimpleNamespace(output=("v", "a"))

        class _Composed(LightricksDistilledMixin, _Base):
            pass

        # Mirror what DiffusionEngine._dummy_run constructs.
        req = SimpleNamespace(
            sampling_params=self._make_sampling_params(num_inference_steps=1, guidance_scale=0.0),
            is_dummy_run=lambda: True,
        )

        with caplog.at_level("WARNING", logger="vllm_omni.diffusion.models.ltx2.distilled_mixin"):
            _Composed().forward(req, num_inference_steps=1, guidance_scale=0.0)

        # No warnings: the warmup path is exempt by design.
        assert not [r for r in caplog.records if r.levelname == "WARNING"]
        # Sanitize was skipped: the request keeps its warmup-minimal values.
        assert captured["req_num_steps"] == 1
        assert captured["req_guidance"] == 0.0
        # Base receives the warmup's own kwargs untouched (mixin did not force 8 / 1.0).
        assert captured["num_inference_steps"] == 1
        assert captured["guidance_scale"] == 0.0


class TestLTX23DistilledPipeline:
    """Tests for the LTX-2.3 Lightricks-distilled T2V variant.

    Covers registry wiring, package exports, MRO ordering, and the inherited
    default injection behavior — most of the forward-behavior coverage lives
    in :class:`TestLightricksDistilledMixin`.
    """

    def test_subclasses_ltx23_pipeline(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23DistilledPipeline, LTX23Pipeline

        assert issubclass(LTX23DistilledPipeline, LTX23Pipeline)

    def test_mixin_appears_before_base_in_mro(self):
        """LightricksDistilledMixin must be in front of LTX23Pipeline so its
        forward() wins resolution and the super() chain reaches the base."""
        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23DistilledPipeline, LTX23Pipeline

        mro = LTX23DistilledPipeline.__mro__
        assert mro.index(LightricksDistilledMixin) < mro.index(LTX23Pipeline)

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23DistilledPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23DistilledPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert _DIFFUSION_POST_PROCESS_FUNCS["LTX23DistilledPipeline"] == "get_ltx2_post_process_func"

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LTX23DistilledPipeline")
        assert "LTX23DistilledPipeline" in ltx2.__all__

    def test_forward_injects_distilled_defaults_via_mixin(self, monkeypatch):
        """Defaults flow through LTX23Pipeline.forward via the mixin's super() call."""
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23DistilledPipeline, LTX23Pipeline

        captured: dict = {}

        def fake_super_forward(self, req, sigmas=None, num_inference_steps=None, guidance_scale=4.0, **kwargs):
            captured["sigmas"] = sigmas
            captured["num_inference_steps"] = num_inference_steps
            captured["guidance_scale"] = guidance_scale
            return SimpleNamespace(output=("video", "audio"))

        monkeypatch.setattr(LTX23Pipeline, "forward", fake_super_forward)

        pipe = object.__new__(LTX23DistilledPipeline)
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(
                num_inference_steps=None,
                guidance_scale=1.0,
                guidance_scale_provided=False,
                do_classifier_free_guidance=False,
            ),
            is_dummy_run=lambda: False,
        )

        pipe.forward(req)

        assert captured["sigmas"] is DISTILLED_SIGMA_VALUES
        assert captured["num_inference_steps"] == 8
        assert captured["guidance_scale"] == 1.0


class TestLTX23ImageToVideoDistilledPipeline:
    """Tests for the LTX-2.3 Lightricks-distilled I2V variant.

    Same shape as :class:`TestLTX23DistilledPipeline` — verifies that the
    reusable mixin composes cleanly on top of the real I2V base.
    """

    def test_subclasses_ltx23_image_to_video_pipeline(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ImageToVideoDistilledPipeline,
            LTX23ImageToVideoPipeline,
        )

        assert issubclass(LTX23ImageToVideoDistilledPipeline, LTX23ImageToVideoPipeline)

    def test_mixin_appears_before_base_in_mro(self):
        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ImageToVideoDistilledPipeline,
            LTX23ImageToVideoPipeline,
        )

        mro = LTX23ImageToVideoDistilledPipeline.__mro__
        assert mro.index(LightricksDistilledMixin) < mro.index(LTX23ImageToVideoPipeline)

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23ImageToVideoDistilledPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23ImageToVideoDistilledPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert (
            _DIFFUSION_POST_PROCESS_FUNCS["LTX23ImageToVideoDistilledPipeline"]
            == "get_ltx2_post_process_func"
        )

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LTX23ImageToVideoDistilledPipeline")
        assert "LTX23ImageToVideoDistilledPipeline" in ltx2.__all__

    def test_forward_injects_distilled_defaults_via_mixin(self, monkeypatch):
        """Defaults flow through LTX23ImageToVideoPipeline.forward via the mixin's super() call."""
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ImageToVideoDistilledPipeline,
            LTX23ImageToVideoPipeline,
        )

        captured: dict = {}

        def fake_super_forward(self, req, sigmas=None, num_inference_steps=None, guidance_scale=4.0, **kwargs):
            captured["sigmas"] = sigmas
            captured["num_inference_steps"] = num_inference_steps
            captured["guidance_scale"] = guidance_scale
            return SimpleNamespace(output=("video", "audio"))

        monkeypatch.setattr(LTX23ImageToVideoPipeline, "forward", fake_super_forward)

        pipe = object.__new__(LTX23ImageToVideoDistilledPipeline)
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(
                num_inference_steps=None,
                guidance_scale=1.0,
                guidance_scale_provided=False,
                do_classifier_free_guidance=False,
            ),
            is_dummy_run=lambda: False,
        )

        pipe.forward(req)

        assert captured["sigmas"] is DISTILLED_SIGMA_VALUES
        assert captured["num_inference_steps"] == 8
        assert captured["guidance_scale"] == 1.0


class TestLTX2VideoCondition:
    """Tests for the LTX2VideoCondition dataclass."""

    def test_default_values(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX2VideoCondition

        dummy = SimpleNamespace()  # any object stands in for the frames field
        cond = LTX2VideoCondition(frames=dummy)
        assert cond.index == 0
        assert cond.strength == 1.0
        assert cond.frames is dummy


class TestPreprocessConditions:
    """Tests for the _preprocess_conditions helper."""

    def _stub_video_processor(self, observed: list):
        class _StubProcessor:
            def preprocess_video(self, frames, height, width):
                observed.append((frames, height, width))
                tensor = torch.zeros(1)
                return tensor

        return _StubProcessor()

    def test_empty_conditions_raises(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import _preprocess_conditions

        with pytest.raises(ValueError, match="at least one condition"):
            _preprocess_conditions(
                [],
                self._stub_video_processor([]),
                height=64,
                width=64,
                latent_num_frames=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_out_of_range_index_raises(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX2VideoCondition, _preprocess_conditions

        with pytest.raises(ValueError, match="outside the valid range"):
            _preprocess_conditions(
                [LTX2VideoCondition(frames=SimpleNamespace(), index=10)],
                self._stub_video_processor([]),
                height=64,
                width=64,
                latent_num_frames=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_invalid_strength_raises(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX2VideoCondition, _preprocess_conditions

        with pytest.raises(ValueError, match="strength must be in"):
            _preprocess_conditions(
                [LTX2VideoCondition(frames=SimpleNamespace(), index=0, strength=1.5)],
                self._stub_video_processor([]),
                height=64,
                width=64,
                latent_num_frames=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_negative_index_resolves_against_num_frames(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX2VideoCondition, _preprocess_conditions

        observed: list = []
        _, _, indices = _preprocess_conditions(
            [
                LTX2VideoCondition(frames=SimpleNamespace(), index=0),
                LTX2VideoCondition(frames=SimpleNamespace(), index=-1),
            ],
            self._stub_video_processor(observed),
            height=64,
            width=64,
            latent_num_frames=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert indices == [0, 4]


class TestApplyVisualConditioning:
    """Tests for the apply_visual_conditioning static method."""

    def test_writes_at_correct_token_offsets(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline

        # Latents: [B=1, seq=10, dim=2]. Use a 5x2 grid (latent_height * latent_width = 10).
        latents = torch.zeros(1, 10, 2)
        mask = torch.zeros(1, 10, 1)
        cond = torch.full((1, 2, 2), 7.0)  # 2 tokens of value 7.0

        latents_out, mask_out, clean_out = LTX23ConditionPipeline.apply_visual_conditioning(
            latents,
            mask,
            condition_latents=[cond],
            condition_strengths=[1.0],
            condition_indices=[1],
            latent_height=1,
            latent_width=2,
        )

        # latent_idx=1 * latent_height=1 * latent_width=2 = 2 → tokens [2:4]
        assert torch.allclose(latents_out[:, 2:4], torch.full((1, 2, 2), 7.0))
        assert torch.allclose(mask_out[:, 2:4], torch.ones(1, 2, 1))
        assert torch.allclose(clean_out[:, 2:4], torch.full((1, 2, 2), 7.0))
        # Outside the conditioned region: unchanged.
        assert torch.allclose(latents_out[:, :2], torch.zeros(1, 2, 2))
        assert torch.allclose(latents_out[:, 4:], torch.zeros(1, 6, 2))


class TestLTX23ConditionPipeline:
    """Tests for the LTX-2.3 multi-anchor frame conditioning pipeline."""

    def test_subclasses_ltx23_pipeline(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline, LTX23Pipeline

        assert issubclass(LTX23ConditionPipeline, LTX23Pipeline)

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23ConditionPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23ConditionPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert (
            _DIFFUSION_POST_PROCESS_FUNCS["LTX23ConditionPipeline"]
            == "get_ltx2_post_process_func"
        )

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        for name in ("LTX23ConditionPipeline", "LTX2VideoCondition"):
            assert hasattr(ltx2, name), f"{name} not exported"
            assert name in ltx2.__all__, f"{name} not in __all__"

    def test_forward_falls_back_to_t2v_when_no_conditions(self, monkeypatch):
        """With conditions=None or [], the pipeline behaves like LTX23Pipeline."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline, LTX23Pipeline

        called = {}

        def fake_super_forward(self, req, **kwargs):
            called["yes"] = True
            return SimpleNamespace(output=("video", "audio"))

        monkeypatch.setattr(LTX23Pipeline, "forward", fake_super_forward)

        pipe = object.__new__(LTX23ConditionPipeline)
        # `prompts=[]` keeps the request-side resolver a no-op so the kwarg path
        # is the only signal in this test.
        req = SimpleNamespace(prompts=[])

        pipe.forward(req, conditions=None)
        assert called == {"yes": True}

        called.clear()
        pipe.forward(req, conditions=[])
        assert called == {"yes": True}

    def test_forward_stashes_conditions_before_delegating(self, monkeypatch):
        """With non-empty conditions, forward stashes them on self and delegates
        to LTX23Pipeline.forward so the hooks pick them up."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionPipeline,
            LTX23Pipeline,
            LTX2VideoCondition,
        )

        seen: dict[str, Any] = {}

        def fake_super_forward(self, req, **kwargs):
            seen["pending"] = self._pending_conditions
            return SimpleNamespace(output=("video", "audio"))

        monkeypatch.setattr(LTX23Pipeline, "forward", fake_super_forward)

        pipe = object.__new__(LTX23ConditionPipeline)
        req = SimpleNamespace(prompts=[])
        conditions = [LTX2VideoCondition(frames=SimpleNamespace())]

        pipe.forward(req, conditions=conditions)
        assert seen["pending"] is conditions
        # State is cleaned up after forward returns.
        assert pipe._pending_conditions is None
        assert pipe._conditioning_mask is None
        assert pipe._clean_latents is None

    def test_resolve_conditions_from_request_duck_types_anchors(self):
        """`_resolve_conditions_from_request` accepts any object exposing
        ``data`` / ``index`` / ``strength`` (the ReferenceConditionAnchor shape
        the serving layer stashes in multi_modal_data)."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionPipeline,
            LTX2VideoCondition,
        )

        anchor_first = SimpleNamespace(data="<first-pil>", index=0, strength=1.0)
        anchor_last = SimpleNamespace(data="<last-pil>", index=-1, strength=0.5)
        req = SimpleNamespace(
            prompts=[{"multi_modal_data": {"conditions": [anchor_first, anchor_last]}}]
        )

        resolved = LTX23ConditionPipeline._resolve_conditions_from_request(req)
        assert resolved is not None
        assert len(resolved) == 2
        assert all(isinstance(c, LTX2VideoCondition) for c in resolved)
        assert resolved[0].frames == "<first-pil>"
        assert resolved[0].index == 0
        assert resolved[0].strength == 1.0
        assert resolved[1].index == -1
        assert resolved[1].strength == 0.5

    def test_resolve_conditions_passes_through_ltx2videocondition(self):
        """Pre-built LTX2VideoCondition instances are passed through unchanged."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionPipeline,
            LTX2VideoCondition,
        )

        cond = LTX2VideoCondition(frames=SimpleNamespace(), index=3, strength=0.8)
        req = SimpleNamespace(prompts=[{"multi_modal_data": {"conditions": [cond]}}])

        resolved = LTX23ConditionPipeline._resolve_conditions_from_request(req)
        assert resolved == [cond]
        # Same object: pipeline didn't copy or rewrap.
        assert resolved[0] is cond

    def test_resolve_conditions_returns_none_when_absent(self):
        """No conditions stashed → resolver returns None (no exception)."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline

        # Empty prompts list.
        assert LTX23ConditionPipeline._resolve_conditions_from_request(
            SimpleNamespace(prompts=[])
        ) is None

        # multi_modal_data missing the key.
        req = SimpleNamespace(prompts=[{"multi_modal_data": {"image": "<some>"}}])
        assert LTX23ConditionPipeline._resolve_conditions_from_request(req) is None

        # String prompt (chat-style).
        assert LTX23ConditionPipeline._resolve_conditions_from_request(
            SimpleNamespace(prompts=["a plain string prompt"])
        ) is None

    def test_forward_reads_conditions_from_request_when_kwarg_absent(self, monkeypatch):
        """The HTTP path stashes anchors in multi_modal_data; forward must pick
        them up when no `conditions` kwarg is supplied (which is what
        DiffusionEngine.run does)."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionPipeline,
            LTX23Pipeline,
            LTX2VideoCondition,
        )

        seen: dict[str, Any] = {}

        def fake_super_forward(self, req, **kwargs):
            seen["pending"] = self._pending_conditions
            return SimpleNamespace(output=("video", "audio"))

        monkeypatch.setattr(LTX23Pipeline, "forward", fake_super_forward)

        pipe = object.__new__(LTX23ConditionPipeline)
        anchor = SimpleNamespace(data="<pil-image>", index=0, strength=1.0)
        req = SimpleNamespace(prompts=[{"multi_modal_data": {"conditions": [anchor]}}])

        pipe.forward(req)  # no kwarg!

        pending = seen["pending"]
        assert pending is not None
        assert len(pending) == 1
        assert isinstance(pending[0], LTX2VideoCondition)
        assert pending[0].frames == "<pil-image>"

    def test_forward_kwarg_conditions_take_precedence_over_request(self, monkeypatch):
        """Explicit kwarg conditions win over any request-side anchors."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionPipeline,
            LTX23Pipeline,
            LTX2VideoCondition,
        )

        seen: dict[str, Any] = {}

        def fake_super_forward(self, req, **kwargs):
            seen["pending"] = self._pending_conditions
            return SimpleNamespace(output=("video", "audio"))

        monkeypatch.setattr(LTX23Pipeline, "forward", fake_super_forward)

        pipe = object.__new__(LTX23ConditionPipeline)
        req_anchor = SimpleNamespace(data="<req-pil>", index=0, strength=1.0)
        req = SimpleNamespace(prompts=[{"multi_modal_data": {"conditions": [req_anchor]}}])
        explicit = [LTX2VideoCondition(frames="<explicit>", index=5, strength=0.3)]

        pipe.forward(req, conditions=explicit)

        assert seen["pending"] is explicit
        assert seen["pending"][0].frames == "<explicit>"

    def test_build_video_timestep_masks_conditioned_tokens(self):
        """The video-timestep hook should zero out conditioned tokens and pass
        the scalar timestep through for unconditioned tokens."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline

        pipe = object.__new__(LTX23ConditionPipeline)
        # mask shape (B, N): token 0 fully conditioned, token 1 half-strength,
        # token 2 unconditioned.
        pipe._conditioning_mask = torch.tensor([[1.0, 0.5, 0.0]])

        ts = torch.tensor([10.0])
        out = pipe._build_video_timestep(ts)

        # (1, 1) * (1 - (1, 3)) = (1, 3) per-token timestep.
        torch.testing.assert_close(out, torch.tensor([[0.0, 5.0, 10.0]]))

    def test_build_video_timestep_duplicates_mask_for_non_parallel_cfg(self):
        """When ts is CFG-duplicated (batch x2) but the mask isn't, the hook
        broadcasts the mask along the batch dim."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline

        pipe = object.__new__(LTX23ConditionPipeline)
        pipe._conditioning_mask = torch.tensor([[1.0, 0.0]])  # (1, 2)

        ts = torch.tensor([10.0, 10.0])  # (2,) — CFG batch=2
        out = pipe._build_video_timestep(ts)

        torch.testing.assert_close(
            out, torch.tensor([[0.0, 10.0], [0.0, 10.0]])
        )

    def test_build_video_timestep_passthrough_without_mask(self):
        """Without a conditioning mask stashed, the hook is a passthrough so
        the T2V code path is unaffected."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionPipeline

        pipe = object.__new__(LTX23ConditionPipeline)
        pipe._conditioning_mask = None

        ts = torch.tensor([1.0, 2.0])
        out = pipe._build_video_timestep(ts)
        torch.testing.assert_close(out, ts)

    def test_condition_video_audio_scheduler_blends_in_x0_space(self):
        """_ConditionVideoAudioScheduler.step applies the x0-blend formula
            x0_blend = (sample - v*sigma) * (1 - m) + clean * m
        then converts back to velocity before delegating to scheduler.step."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionPipeline,
            _ConditionVideoAudioScheduler,
        )

        pipe = object.__new__(LTX23ConditionPipeline)
        # Single batch, 2 tokens, 1 channel.
        pipe._conditioning_mask = torch.tensor([[1.0, 0.0]])
        pipe._clean_latents = torch.tensor([[[7.0], [0.0]]])

        captured = {}

        class FakeVideoScheduler:
            step_index = 0
            sigmas = torch.tensor([2.0, 1.0, 0.0])

            def _init_step_index(self, _t):  # pragma: no cover - already set
                pass

            def step(self, noise_pred, t, latents, return_dict=False, generator=None):
                captured["video_noise_pred"] = noise_pred
                return (latents - noise_pred,)

        class FakeAudioScheduler:
            def step(self, noise_pred, t, latents, return_dict=False, generator=None):
                return (latents + noise_pred,)

        pipe.scheduler = FakeVideoScheduler()
        wrapper = _ConditionVideoAudioScheduler(pipe, FakeAudioScheduler())

        # sample at token 0 = 5, token 1 = 3. velocity pred = 0.5 everywhere.
        # sigma at step 0 = 2.0
        # x0       = sample - v*sigma = [[[5 - 1.0], [3 - 1.0]]] = [[[4.0], [2.0]]]
        # mask     = [[1, 0]] → mask_3d = [[[1], [0]]]
        # blended  = x0 * (1 - mask) + clean * mask = [[[7.0], [2.0]]]
        # v_corr   = (sample - blended) / sigma = [[[ -1.0], [0.5]]]
        sample = torch.tensor([[[5.0], [3.0]]])
        noise_pred_video = torch.tensor([[[0.5], [0.5]]])
        noise_pred_audio = torch.tensor([[[9.0]]])
        out, = wrapper.step(
            (noise_pred_video, noise_pred_audio),
            (torch.tensor(0.0), torch.tensor(0.0)),
            (sample, torch.tensor([[[1.0]]])),
        )

        torch.testing.assert_close(
            captured["video_noise_pred"], torch.tensor([[[-1.0], [0.5]]])
        )
        # video out = sample - v_corr = [[[6.0], [2.5]]]
        torch.testing.assert_close(out[0], torch.tensor([[[6.0], [2.5]]]))

    def test_prepare_condition_latents_casts_condition_to_vae_dtype(self, monkeypatch):
        """Each condition_tensor must be cast to vae.dtype before vae.encode.

        Same regression as the I2V image path: passing the latent dtype
        (float32) here trips the VAE conv layers at engine warmup.
        """
        from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2_3 as ltx23

        pipe = object.__new__(ltx23.LTX23ConditionPipeline)
        torch.nn.Module.__init__(pipe)
        pipe.vae_spatial_compression_ratio = 32
        pipe.vae_temporal_compression_ratio = 1
        pipe.transformer_spatial_patch_size = 1
        pipe.transformer_temporal_patch_size = 1

        seen_encode_dtypes: list[torch.dtype] = []

        def fake_encode(x):
            seen_encode_dtypes.append(x.dtype)
            return SimpleNamespace(
                latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, 4, 1, 1, 1, dtype=x.dtype))
            )

        pipe.vae = SimpleNamespace(
            dtype=torch.bfloat16,
            encode=fake_encode,
            latents_mean=torch.zeros(4),
            latents_std=torch.ones(4),
            config=SimpleNamespace(scaling_factor=1.0),
        )
        pipe.video_processor = SimpleNamespace()

        condition = ltx23.LTX2VideoCondition(
            frames=SimpleNamespace(), index=0, strength=1.0
        )

        def fake_preprocess(conditions, video_processor, height, width, latent_num_frames, *, device, dtype):
            # Emulate the real path: return a float32 condition tensor.
            return ([torch.zeros(1, 3, 1, 1, 1, dtype=dtype)], [1.0], [0])

        monkeypatch.setattr(ltx23, "_preprocess_conditions", fake_preprocess)
        monkeypatch.setattr(ltx23, "retrieve_latents", lambda enc, generator, sample_mode: enc.latent_dist.mode())
        monkeypatch.setattr(
            ltx23.LTX23ConditionPipeline,
            "apply_visual_conditioning",
            lambda self, latents, mask, *_args, **_kw: (latents, mask, torch.zeros_like(latents)),
        )

        pipe.prepare_condition_latents(
            conditions=[condition],
            batch_size=1,
            num_channels_latents=4,
            height=32,
            width=32,
            num_frames=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        assert seen_encode_dtypes == [torch.bfloat16], (
            f"vae.encode must receive vae.dtype, got {seen_encode_dtypes}"
        )


class TestLTX23ConditionDistilledPipeline:
    """Tests for the distilled variant composed via the mixin."""

    def test_subclasses_condition_pipeline(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionDistilledPipeline,
            LTX23ConditionPipeline,
        )

        assert issubclass(LTX23ConditionDistilledPipeline, LTX23ConditionPipeline)

    def test_mixin_appears_before_base_in_mro(self):
        from vllm_omni.diffusion.models.ltx2.distilled_mixin import LightricksDistilledMixin
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
            LTX23ConditionDistilledPipeline,
            LTX23ConditionPipeline,
        )

        mro = LTX23ConditionDistilledPipeline.__mro__
        assert mro.index(LightricksDistilledMixin) < mro.index(LTX23ConditionPipeline)

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23ConditionDistilledPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23ConditionDistilledPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert (
            _DIFFUSION_POST_PROCESS_FUNCS["LTX23ConditionDistilledPipeline"]
            == "get_ltx2_post_process_func"
        )

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LTX23ConditionDistilledPipeline")
        assert "LTX23ConditionDistilledPipeline" in ltx2.__all__


class TestLTX23TwoStagesPipeline:
    """Tests for the LTX-2.3 two-stage T2V pipeline (1080p / 1440p refine)."""

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23TwoStagesPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23TwoStagesPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert (
            _DIFFUSION_POST_PROCESS_FUNCS["LTX23TwoStagesPipeline"]
            == "get_ltx2_post_process_func"
        )

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LTX23TwoStagesPipeline")
        assert "LTX23TwoStagesPipeline" in ltx2.__all__

    def test_lora_filename_targets_v1_1(self):
        """The class-level stage-2 LoRA filename must point at the v1.1 adapter."""
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23TwoStagesPipeline

        assert LTX23TwoStagesPipeline._STAGE_2_LORA_FILENAME == (
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )

    @pytest.mark.parametrize(
        ("model_path", "expected_distilled"),
        [
            # The official HF repo uses a capital D — detection must be
            # case-insensitive so this path triggers the distilled fast path
            # (skip LoRA load) rather than crashing on a missing LoRA file.
            ("/some/local/path/LTX-2.3-Distilled-Diffusers", True),
            ("/some/local/path/LTX-2.3-distilled-Diffusers", True),
            ("/some/local/path/LTX-2.3-DISTILLED-Diffusers", True),
            # Dev / non-distilled checkpoints — must NOT be flagged as distilled
            # because the LoRA load path is required to reach the distilled
            # behavior at stage 2.
            ("/some/local/path/LTX-2.3-Diffusers", False),
            ("/some/local/path/dg845-LTX-2.3-Diffusers", False),
        ],
    )
    def test_distilled_flag_detection_is_case_insensitive(self, model_path, expected_distilled):
        """Detection of the distilled vs dev branch must be case-insensitive.

        Mirrors the inline computation in ``LTX23TwoStagesPipeline.__init__``;
        if the production logic drops ``.lower()`` again, this test fails on
        the capital-D HF repo name.
        """
        import os

        detected = "distilled" in os.path.basename(os.path.normpath(model_path)).lower()
        assert detected is expected_distilled


class TestLTX23ImageToVideoTwoStagesPipeline:
    """Tests for the LTX-2.3 two-stage I2V pipeline (distilled-only)."""

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23ImageToVideoTwoStagesPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23ImageToVideoTwoStagesPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert (
            _DIFFUSION_POST_PROCESS_FUNCS["LTX23ImageToVideoTwoStagesPipeline"]
            == "get_ltx2_post_process_func"
        )

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LTX23ImageToVideoTwoStagesPipeline")
        assert "LTX23ImageToVideoTwoStagesPipeline" in ltx2.__all__

    def test_supports_image_input_class_attribute(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ImageToVideoTwoStagesPipeline

        assert LTX23ImageToVideoTwoStagesPipeline.support_image_input is True


class TestLTX23ConditionTwoStagesPipeline:
    """Tests for the LTX-2.3 two-stage Condition pipeline (distilled-only)."""

    def test_registered_in_diffusion_models(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        assert _DIFFUSION_MODELS["LTX23ConditionTwoStagesPipeline"] == (
            "ltx2",
            "pipeline_ltx2_3",
            "LTX23ConditionTwoStagesPipeline",
        )

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert (
            _DIFFUSION_POST_PROCESS_FUNCS["LTX23ConditionTwoStagesPipeline"]
            == "get_ltx2_post_process_func"
        )

    def test_exported_from_ltx2_package(self):
        from vllm_omni.diffusion.models import ltx2

        assert hasattr(ltx2, "LTX23ConditionTwoStagesPipeline")
        assert "LTX23ConditionTwoStagesPipeline" in ltx2.__all__

    def test_supports_image_input_class_attribute(self):
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionTwoStagesPipeline

        assert LTX23ConditionTwoStagesPipeline.support_image_input is True

    def test_distilled_only_guard(self):
        """A non-distilled model path must raise NotImplementedError at init time."""
        from types import SimpleNamespace

        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23ConditionTwoStagesPipeline

        od_config = SimpleNamespace(model="dg845/LTX-2.3-Diffusers", max_cpu_loras=1)
        with pytest.raises(NotImplementedError, match="requires a distilled checkpoint"):
            LTX23ConditionTwoStagesPipeline(od_config=od_config)
