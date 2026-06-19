# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightricks step-distillation mixin for LTX-2.3 pipelines.

Pre-fills the Lightricks single-stage 8-step distilled inference defaults
(``DISTILLED_SIGMA_VALUES`` schedule, ``guidance_scale=1.0``, 8 inference
steps) on top of any LTX-2.3 base pipeline. Mirrors the
:class:`vllm_omni.diffusion.models.dmd2.mixin.DMD2PipelineMixin` pattern but
applies the Lightricks distillation schedule instead of the FastGen DMD2 one.

Compose with any LTX-2.3 mode pipeline (T2V, I2V, Condition, ...) via
multiple inheritance — the mixin **must appear before** the base pipeline in
the MRO::

    class LTX23DistilledPipeline(LightricksDistilledMixin, LTX23Pipeline):
        pass
"""

from __future__ import annotations

from typing import Any

from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest


class LightricksDistilledMixin:
    """Inject Lightricks 8-step distilled defaults into the pipeline ``forward``.

    Defaults applied when not overridden by the caller / request:

    - ``sigmas = DISTILLED_SIGMA_VALUES``
    - ``num_inference_steps = 8``
    - ``guidance_scale = 1.0`` (CFG disabled)

    Explicit kwargs supplied to ``forward()`` always win. The base pipeline's
    own request-payload lookup (e.g. ``req.sampling_params.guidance_scale_provided``)
    continues to apply on top, so a user can still override these per-request.

    Because ``guidance_scale`` defaults to ``1.0`` (CFG disabled), pipelines
    composed with this mixin are not compatible with ``cfg_parallel_size > 1``.
    """

    def forward(
        self,
        req: OmniDiffusionRequest,
        sigmas: list[float] | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> DiffusionOutput:
        if sigmas is None:
            sigmas = DISTILLED_SIGMA_VALUES
        if num_inference_steps is None:
            num_inference_steps = 8
        return super().forward(
            req,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
