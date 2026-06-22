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

import logging
from typing import Any

from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = logging.getLogger(__name__)

_DISTILLED_NUM_INFERENCE_STEPS = 8
_DISTILLED_GUIDANCE_SCALE = 1.0


class LightricksDistilledMixin:
    """Inject Lightricks 8-step distilled defaults into the pipeline ``forward``.

    Defaults enforced on every call:

    - ``sigmas = DISTILLED_SIGMA_VALUES``
    - ``num_inference_steps = 8``
    - ``guidance_scale = 1.0`` (CFG disabled)

    The base pipeline resolves ``num_inference_steps`` and ``guidance_scale``
    from ``req.sampling_params`` and would otherwise silently override the
    mixin's kwargs — so we sanitize the request *in place* before delegating
    to ``super().forward()`` and log a warning when caller-supplied values are
    dropped. ``sigmas`` is never read from the request payload, so the
    schedule is inattackable from the HTTP layer.

    Because ``guidance_scale`` is forced to ``1.0`` (CFG disabled), pipelines
    composed with this mixin are not compatible with ``cfg_parallel_size > 1``.
    """

    def _sanitize_lightricks_request(self, req: OmniDiffusionRequest) -> None:
        """Force Lightricks distilled defaults on ``req.sampling_params``.

        Mirrors :meth:`DMD2PipelineMixin._sanitize_dmd2_request` but skips
        the fields that LTX-2.3 does not consume (``guidance_scale_2``,
        ``extra_args.sample_solver``). Negative prompts are left intact
        because the base pipeline already gates negative-prompt encoding on
        ``do_classifier_free_guidance``, which is off once ``guidance_scale``
        is forced to 1.0.
        """
        sp = req.sampling_params

        if sp.num_inference_steps and sp.num_inference_steps != _DISTILLED_NUM_INFERENCE_STEPS:
            logger.warning(
                "Lightricks distilled: ignoring num_inference_steps=%d, forcing %d.",
                sp.num_inference_steps,
                _DISTILLED_NUM_INFERENCE_STEPS,
            )
        sp.num_inference_steps = _DISTILLED_NUM_INFERENCE_STEPS

        if sp.guidance_scale_provided and sp.guidance_scale != _DISTILLED_GUIDANCE_SCALE:
            logger.warning(
                "Lightricks distilled: ignoring guidance_scale=%.2f, forcing %.2f (CFG disabled).",
                sp.guidance_scale,
                _DISTILLED_GUIDANCE_SCALE,
            )
        sp.guidance_scale = _DISTILLED_GUIDANCE_SCALE
        sp.guidance_scale_provided = False
        sp.do_classifier_free_guidance = False

    def forward(
        self,
        req: OmniDiffusionRequest,
        sigmas: list[float] | None = None,
        **kwargs: Any,
    ) -> DiffusionOutput:
        # The engine's internal warmup pass uses a minimal config
        # (``num_inference_steps=1``, ``guidance_scale=0.0``) on purpose to
        # keep cold-start cheap. Skip sanitize for it — forcing 8 steps would
        # multiply warmup cost without benefit, and the warning would
        # otherwise mislead users into thinking a client sent num_steps=1.
        if req.is_dummy_run():
            return super().forward(req, sigmas=sigmas, **kwargs)

        self._sanitize_lightricks_request(req)
        kwargs.pop("num_inference_steps", None)
        kwargs.pop("guidance_scale", None)
        return super().forward(
            req,
            sigmas=sigmas if sigmas is not None else DISTILLED_SIGMA_VALUES,
            num_inference_steps=_DISTILLED_NUM_INFERENCE_STEPS,
            guidance_scale=_DISTILLED_GUIDANCE_SCALE,
            **kwargs,
        )
