import asyncio

from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)

from modelship.infer.image_serving_common import (
    alpha_mask as _alpha_mask,
)
from modelship.infer.image_serving_common import (
    build_response as _build_response,
)
from modelship.infer.image_serving_common import (
    decode_image as _decode_image,
)
from modelship.infer.image_serving_common import (
    load_image as _load_image,
)
from modelship.infer.image_serving_common import (
    load_mask as _load_mask,
)
from modelship.infer.image_serving_common import (
    parse_size as _parse_size,
)
from modelship.infer.infer_config import DiffusersConfig, RawRequestProxy
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageVariationRequest,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.diffusers.image")

# Default img2img strength when the request omits it. Edits stay closer to the
# source (lower strength) since the prompt guides the change; variations have
# no prompt, so they need a higher strength to diverge from the input.
_DEFAULT_EDIT_STRENGTH = 0.7
_DEFAULT_VARIATION_STRENGTH = 0.8


class OpenAIServingImage:
    request_id_prefix = "img"

    def __init__(
        self,
        pipeline: AutoPipelineForText2Image,
        config: DiffusersConfig,
        img2img_pipeline: AutoPipelineForImage2Image | None = None,
        inpaint_pipeline: AutoPipelineForInpainting | None = None,
    ):
        self.pipeline = pipeline
        self.config = config
        self.img2img_pipeline = img2img_pipeline
        self.inpaint_pipeline = inpaint_pipeline
        # Diffusers pipelines are not thread-safe and the three pipelines share
        # the same UNet/VAE/text-encoder, so concurrent forward passes (across
        # executor threads within a replica) would race. Serialize all GPU
        # inference through this lock; it gates only the executor hop, so the
        # event loop stays free while a pass runs.
        self._gpu_lock = asyncio.Lock()

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: RawRequestProxy
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info(
            "image generation request %s: prompt=%r, n=%d, size=%s", request_id, request.prompt, request.n, request.size
        )
        logger.log(
            TRACE,
            "image request %s: prompt=%r, n=%d, size=%s, steps=%s, guidance=%s",
            request_id,
            request.prompt,
            request.n,
            request.size,
            self.config.num_inference_steps,
            self.config.guidance_scale,
        )

        try:
            width, height = _parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        steps = self.config.num_inference_steps
        guidance = self.config.guidance_scale

        def _run() -> ImageGenerationResponse:
            images = self.pipeline(  # type: ignore[reportCallIssue]
                prompt=request.prompt,
                num_images_per_prompt=request.n,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images
            return _build_response(images, revised_prompt=request.prompt)

        # Inference and PNG/base64 encoding are both CPU-bound; run the whole
        # chain in the executor so the event loop stays responsive.
        loop = asyncio.get_event_loop()
        async with self._gpu_lock:
            response = await loop.run_in_executor(None, _run)
        logger.log(TRACE, "image response %s: num_images=%d", request_id, len(response.data))
        return response

    async def create_image_edit(
        self,
        image_data: bytes,
        mask_data: bytes | None,
        request: ImageEditRequest,
        raw_request: RawRequestProxy,
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"

        try:
            width, height = _parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        steps = self.config.num_inference_steps
        guidance = self.config.guidance_scale
        strength = request.strength if request.strength is not None else _DEFAULT_EDIT_STRENGTH

        # All image decode / resize / mask work is CPU-bound; run it in the
        # executor alongside inference and encoding so the event loop is never
        # blocked. _run returns the final response (or an error response).
        def _run() -> ImageGenerationResponse | ErrorResponse:
            try:
                # Decode the input once; derive both the RGB image and (if
                # needed) the alpha mask from it.
                src = _decode_image(image_data)
                image = src.convert("RGB").resize((width, height))
                # Determine the inpaint mask. An explicit `mask` upload wins;
                # otherwise, per the OpenAI edits spec, the input image's own
                # transparency is used as the mask (transparent areas mark the
                # region to edit). With neither, this is a plain img2img edit.
                if mask_data is not None:
                    mask = _load_mask(mask_data, width, height)
                else:
                    mask = _alpha_mask(src, width, height)
            except ValueError as e:
                return create_error_response(str(e))

            inpaint = mask is not None
            logger.info(
                "image edit request %s: prompt=%r, n=%d, size=%s, inpaint=%s",
                request_id,
                request.prompt,
                request.n,
                request.size,
                inpaint,
            )

            if inpaint:
                if self.inpaint_pipeline is None:
                    return create_error_response("model does not support image editing")
                pipeline = self.inpaint_pipeline
                kwargs = {"mask_image": mask}
            else:
                if self.img2img_pipeline is None:
                    return create_error_response("model does not support image editing")
                pipeline = self.img2img_pipeline
                kwargs = {}

            images = pipeline(  # type: ignore[reportCallIssue]
                prompt=request.prompt,
                image=image,
                num_images_per_prompt=request.n,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
                **kwargs,
            ).images
            return _build_response(images, revised_prompt=request.prompt)

        loop = asyncio.get_event_loop()
        async with self._gpu_lock:
            response = await loop.run_in_executor(None, _run)
        if isinstance(response, ImageGenerationResponse):
            logger.log(TRACE, "image edit response %s: num_images=%d", request_id, len(response.data))
        return response

    async def create_image_variation(
        self, image_data: bytes, request: ImageVariationRequest, raw_request: RawRequestProxy
    ) -> ImageGenerationResponse | ErrorResponse:
        request_id = f"{self.request_id_prefix}-{base_request_id(raw_request)}"
        logger.info("image variation request %s: n=%d, size=%s", request_id, request.n, request.size)

        if self.img2img_pipeline is None:
            return create_error_response("model does not support image variations")

        try:
            width, height = _parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        steps = self.config.num_inference_steps
        guidance = self.config.guidance_scale
        strength = request.strength if request.strength is not None else _DEFAULT_VARIATION_STRENGTH

        # Decode / resize / encode are CPU-bound; keep them off the event loop.
        def _run() -> ImageGenerationResponse | ErrorResponse:
            try:
                image = _load_image(image_data, width, height)
            except ValueError as e:
                return create_error_response(str(e))
            images = self.img2img_pipeline(  # type: ignore[reportCallIssue]
                prompt="",
                image=image,
                num_images_per_prompt=request.n,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
            ).images
            return _build_response(images, revised_prompt=None)

        loop = asyncio.get_event_loop()
        async with self._gpu_lock:
            response = await loop.run_in_executor(None, _run)
        if isinstance(response, ImageGenerationResponse):
            logger.log(TRACE, "image variation response %s: num_images=%d", request_id, len(response.data))
        return response
