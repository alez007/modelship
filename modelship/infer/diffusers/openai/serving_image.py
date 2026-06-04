import asyncio
import base64
import io
import time

from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)
from PIL import Image

from modelship.infer.infer_config import DiffusersConfig, RawRequestProxy
from modelship.logging import TRACE, get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    ImageVariationRequest,
    create_error_response,
)
from modelship.utils import base_request_id

logger = get_logger("infer.diffusers.image")

# Default img2img strength when the request omits it. Edits stay closer to the
# source (lower strength); variations diverge more (higher strength).
_DEFAULT_EDIT_STRENGTH = 0.8
_DEFAULT_VARIATION_STRENGTH = 0.7


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
        inpaint = mask_data is not None
        logger.info(
            "image edit request %s: prompt=%r, n=%d, size=%s, mask=%s",
            request_id,
            request.prompt,
            request.n,
            request.size,
            inpaint,
        )

        try:
            width, height = _parse_size(request.size)
        except ValueError as e:
            return create_error_response(str(e))

        try:
            image = _load_image(image_data, width, height)
        except ValueError as e:
            return create_error_response(str(e))

        steps = self.config.num_inference_steps
        guidance = self.config.guidance_scale
        strength = request.strength if request.strength is not None else _DEFAULT_EDIT_STRENGTH

        if inpaint:
            if self.inpaint_pipeline is None:
                return create_error_response("model does not support image editing")
            try:
                mask = _load_image(mask_data, width, height)  # type: ignore[arg-type]
            except ValueError as e:
                return create_error_response(str(e))
            pipeline = self.inpaint_pipeline
            kwargs = {"mask_image": mask}
        else:
            if self.img2img_pipeline is None:
                return create_error_response("model does not support image editing")
            pipeline = self.img2img_pipeline
            kwargs = {}

        def _run() -> ImageGenerationResponse:
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
        response = await loop.run_in_executor(None, _run)
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

        try:
            image = _load_image(image_data, width, height)
        except ValueError as e:
            return create_error_response(str(e))

        steps = self.config.num_inference_steps
        guidance = self.config.guidance_scale
        strength = request.strength if request.strength is not None else _DEFAULT_VARIATION_STRENGTH

        def _run() -> ImageGenerationResponse:
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
        response = await loop.run_in_executor(None, _run)
        logger.log(TRACE, "image variation response %s: num_images=%d", request_id, len(response.data))
        return response


def _build_response(images: list, revised_prompt: str | None) -> ImageGenerationResponse:
    data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data.append(ImageObject(b64_json=b64, revised_prompt=revised_prompt))
    return ImageGenerationResponse(created=int(time.time()), data=data)


def _load_image(data: bytes, width: int, height: int) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not decode input image: {e}") from e
    return img.resize((width, height))


def _parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format '{size}', expected WxH (e.g. '512x512')")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be positive, got {w}x{h}")
    if w % 8 != 0 or h % 8 != 0:
        raise ValueError(f"Width and height must be multiples of 8, got {w}x{h}")
    return w, h
