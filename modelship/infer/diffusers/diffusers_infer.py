import torch

from modelship.infer.base_infer import BaseInfer
from modelship.infer.diffusers.openai.serving_image import OpenAIServingImage
from modelship.infer.infer_config import DiffusersConfig, ModelshipModelConfig, ModelUsecase, RawRequestProxy
from modelship.logging import get_logger
from modelship.openai.protocol import (
    ErrorResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
)

logger = get_logger("infer.diffusers")

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class DiffusersInfer(BaseInfer):
    def __init__(self, model_config: ModelshipModelConfig):
        super().__init__(model_config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        mem_frac = self._get_memory_fraction()
        if torch.cuda.is_available() and mem_frac is not None:
            torch.cuda.set_per_process_memory_fraction(mem_frac)

    def shutdown(self) -> None:
        pass

    def __del__(self):
        try:
            if pipeline := getattr(self, "_pipeline", None):
                del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            from modelship.metrics import RESOURCE_CLEANUP_ERRORS_TOTAL

            RESOURCE_CLEANUP_ERRORS_TOTAL.inc(tags={"model": self.model_config.name, "component": "diffusers_pipeline"})

    async def start(self):
        from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

        config = self.model_config.diffusers_config or DiffusersConfig()
        dtype = _TORCH_DTYPES.get(config.torch_dtype, torch.float16)

        if not self.model_config._resolved_path:
            raise ValueError(
                f"Diffusers deployment '{self.model_config.name}' is missing a resolved model path. "
                f"Check driver logs for resolution errors."
            )

        logger.info(
            "Loading diffusers pipeline: %s (dtype=%s, device=%s)",
            self.model_config._resolved_path,
            config.torch_dtype,
            self.device,
        )
        self._pipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_config._resolved_path,
            torch_dtype=dtype,
        ).to(device=self.device, dtype=dtype)

        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None:
            self._set_max_context_length(getattr(tokenizer, "model_max_length", None))

        self.serving_image: OpenAIServingImage | None = (
            OpenAIServingImage(pipeline=self._pipeline, config=config)
            if self.model_config.usecase is ModelUsecase.image
            else None
        )

    async def warmup(self) -> None:
        if self.serving_image is None:
            return
        logger.info("Warming up diffusers model: %s", self.model_config.name)
        request = ImageGenerationRequest(
            model=self.model_config.name,
            prompt="warmup",
            n=1,
            size="64x64",
        )
        await self.create_image_generation(request, RawRequestProxy(None, {}))
        logger.info("Warmup image generation done for %s", self.model_config.name)

    async def create_image_generation(
        self, request: ImageGenerationRequest, raw_request: RawRequestProxy
    ) -> ErrorResponse | ImageGenerationResponse:
        if self.serving_image is None:
            return await super().create_image_generation(request, raw_request)
        return await self.serving_image.create_image_generation(request, raw_request)
