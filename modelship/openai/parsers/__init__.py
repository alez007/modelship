"""Chat-template parser-name detection, shared by driver preflight and the vllm loader.

``tool_calling.py`` / ``reasoning.py`` classify a model's chat template into the
vLLM-native tool-call / reasoning parser name it needs; ``utils.py`` reads chat
templates off disk and renders minimal generation prompts. All three are consumed
directly by module path (``modelship.openai.parsers.{tool_calling,reasoning,utils}``)
rather than re-exported here.
"""
