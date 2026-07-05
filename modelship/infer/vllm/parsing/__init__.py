"""Chat-template parser-name detection for the vllm loader.

``detect.py`` classifies a model's chat template into the vLLM-native tool-call /
reasoning parser name it needs. Consumed directly by module path
(``modelship.infer.vllm.parsing.detect``) rather than re-exported here.
"""
