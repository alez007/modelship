# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.6.0] - 2026-07-07

### Added
- pin greedy load client and make bench result-parity gate relative
- extend bench to llama_server/CPU and add --no-preflight for fair A/B
- fool-proof preflight sizing for vllm CPU deploys and llama_server GPU offload
- make vllm loader installable on the cpu extra
- add native streaming support to /v1/responses for vllm and llama_server
- shape /v1/responses natively from ParsedChatOutput for vllm and llama_server
- rewire vLLM streaming chat onto engine_ops
- abort llama_server non-stream requests on client disconnect
- rewire vLLM non-stream chat onto engine_ops
- quarantine vLLM-internal touchpoints behind engine_ops
- implement embeddings, vision, logprobs, and concurrency coupling for llama_server loader (Stage B4)
- extract 3-field DTO and rewire llama_server non-stream projection (Stage B3)
- ship llama-server in Docker images and wire GPU offload (Stage B2)
- add llama_server loader (Stage B1 of the parser-migration roadmap)

### Fixed
- harden bench A/B harness for fair modelship-vs-raw comparisons
- cancel the in-flight next_item before closing work on teardown
- close stream generators and cancel embeddings on disconnect
- guard vllm CPU preflight against an undiscoverable host RAM probe
- count the output layer's weight in llama_server CPU-resident RAM sizing
- don't let thread-alignment preflight starve declared parallel slots
- vllm embed init, transcription/translation request construction, and Responses streaming usage
- wire llama_server streaming chat onto client disconnect and stamp chunk ids
- guard against out-of-bounds top_logprobs index in vLLM logprobs projection
- replace assert with a defensive check for vLLM prompt_token_ids
- resolve mmproj once on the driver instead of again in the actor
- guard against malformed and non-object JSON responses from llama-server
- derive finish_reason for out-of-range list entries instead of hardcoding stop
- use explicit None check for created timestamp fallback in embeddings projection
- suppress interpreter teardown exceptions inside __del__
- secure pending_client_closes against python interpreter teardown
- intercept and handle mid-stream JSON error payloads from llama-server
- intercept and parse inline JSON error payloads on 2xx responses in llama_server loader
- call self.shutdown() on any exception during llama-server startup
- address concurrency, early-crash thread leaks, and closed-loop shutdown issues in llama_server loader
- assert self._proc is not None to resolve pyright type-checking error
- optimize llama_server loader concurrency, timeouts, and protocol alignment
- reject non-positive parallel and close httpx client on shutdown
- harden llama_server loader streaming and subprocess log draining

### Changed
- remove Home Assistant/Wyoming integration doc
- resolve vllm gpu_memory_utilization default lazily instead of auto-flagging it
- remove one-click profiles system
- repoint driver preflight and the vllm actor at the new parser module
- move vLLM parser detection out of driver preflight into the actor
- delete dead raw-text parser engine from openai/parsers/
- delete vLLM OpenAIServingChat monolith usage
- repoint llama_cpp non-stream chat onto build_from_parsed
- add missing llama_server integration coverage
- document the llama_server loader
- reposition Modelship around its agentic + GPU-sharing wedge

## [0.5.8] - 2026-07-01

### Added
- add GPU offload support to the llama_cpp loader

### Fixed
- gate llama_cpp GPU warning correctly and unbreak CI import of cu130 wheel
- match .GGUF extension case-insensitively in vllm loader guard

### Changed
- bump vllm to 0.24.0

## [0.5.7] - 2026-06-30

### Added
- drive tool calling from per-request tool_choice and infer reasoning state

### Fixed
- harden reasoning probe and auto-path reasoning check
- guard get_parser in reasoning probe against startup crash

## [0.5.6] - 2026-06-29

### Added
- enforce required arguments in Gemma tool-call GBNF grammar
- allow optional leading/trailing whitespace in Gemma GBNF tool call grammar
- enforce schema order and uniqueness in Gemma tool call GBNF grammar
- back require_tool_call with LlamaCppConfig, imply constraining
- constrain FunctionGemma/Gemma4 tool calls with a GBNF grammar
- add llama.cpp native prompt cache support

### Fixed
- robustly filter required schema property elements
- guard empty type list in Gemma value emitter
- emit generic recursive value rules for free-form Gemma tool args
- scope disk-cache replica guard to the llama_cpp loader
- key llama.cpp disk cache by deployment name, not model name
- reject llama.cpp disk cache with multiple replicas
- isolate llama.cpp disk cache per-model under MSHIP_CACHE_DIR
- drop stale registry entries for resurrected deployments on reconcile

### Changed
- simplify Gemma GBNF grammar generator and add defensive schema handling
- set _require_tool_call on __new__-built serving chat in trace test
- simplify multi-replica check for disk cache guard

## [0.5.5] - 2026-06-24

### Added
- add generic chat_template_kwargs for text loaders

### Fixed
- reserve "conversation" key on transformers chat_template_kwargs
- reserve "messages" key in chat_template_kwargs
- honor chat_template_kwargs on transformers streaming path
- drop reserved keys from chat_template_kwargs
- skip tool-call grammar on reasoning deployments
- tolerate null assistant content in prompt rendering

### Changed
- merge chat_template_kwargs before building vllm request

## [0.5.4] - 2026-06-24

### Added
- grammar-constrained tool calling (constrain_tool_calls)
- TRACE-log parsed tool calls handed to client

### Fixed
- allow conversational text around grammar-constrained tool calls
- finalize tool calls in ChatOutputStreamer.finalize()
- fall back to default voice for unknown voice names
- require string call_id and name when building tool-name map
- guard tool-name backfill against malformed messages
- backfill tool-message name for strict chat templates

### Changed
- skip tool-call summary build when TRACE disabled

## [0.5.3] - 2026-06-23

### Added
- log chat request/response payloads at TRACE

### Fixed
- forward passthrough env vars and configure logging in gateway replica

### Changed
- cover streaming TRACE response logging

## [0.5.2] - 2026-06-22

### Fixed
- recover disconnect registry from actor death
- TTL-evict disconnect entries instead of clearing on teardown
- stop request watcher on cancellation during initial response
- time streaming generation/request duration after the stream drains
- count GPU models' host RAM against the RAM budget
- avoid double-counting reclaimable cache on cgroup v1
- refuse cleanly when a capability has no catalog models
- size stacks against free RAM with a weighted knapsack selector

### Changed
- keep gateway route tests off a real Ray cluster
- build and push CPU image before GPU
- add integration tests deploying profiles at cpu/gpu stages

## [0.5.1] - 2026-06-20

### Fixed
- make download atomic to avoid corrupt cached files

### Changed
- only run the chart job when the Helm chart changes

## [0.5.0] - 2026-06-19

### Added
- HA control-plane metrics, per-gateway dimension, chart-shipped alerts/dashboard
- reclaim Ray temp disk on restart and drop --redeploy

### Fixed
- warn when backed by a non-durable memory state store
- retry routing reconcile when a deployment handle isn't ready yet
- wire the Model dropdown into per-model panels
- never let metric emission mask a state-store error
- forward MSHIP_METRICS to replicas so --no-metrics is cluster-wide
- export Ray metrics on the declared port

### Changed
- correct gateway-tagged HA metric count (six, not three)
- make metric/state-store proxies transparent via __getattr__

## [0.4.0] - 2026-06-18

### Added
- image.variant selector (gpu default) for cpu/gpu image tags
- redis-backed GCS fault tolerance, retire reassert cron
- durable coordinator state for head-restart self-heal
- URI-selectable state stores (memory/file/redis)
- multi-node ingress — proxy on every node, Service spans all pods
- coordinator-driven watch model for multi-replica routing
- per-model autoscaling + --state-dir flag
- self-heal CronJob reconciling to the effective config
- durable per-gateway effective config for self-heal
- per-group image/runtimeClassName override; empty workerGroups default
- deploy via RayJob on the cluster, not an off-cluster Job
- KubeRay chart — RayCluster + deploy Job, /readyz gating
- k8s/KubeRay readiness — gateway self-heal, ownership registry, /readyz

### Fixed
- reject file:// URIs with a non-empty host
- opt out of vLLM dictConfig so head-restart recovery doesn't crash GPU replicas
- resolve coordinator off the event loop in the watch loop
- drop stale coordinator handle so the watch loop recovers
- address code-review feedback
- declare head dashboard port (8265) for RayJob submission
- RayJob clusterSelector rejects backoffLimit; drop head /readyz probe
- exit instead of blocking/teardown on an external cluster
- compare MSHIP_RAY_DASHBOARD case-insensitively

### Changed
- chart OCI install + image.variant
- validate Helm chart (lint, kubeconform, kind server dry-run)
- publish Helm chart as OCI artifact to GHCR
- stamp Helm chart version/appVersion + image tag from release tag
- head-node HA, state-store URI, reassert cron removal
- per-model autoscaling, gateway HA, state-dir/self-heal
- mship_deploy owns its Ray head; --use-existing connects
- disable Ray dashboard by default to cut host RAM

## [0.3.0] - 2026-06-14

### Added
- one-click model stacks via MSHIP_MODEL_STACK
- add stable_diffusion_cpp CPU image-generation loader
- streaming event protocol for /v1/responses (Phase A2)
- stateless /v1/responses endpoint (Phase A)

### Fixed
- treat cgroup v1 unlimited sentinel as no-limit at the source
- give CPU leftover to a single anchor, not every generate
- exit cleanly when the stack file can't be written
- keep scaled-down CPU allocation within the budget
- check per-model VRAM, not the sum, on multi-GPU boxes
- create parent dir before writing generated stack yaml
- guard _is_moe against non-dict config sections
- fall back to cgroup limit when psutil RAM probe fails
- validate MSHIP_MODEL_STACK before any filesystem op
- allocate whole-integer num_gpus on multi-GPU boxes
- parse all bundled SSE messages per chat chunk
- guard None tool_calls in streaming delta loop
- validate tool-call association ids on input items
- robust error mapping and full usage-detail propagation
- robust status_code handling and list default_factory
- drop logprobs/top_logprobs defaults from completion kwargs
- reject logprobs explicitly instead of silently dropping

### Changed
- use asyncio.get_running_loop() in async paths
- rebind handle to None instead of del on shutdown
- warm up + repeat sweeps for stable A/B numbers
- update _parse_chat_sse tests for multi-message parsing
- document /v1/responses endpoint and streaming
- split protocol.py into a protocol package

## [0.2.0] - 2026-06-06

### Added
- make Ray Serve concurrency caps configurable
- default usecase to image and reject non-image
- /v1/images/edits and /v1/images/variations
- vision / image_url input support
- structured outputs + tools/response_format compat gate
- enforce tools-supersede-response_format precedence
- extend hardware-aware preflight to llama_cpp loader
- qwen3-coder tool parser and custom-loader preflight

### Fixed
- robust memory-unit parsing for docker stats output
- tolerate single/unquoted yaml values when parsing bench.yaml
- guard nvidia-smi/docker stats against pipefail+set -e
- validate gateway concurrency env vars are positive ints
- accept Open WebUI image[] edit uploads and log 422s
- guard getextrema and soften alpha mask edges
- serialize GPU inference with an asyncio lock
- force image decode so truncated uploads error cleanly
- tolerate from_pipe failures for img2img/inpaint
- decode images in executor and honor alpha masks
- release all shared pipelines on teardown
- swap edit/variation default strengths
- close audio upload files after reading
- close image upload files after reading
- use input image alpha as mask on edits
- emit task field on verbose transcription/translation responses
- drop max_completion_tokens after mapping to max_tokens
- unwrap numpy arrays from gguf ReaderField.contents()
- cap llama_cpp n_ctx when GGUF omits context_length
- accept `call <name>` (whitespace) in addition to `call:<name>`
- account for PG bundle CPUs in coordinator reservation
- strip only structural JSON suffix in streamed args

### Changed
- add modelship vs raw vLLM A/B benchmark harness
- record push ownership and no-amend commit policy
- compute _world_size once in build_deployment_options
- declare image[] as an explicit aliased field
- move teardown into shutdown(), delegate from __del__
- decode edit input image only once
- load uploaded edit mask as grayscale
- run image PNG/base64 encoding in the executor
- tighten protocol shapes to OpenAI spec
- cache the json_object LlamaGrammar
- drop tools/response_format precedence validator
- always use ray placement groups for multi-slot deploys

## [0.1.36] - 2026-05-14

### Added
- include PyTorch .bin/.pt weights in footprint estimate
- preflight estimator, pipeline parallelism, and runtime hardening
- add Gemma 4 and FunctionGemma tool/reasoning parsers
- llama3_json tool-call parser
- mistral tool-call parser
- transformers reasoning content
- llama_cpp reasoning content + parser unification
- vllm reasoning content + auto-detect
- llama_cpp tool calling + cross-loader auto-detection
- auto-detect tool-call parser for transformers loader
- incremental streaming for tool-call parsing
- cross-loader tool-calling toolkit + transformers wiring
- add integration testing suite for OpenAI endpoints

### Fixed
- decouple multimodal max_num_batched_tokens from max_model_len
- restore envelope-} strip and harden Gemma parsers
- consider reasoning parsers when resolving skip_special_tokens
- preserve preamble whitespace next to tool_calls
- case-insensitive .gguf suffix check in chat-template reader
- maintain consistent created timestamp in chat streaming
- make tool-call finalization robust to skipped blocks
- integration tests

### Changed
- add bitsandbytes to gpu extra
- narrow exception scope in Gemma args parser
- optimize noise stripping using delta processing and regex
- improve noise-stripping robustness and reasoning support
- update testing target score to 9/10
- remove python sdk example from quick start
- optimize README for user adoption and clarify production readiness
- refresh roadmap and production readiness state
- tighten agent notes wording
- bump transformers to 5.8.0 and llama-cpp-python to 0.3.22
- unify openai parsers under modelship.openai.parsers
- per-model deploy/reconcile in integration tests
- bump vllm to 0.20.1
- simplify tool-parser detection warning and file open
- optimize transformers stream complexity
- remove unused `_content_parts_len` attribute from ToolCallStreamer

## [0.1.35] - 2026-05-01

### Added
- centralize model source resolution on the driver
- add reconciliation logic for deployments based on models.yaml
- add max_num_batched_tokens to VllmEngineConfig
- add flatten_message_content utility and integrate into llama_cpp
- upgrade vLLM to 0.20.0 and harden inference loaders

### Fixed
- resolver returns file paths for GGUF and sets HF_HOME pre-import
- defer ray cluster env var checks to avoid key error in auto mode
- handle direct Response chunks and improve vLLM embedding error conversion
- remove non-existent io_processor argument from OpenAIServingRender
- type llama_cpp stream iterator so pyright accepts run_in_executor
- tag REQUEST_TOTAL by outcome instead of marking every request processed

### Changed
- simplify LlamaCpp plugin by delegating resolution to the driver
- extract deployment components into modelship.deploy
- fix unused import in mship_deploy.py
- simplify mship_deploy.py by extracting logic
- detect capabilities and emit OpenAI-compliant chat responses for transformers loader
- split llama_cpp into per-surface OpenAI serving handlers

## [0.1.34] - 2026-04-27

### Added
- upgrade vllm to 0.19.1

### Fixed
- propagate log levels before ray import and add pip for runtime_env
- reap orphan vLLM workers on actor death and quiet shutdown noise
- use async fatal error reporting and unique deployment keys
- handle fatal deployment initialization errors to prevent infinite retries
- harden orphan reaping and vllm audio response handling
- vllm 0.19.1 response types, tp>1 init, orphan workers

## [0.1.33] - 2026-04-25

### Fixed
- propagate UID/GID ARGs to all Dockerfile stages

## [0.1.32] - 2026-04-25

### Added
- make Ray CPU/GPU allocation auto-detect by default
- implement dynamic wheel-based plugin deployment

### Fixed
- restrict plugin discovery to directories in Makefile
- normalize plugin wheel names to match PEP 427

### Changed
- refresh roadmap and remove stale MSHIP_PLUGINS references
- use Bash arrays for safe argument handling in scripts
- unify GPU/CPU Dockerfiles and update docs for dynamic extras
- dynamically load plugin extras in dev docker stage

## [0.1.31] - 2026-04-24

### Changed
- drop --compile-bytecode from uv sync in Docker builds

## [0.1.30] - 2026-04-23

### Added
- cluster-wide deploy coordinator and retry-pass deploy loop
- /status readiness endpoint with per-model load timings

### Changed
- slim CUDA runtime, MSHIP_SKIP_SYNC fast-path, misc

## [0.1.29] - 2026-04-20

### Added
- make kokoroonnx plugin engine-agnostic
- add whispercpp STT plugin, expand custom plugin system to all usecases

### Changed
- relicense from MIT to Apache-2.0

## [0.1.28] - 2026-04-19

## [0.1.27] - 2026-04-19

### Changed
- consolidated documentation

## [0.1.26] - 2026-04-19

### Fixed
- incorrect syntax on github release

## [0.1.25] - 2026-04-19

### Fixed
- resolve UnboundLocalError and enable arm64 builds

### Changed
- fix cache volume mounts and update llama_cpp example
- clean up env var building and enable arm64 builds

## [0.1.24] - 2026-04-18

### Added
- add llama_cpp loader for cpu-only gguf inference

## [0.1.23] - 2026-04-17

### Added
- migrate cache to /.cache, fix CUDA 12 mismatch, and logging typos
- add --openai-api-port flag and run container as non-root user

### Fixed
- update for ci
- update for ci
- update for ci

### Changed
- decouple OpenAI protocol models from vLLM
- improve quick start with correct docker env vars and CPU-first example
- add public roadmap
- add badges and "Why Modelship?" section to README

## [0.1.22] - 2026-04-15

### Added
- add transformers CPU inference, TRACE logging, and fix audio resampling

### Fixed
- resolve pyright type errors across serving modules

## [0.1.21] - 2026-04-13

### Fixed
- remove dockerfile old config folder setup

## [0.1.20] - 2026-04-13

### Added
- auto-generate changelog from conventional commits during release
- add Prometheus alerting rules, Grafana alerts row, and monitoring docs
- add syslog and OpenTelemetry log export
- additive deploys with --redeploy flag and multi-gateway support

### Fixed
- makefile fix for multi-line changelog

## [0.1.11] - 2025-06-20

### Fixed
- Makefile release process

### Changed
- Consolidated environment variables

## [0.1.10] - 2025-06-19

### Added
- Security policy and vulnerability reporting guidelines

## [0.1.8] - 2025-06-18

### Fixed
- Production Docker build

## [0.1.7] - 2025-06-17

### Changed
- Upgraded plugin system
- Migrated Orpheus to new plugin architecture

## [0.1.6] - 2025-06-16

### Fixed
- GitHub Actions release workflow

## [0.1.5] - 2025-06-15

### Added
- Multi-GPU fractional model support
- Sequential Ray deployment to prevent model load memory spikes
- Kokoro plugin configuration
- Fine-tuned example configs for various GPU sizes

### Fixed
- Tool calling bugfix

## [0.1.4] - 2025-06-14

### Added
- Per-actor cache environment variables
- Dedicated Ray actor for each model
- Cache folder for downloaded models

### Fixed
- Type fix and stability improvement

## [0.1.3] - 2025-06-13

### Fixed
- uv lock file

## [0.1.2] - 2025-06-12

### Added
- Lock file for reproducible builds

## [0.1.1] - 2025-06-11

### Added
- Initial release with GitHub Actions CI/CD
