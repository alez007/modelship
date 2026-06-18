VERSION := $(shell grep -m1 '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
MAJOR   := $(shell echo $(VERSION) | cut -d. -f1)
MINOR   := $(shell echo $(VERSION) | cut -d. -f2)
PATCH   := $(shell echo $(VERSION) | cut -d. -f3)

MSHIP_PLUGIN_WHEEL_DIR ?= .build/plugin-wheels
PLUGIN_STAMP_DIR := .build/plugin-stamps
PLUGINS          := $(notdir $(patsubst %/,%,$(wildcard plugins/*/)))

.PHONY: test lint lint-fix release-patch release-minor release-major _release plugin-wheels plugin-wheels-clean

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

# Per-plugin source-tracked wheel build. Each plugin gets a stamp file whose
# prereqs are every file under its source tree; changing a source invalidates
# only that plugin's stamp, so subsequent `make plugin-wheels` runs are
# incremental. Stale wheels for the same plugin are purged before each build
# to keep a single wheel per plugin in the output dir (simplifies lookup from
# mship_deploy.py, which globs for <name>-*.whl).
plugin-wheels: $(foreach p,$(PLUGINS),$(PLUGIN_STAMP_DIR)/$(p).stamp)

PLUGIN_SOURCES = $(shell find plugins/$(1) -type f -not -path '*/.*' -not -path '*/__pycache__/*' 2>/dev/null)

.SECONDEXPANSION:
$(PLUGIN_STAMP_DIR)/%.stamp: $$(call PLUGIN_SOURCES,%)
	@mkdir -p $(MSHIP_PLUGIN_WHEEL_DIR) $(PLUGIN_STAMP_DIR)
	@rm -f $(MSHIP_PLUGIN_WHEEL_DIR)/$(subst -,_,$*)-*.whl
	uv build --package $* --wheel --out-dir $(MSHIP_PLUGIN_WHEEL_DIR)
	@touch $@

plugin-wheels-clean:
	rm -rf $(MSHIP_PLUGIN_WHEEL_DIR) $(PLUGIN_STAMP_DIR)

release-patch:
	$(eval NEW_VERSION := $(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

release-minor:
	$(eval NEW_VERSION := $(MAJOR).$(shell echo $$(($(MINOR)+1))).0)
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

release-major:
	$(eval NEW_VERSION := $(shell echo $$(($(MAJOR)+1))).0.0)
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

_release:
	@if [ "$$(git branch --show-current)" != "main" ]; then echo "Error: releases must be made from the main branch" >&2; exit 1; fi
	@if [ -n "$$(git status --porcelain)" ]; then echo "Error: working tree is dirty, commit or stash changes first" >&2; exit 1; fi
	@git pull --rebase origin main
	@echo "Bumping version: $(VERSION) -> $(NEW_VERSION)"
	@sed -i '0,/^version = ".*"/{s/^version = ".*"/version = "$(NEW_VERSION)"/}' pyproject.toml
	@uv lock
	@# --- lockstep the Helm chart with the app version (single source of truth: the tag) ---
	@# chart version == appVersion == image tag == app version, so a checked-out tag
	@# renders an installable chart and `helm install --version X.Y.Z` pairs image X.Y.Z.
	@sed -i 's/^version: .*/version: $(NEW_VERSION)/' helm/modelship/Chart.yaml
	@sed -i 's/^appVersion: .*/appVersion: "$(NEW_VERSION)"/' helm/modelship/Chart.yaml
	@sed -i '0,/^  tag: ".*"/{s/^  tag: ".*"/  tag: "$(NEW_VERSION)"/}' helm/modelship/values.yaml
	@# --- auto-update CHANGELOG.md ---
	@PREV_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo ""); \
	if [ -n "$$PREV_TAG" ]; then \
		RANGE="$$PREV_TAG..HEAD"; \
	else \
		RANGE=""; \
	fi; \
	ADDED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^feat(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	FIXED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^fix(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	CHANGED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^(refactor|perf|docs|chore|build|ci|style|test)(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	TMPF=$$(mktemp); \
	echo "" >> "$$TMPF"; \
	echo "## [$(NEW_VERSION)] - $$(date +%Y-%m-%d)" >> "$$TMPF"; \
	if [ -n "$$ADDED" ]; then echo "" >> "$$TMPF"; echo "### Added" >> "$$TMPF"; echo "$$ADDED" >> "$$TMPF"; fi; \
	if [ -n "$$FIXED" ]; then echo "" >> "$$TMPF"; echo "### Fixed" >> "$$TMPF"; echo "$$FIXED" >> "$$TMPF"; fi; \
	if [ -n "$$CHANGED" ]; then echo "" >> "$$TMPF"; echo "### Changed" >> "$$TMPF"; echo "$$CHANGED" >> "$$TMPF"; fi; \
	sed -i "/^The format is based on/r $$TMPF" CHANGELOG.md; \
	rm -f "$$TMPF"
	@git add pyproject.toml uv.lock CHANGELOG.md helm/modelship/Chart.yaml helm/modelship/values.yaml
	@git commit -m "release: v$(NEW_VERSION)"
	@git tag -a "v$(NEW_VERSION)" -m "Release v$(NEW_VERSION)"
	@git push origin main --follow-tags
	@echo "Done. GitHub Actions will build and publish the release."
