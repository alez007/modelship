"""One-click model "profiles" (a.k.a. model stacks).

A user picks a capability set via `MSHIP_MODEL_STACK` (e.g. `chat`, `assistant`,
`studio`, `everything`) and the generator writes a concrete, editable
`config/models.yaml` sized to the detected hardware — removing the need to know
models, allocate resources, or hand-write config. See `budget.py` for how
hardware budgets are read.
"""

from modelship.deploy.profiles.budget import DeployBudget, read_deploy_budget

__all__ = ["DeployBudget", "read_deploy_budget"]
