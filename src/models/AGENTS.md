# AGENTS.md - src/models/

**Core Bi-Mamba architecture.** All model code lives here.

## Structure
```
src/models/
├── bimamba.py                  # Manual SSM (primary)
└── bimamba_with_mamba_ssm.py   # Wrapper using mamba-ssm package
```

## Key Classes

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `BiMambaBlock` | class | bimamba.py:15 | Selective SSM core — in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj |
| `BiMambaEncoder` | class | bimamba.py | Forward + backward BiMambaBlock stacks, fusion |
| `BiMambaForPropertyPrediction` | class | bimamba.py | Full model: embedding → encoder → pooling → head |
| `create_bimamba_model` | factory | bimamba.py | `d_model`, `n_layers`, `fusion` (concat/add/gate), `pool_type` (mean/max/cls) |

## Fusion Modes
- `gate` (default): `sigmoid(W_fwd) * fwd + (1-sigmoid(W_fwd)) * bwd`
- `concat`: `W * concat(fwd, bwd)`
- `add`: `fwd + bwd`

## Pooling Types
- `mean` (default): global average pooling
- `max`: global max pooling
- `cls`: [CLS] token pooling

## Conventions (THIS MODULE)
- SSM parameters: `d_state=16`, `d_conv=4`, `expand=2`
- dt_rank: `auto` = `ceil(d_model / 16)`
- Activation: `SiLU` (nn.SiLU)
- A_log: `nn.Parameter(log(A))` where A =.arange(1, d_state+1)

## Anti-Patterns (THIS MODULE)
- **NEVER** use `as any` or `@ts-ignore` — type safety is required
- **NEVER** use bare dicts — use dataclasses