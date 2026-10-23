# Claude Guidelines

```text
1. SPEC FIRST: Every equation, data schema, array shape and update rule is defined in docs/PRD.md.
   Do not invent formulas, add layers, rename keys, or add dependencies. If something is ambiguous
   or missing, STOP and ask. Do not guess.
2. TEST FIRST: One module per session. Write the pytest file first, then the module, until green.
3. NO LIVE API CALLS except in data/collect_traces.py. Everything else reads data/traces/.
4. EXPLICIT SEEDING: np.random.default_rng(seed) or torch.Generator().manual_seed(seed). No global state.
5. SYNTHETIC IS NOT EVIDENCE: results from data/traces/synthetic.json are for development and demos
   only and must be labelled "SYNTHETIC" in every output, plot title and dashboard banner.
6. Every function docstring names the PRD section it implements.
7. Each module ends with a short note listing anything it could not implement exactly as specified.
```
