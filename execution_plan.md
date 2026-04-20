### Deliverables
- `docs/temporal_extension_v2_plan.md`
### Gate to Pass
- Baseline v1 is stable and externally evaluated before 4D model development starts.
---
## Suggested Timeline
- **Day 1-2:** Stage A (pairing policy + Slicer verification)
- **Day 3:** Stage B (manifest) + Stage C (splits)
- **Day 4:** Stage D (baseline protocol freeze)
- **Day 5-7:** Stage E (first training run + results summary)
- **Afterward:** Stage F planning + incremental temporal prototype
---
## Risk Register (Watch-Outs)
- Label anomalies (e.g., value 5 or missing class 2) -> decide remap/drop policy explicitly.
- 4D handling inconsistency -> define one frame extraction rule and apply globally.
- Crop/full confusion -> enforce one baseline input rule first.
- Data leakage in CV -> strict patient and site grouping is required.
- External data used for tuning -> preserve strict final holdout.
---
## Immediate Focus
Current execution priority:
1. Stage A (pairing policy + visual QA)
2. Stage B (dataset manifest creation)
## Status Note
No new code is required for this planning milestone. The next implementation action can begin with Stage B manifest automation.
