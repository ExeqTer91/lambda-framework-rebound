# λ-Framework REBOUND Stability Validation

N-body simulations validating that λ = √φ ≈ 1.272 provides dynamical stability comparable to exact mean-motion resonances in planetary systems.

## Paper
Ursachi, A.-S. (2026). "The λ-Framework: Golden Ratio Spacing in Planetary Systems as Emergent Property of Resonant Mode Coupling"

## Results Summary
| Configuration | Period Ratio | Stability (N=50, 10⁵ orbits) |
|---------------|--------------|------------------------------|
| λ-spacing (√φ) | 1.2720 | 100% |
| 4:3 resonance | 1.3333 | 100% |
| 3:2 resonance | 1.5000 | 100% |
| φ-spacing | 1.6180 | 100% |
| 2:1 resonance | 2.0000 | 100% |
| Wide (1.8) | 1.8000 | 100% |
| Random (1.1-1.5) | Variable | 98% |
| Tight (1.15) | 1.1500 | 72% |

## Key Finding
λ-spacing achieves 100% stability over 10⁵ orbits, matching all tested exact mean-motion resonances. Random spacing shows 98% stability (1 ejection in 50 runs), while tight spacing (1.15) dramatically collapses to 72% stability (14 ejections in 50 runs).

## Requirements
```bash
pip install rebound numpy matplotlib
```

## Usage
```bash
python lambda_stability_simulation.py
```

Results are saved to `results/simulation_results.json` and a plot is generated as `results/results.png`.

## Features
- Checkpoint saving after each configuration
- Resume from interruption
- Configurable parameters (N_SIMULATIONS, INTEGRATION_TIME, etc.)

## Author
Andrei-Sebastian Ursachi  
ORCID: [0009-0002-6114-5011](https://orcid.org/0009-0002-6114-5011)

## License
MIT License
