# λ-Framework REBOUND Stability Validation

N-body simulations validating that λ = √φ ≈ 1.272 provides dynamical stability comparable to exact mean-motion resonances in planetary systems.

## Paper
Ursachi, A.-S. (2026). *"The λ-Framework: Golden Ratio Spacing in Planetary Systems as Emergent Property of Resonant Mode Coupling"*

## Installation

```bash
pip install rebound numpy matplotlib
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Command Line
```bash
python lambda_stability_simulation.py
```

### Option 2: Google Colab (Recommended for 10⁵ orbits)
Open `lambda_framework_colab.ipynb` in [Google Colab](https://colab.research.google.com) and run all cells. Results are saved to Google Drive automatically.

### Quick Example
```python
import rebound
import numpy as np

PHI = (1 + np.sqrt(5)) / 2
LAMBDA = np.sqrt(PHI)  # ≈ 1.272

sim = rebound.Simulation()
sim.units = ('AU', 'yr', 'Msun')
sim.add(m=1.0)  # Star

# Add 4 planets with λ-spacing
a = 0.1  # Inner planet at 0.1 AU
for i in range(4):
    sim.add(m=3e-6, a=a)  # ~1 Earth mass
    a *= LAMBDA ** (2/3)  # λ-spacing

sim.integrator = "whfast"
sim.integrate(1e5 * sim.particles[1].P)  # 100,000 orbits
print("Simulation complete!")
```

## Results Summary

| Configuration | Period Ratio | Stability (N=50, 10⁵ orbits) |
|---------------|--------------|------------------------------|
| **λ-spacing (√φ)** | **1.2720** | **100%** |
| 4:3 resonance | 1.3333 | 100% |
| 3:2 resonance | 1.5000 | 100% |
| φ-spacing | 1.6180 | 100% |
| 2:1 resonance | 2.0000 | 100% |
| Wide (1.8) | 1.8000 | 100% |
| Random (1.1-1.5) | Variable | 98% |
| Tight (1.15) | 1.1500 | 72% |

### Key Finding
λ-spacing achieves **100% stability** over 10⁵ orbits, matching all tested exact mean-motion resonances. Random spacing shows 98% stability (1 ejection in 50 runs), while tight spacing (1.15) dramatically collapses to **72% stability** (14 ejections in 50 runs).

## Features
- Checkpoint saving after each configuration (resume if interrupted)
- Configurable parameters (N_SIMULATIONS, INTEGRATION_TIME, etc.)
- Automatic plot generation
- Results saved to JSON

## Repository Structure
```
lambda-framework-rebound/
├── lambda_stability_simulation.py  # Main simulation script
├── lambda_framework_colab.ipynb    # Google Colab notebook
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── README.md
└── results/
    └── simulation_results.json     # Output data
```

## Output Files
- `results/simulation_results.json` - Full results data
- `results/results.png` - Stability comparison plot

## Author
**Andrei-Sebastian Ursachi**  
ORCID: [0009-0002-6114-5011](https://orcid.org/0009-0002-6114-5011)

## License
MIT License - See [LICENSE](LICENSE) file for details.

## Citation
If you use this code, please cite:
```bibtex
@article{ursachi2026lambda,
  title={The λ-Framework: Golden Ratio Spacing in Planetary Systems as Emergent Property of Resonant Mode Coupling},
  author={Ursachi, Andrei-Sebastian},
  year={2026}
}
```
