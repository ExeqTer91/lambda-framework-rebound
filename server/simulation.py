"""
λ-Framework Validation with REBOUND N-Body Simulator
=====================================================
Tests whether planetary systems with λ = √φ ≈ 1.272 spacing
are more stable than random or resonant configurations.

Author: Andrei Ursachi
ORCID: 0009-0002-6114-5011

Features:
- Intermediate saving after each configuration
- Resume from checkpoint if interrupted
- Configurable parameters
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime

try:
    import rebound
except ImportError:
    print("Rebound not found. Please install it with: pip install rebound")
    sys.exit(1)

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
LAMBDA = np.sqrt(PHI)        # √φ ≈ 1.272

#==============================================================================
# SIMULATION PARAMETERS - ADJUST THESE FOR YOUR NEEDS
#==============================================================================

N_SIMULATIONS = 50          # Number of systems per configuration
INTEGRATION_TIME = 1e5      # Orbital periods of inner planet
PLANET_MASS = 3e-6          # ~1 Earth mass around Sun
N_PLANETS = 4               # Number of planets per system
INNER_SEMI_MAJOR = 0.1      # AU (inner planet)

# Output paths
CHECKPOINT_FILE = "simulation_checkpoint.json"
RESULTS_FILE = "simulation_results.json"
PLOT_FILE = "results.png"

# Period ratio configurations to test
CONFIGURATIONS = {
    'λ-spacing (√φ)': LAMBDA,
    '4:3 resonance': 4/3,
    '3:2 resonance': 3/2,
    'φ-spacing': PHI,
    '2:1 resonance': 2.0,
    'Random (1.1-1.5)': None,
    'Tight (1.15)': 1.15,
    'Wide (1.8)': 1.8,
}

#==============================================================================
# CHECKPOINT FUNCTIONS
#==============================================================================

def load_checkpoint():
    """Load checkpoint if exists, return completed configs and results."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                print(f"\n*** CHECKPOINT FOUND - Resuming from previous run ***")
                print(f"    Completed: {len(data.get('results', {}))} configurations")
                print(f"    Last saved: {data.get('timestamp', 'unknown')}")
                return data.get('results', {}), data.get('completed', [])
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return {}, []

def save_checkpoint(results, completed_configs):
    """Save current progress to checkpoint file."""
    data = {
        'results': results,
        'completed': completed_configs,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'N_SIMULATIONS': N_SIMULATIONS,
            'INTEGRATION_TIME': INTEGRATION_TIME,
            'N_PLANETS': N_PLANETS,
            'PLANET_MASS': PLANET_MASS,
        }
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"    [Checkpoint saved]")

def save_final_results(results):
    """Save final results to JSON file."""
    data = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'N_SIMULATIONS': N_SIMULATIONS,
            'INTEGRATION_TIME': INTEGRATION_TIME,
            'N_PLANETS': N_PLANETS,
            'PLANET_MASS': PLANET_MASS,
        },
        'constants': {
            'PHI': PHI,
            'LAMBDA': LAMBDA,
        }
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nFinal results saved to {RESULTS_FILE}")

#==============================================================================
# SIMULATION FUNCTIONS
#==============================================================================

def create_system(period_ratio, n_planets=N_PLANETS, planet_mass=PLANET_MASS, 
                  inner_a=INNER_SEMI_MAJOR, random_ratio=False):
    """Create a planetary system with given period ratio spacing."""
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    
    sim.add(m=1.0)
    
    a = inner_a
    for i in range(n_planets):
        ecc = np.random.uniform(0.001, 0.02)
        inc = np.random.uniform(0, 0.02)
        omega = np.random.uniform(0, 2*np.pi)
        Omega = np.random.uniform(0, 2*np.pi)
        f = np.random.uniform(0, 2*np.pi)
        
        sim.add(m=planet_mass, a=a, e=ecc, inc=inc, omega=omega, Omega=Omega, f=f)
        
        if random_ratio:
            pr = np.random.uniform(1.1, 1.5)
        else:
            pr = period_ratio
        
        a *= pr ** (2/3)
    
    sim.move_to_com()
    return sim

def check_stability(sim, integration_time=INTEGRATION_TIME):
    """Integrate system and check if it remains stable."""
    sim.integrator = "whfast"
    sim.dt = sim.particles[1].P / 20
    
    try:
        n_checks = 100
        check_interval = integration_time / n_checks
        
        for i in range(n_checks):
            sim.integrate(sim.t + check_interval)
            
            for p in sim.particles[1:]:
                d = np.sqrt(p.x**2 + p.y**2 + p.z**2)
                if d > 100 or d < 0.01:
                    return False, sim.t, "Ejection/Collision"
                
        return True, integration_time, "Stable"
        
    except rebound.Collision:
        return False, sim.t, "Collision"
    except Exception as e:
        return False, sim.t, f"Error: {str(e)}"

def generate_plot(results, output_path=PLOT_FILE):
    """Generate and save the results plot."""
    plt.figure(figsize=(12, 6))
    
    names = list(results.keys())
    values = [results[name]['stability_rate'] for name in names]
    
    bars = plt.bar(names, values, color='skyblue')
    
    if 'λ-spacing (√φ)' in results:
        idx = names.index('λ-spacing (√φ)')
        bars[idx].set_color('gold')
    
    # Color low stability configs differently
    for i, (name, val) in enumerate(zip(names, values)):
        if val < 100:
            bars[i].set_color('salmon')
        
    plt.title(f'Planetary System Stability\n(N={N_PLANETS} planets, {N_SIMULATIONS} runs, T={INTEGRATION_TIME:.0e} orbits)')
    plt.ylabel('Stability Rate (%)')
    plt.xlabel('Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

#==============================================================================
# MAIN EXECUTION
#==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("λ-FRAMEWORK REBOUND STABILITY VALIDATION")
    print("="*70)
    print(f"\nFundamental Constants:")
    print(f"  φ (golden ratio) = {PHI:.10f}")
    print(f"  λ = √φ           = {LAMBDA:.10f}")
    print(f"\nSimulation Parameters:")
    print(f"  Runs per config:  {N_SIMULATIONS}")
    print(f"  Integration time: {INTEGRATION_TIME:.0e} orbits")
    print(f"  Planets:          {N_PLANETS}")
    print(f"  Planet mass:      {PLANET_MASS:.0e} M☉")
    
    # Load checkpoint if exists
    results, completed_configs = load_checkpoint()
    
    print(f"\nStarting simulations...")
    print("-"*70)
    
    for name, ratio in CONFIGURATIONS.items():
        # Skip already completed configs
        if name in completed_configs:
            print(f"\n{name}: Already completed (Stability: {results[name]['stability_rate']:.1f}%)")
            continue
            
        print(f"\nTesting {name}...")
        stable_count = 0
        total_time = 0
        
        for i in range(N_SIMULATIONS):
            is_random = (ratio is None)
            sim = create_system(ratio if not is_random else 1.0, random_ratio=is_random)
            
            stable, time, state = check_stability(sim)
            
            if stable:
                stable_count += 1
            total_time += time
            
            # Progress indicator
            if (i+1) % 10 == 0:
                print(f"  Progress: {i+1}/{N_SIMULATIONS} ({stable_count} stable so far)")
        
        stability_rate = stable_count / N_SIMULATIONS * 100
        avg_survival = total_time / N_SIMULATIONS
        
        results[name] = {
            'stability_rate': stability_rate,
            'avg_survival': avg_survival,
            'stable_count': stable_count,
            'total_runs': N_SIMULATIONS
        }
        completed_configs.append(name)
        
        print(f"  RESULT: Stability Rate = {stability_rate:.1f}%")
        print(f"          Avg Survival   = {avg_survival:.2e} yrs")
        
        # Save checkpoint after each configuration
        save_checkpoint(results, completed_configs)

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    
    # Print summary table
    print("\n{:<25} {:>15} {:>15}".format("Configuration", "Stability", "Avg Survival"))
    print("-"*55)
    for name, data in results.items():
        print("{:<25} {:>14.1f}% {:>14.2e}".format(
            name, data['stability_rate'], data['avg_survival']))
    
    # Save final results
    save_final_results(results)
    
    # Generate plot
    print("\nGenerating results plot...")
    
    # Try to save to client/public first, fallback to current dir
    try:
        public_path = os.path.join(os.getcwd(), 'client', 'public', 'results.png')
        generate_plot(results, public_path)
    except:
        generate_plot(results, PLOT_FILE)
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint file cleaned up.")
    
    print("\nDone!")
