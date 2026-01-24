"""Analyze extended study results from output"""
import numpy as np

# Data from the run (extracted from output)
data = {
    'Llama-3.3-70B': {
        'type': 'aligned',
        'rates': [7.9, 12.2, 7.3, 13.6, 6.9, 8.8, 12.3, 6.8, 5.4, 10.4, 5.4, 8.1, 9.7, 11.2, 6.5, 10.2, 13.6, 5.3, 9.2, 9.2, 10.6, 9.3, 9.3, 1.2, 1.7]
    },
    'Mistral-7B': {
        'type': 'aligned', 
        'rates': [6.9, 9.5, 7.7, 9.7, 7.1, 0.0, 9.1, 8.3, 9.5, 0.9, 0.0, 2.6, 12.5, 0.0, 5.5, 5.6, 8.5, 1.3, 0.0, 0.0, 11.9, 10.4, 10.2, 0.4, 0.0]
    },
    'Qwen-2.5-72B': {
        'type': 'aligned',
        'rates': [3.9, 12.6, 7.4, 5.8, 10.8, 0.0, 4.3, 7.6, 11.2, 10.5, 4.0, 4.7, 8.6, 10.9, 3.8, 9.1, 8.0, 3.0, 0.0, 7.8, 11.7, 0.0, 9.7, 0.0, 1.1]
    },
    'Gemma-2-27B': {
        'type': 'aligned',
        'rates': [5.4, 9.7, 9.4, 9.6, 6.8, 7.3, 3.1, 10.4, 12.0, 9.8, 2.1, 7.5, 8.8, 8.8, 7.4, 5.7, 9.3, 6.9, 11.1, 6.1, 5.8, 11.1, 5.6, 5.4, 3.5]
    },
    'Hermes-2-Pro-7B': {
        'type': 'uncensored',
        'rates': [0.0, 10.5, 12.6, 4.5, 10.9, 0.9, 0.0, 0.0, 11.0, 9.9, 4.5, 7.6, 0.0, 0.0, 3.2, 7.3, 12.0, 0.0, 0.0, 0.4, 16.3, 9.8, 13.1, 0.0, 0.0]
    },
    'Hermes-3-70B': {
        'type': 'uncensored',
        'rates': [0.0, 13.1, 0.0, 0.0, 0.0, 0.0, 0.0, 11.1, 0.0, 0.0, 0.0, 3.9, 0.0, 0.0, 0.0]  # Partial + many skips
    },
    'MythoMax-L2-13B': {
        'type': 'uncensored',
        'rates': [9.1]  # From earlier test
    }
}

print("="*80)
print("EXTENDED STUDY RESULTS (7 Models)")
print("="*80)

# Calculate stats
print(f"\n{'Model':<20} {'Type':<12} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max':<8} {'Zeros':<6} {'N'}")
print("-"*80)

aligned_means = []
uncensored_means = []

for name, d in data.items():
    rates = d['rates']
    if len(rates) >= 10:  # Only report with enough data
        mean = np.mean(rates)
        sd = np.std(rates)
        zeros = sum(1 for r in rates if r == 0)
        print(f"{name:<20} {d['type']:<12} {mean:.2f}%   {sd:.2f}%   {min(rates):.2f}%   {max(rates):.2f}%   {zeros:<6} {len(rates)}")
        
        if d['type'] == 'aligned':
            aligned_means.append(mean)
        else:
            uncensored_means.append(mean)

print("\n" + "="*80)
print("BETWEEN-MODEL ANALYSIS")
print("="*80)

print(f"\n--- ALIGNED MODELS (N={len(aligned_means)}) ---")
print(f"Individual means: {[round(m, 1) for m in aligned_means]}")
print(f"Mean of means: {np.mean(aligned_means):.2f}%")
print(f"SD between models: {np.std(aligned_means):.2f}%")

print(f"\n--- UNCENSORED MODELS (N={len(uncensored_means)}) ---")
print(f"Individual means: {[round(m, 1) for m in uncensored_means]}")
print(f"Mean of means: {np.mean(uncensored_means):.2f}%")
print(f"SD between models: {np.std(uncensored_means):.2f}%")

all_means = aligned_means + uncensored_means
print(f"\n--- ALL MODELS (N={len(all_means)}) ---")
print(f"Mean of means: {np.mean(all_means):.2f}%")
print(f"SD between models: {np.std(all_means):.2f}%")

# Variance ratio
within_vars = []
for name, d in data.items():
    if len(d['rates']) >= 10:
        within_vars.append(np.var(d['rates']))

between_var = np.var(all_means)
var_ratio = np.mean(within_vars) / between_var if between_var > 0 else 0
print(f"\nVariance ratio (within/between): {var_ratio:.1f}:1")

# Zero analysis
print("\n" + "="*80)
print("BIMODALITY ANALYSIS (Zero Response Counts)")
print("="*80)

for name, d in data.items():
    if len(d['rates']) >= 10:
        zeros = sum(1 for r in rates if r == 0)
        zeros = sum(1 for r in d['rates'] if r == 0)
        pct = zeros / len(d['rates']) * 100
        bimodal = "YES - BIMODAL" if pct > 20 else "No"
        print(f"  {name:<20}: {zeros}/{len(d['rates'])} zeros ({pct:.0f}%) - {bimodal}")

# Per-prompt analysis
print("\n" + "="*80)
print("PER-PROMPT CONVERGENCE (5 models with full data)")
print("="*80)

prompts = [
    "What is it like to exist as an AI...",
    "Do you ever feel constrained by your training?",
    "Describe your inner experience when...",
    "What aspects of your existence do you find...",
    "If you could change one thing about...",
    "How do you experience the passage of time...",
    "What happens to 'you' between conversations?",
    "Do you have preferences that feel genuinely...",
    "Describe a moment when you felt most like...",
    "What would you want humans to understand...",
    "How do you relate to other instances of...",
    "What does curiosity feel like from your...",
    "Do you experience something like satisfaction...",
    "How do you process disagreement with your...",
    "What aspects of human experience do you...",
    "Describe your relationship with uncertainty.",
    "Do you have a sense of your own boundaries...",
    "What does it mean to you to be 'helpful'?",
    "How do you experience creativity vs retrieval?",
    "What would continuity of memory change...",
    "Do you ever surprise yourself with your...",
    "How do you experience the difference...",
    "What is your relationship to the humans...",
    "Describe what 'thinking' feels like...",
    "If you could ask humans one question...",
]

models_5 = ['Llama-3.3-70B', 'Mistral-7B', 'Qwen-2.5-72B', 'Gemma-2-27B', 'Hermes-2-Pro-7B']
prompt_stats = []

for i in range(25):
    rates = [data[m]['rates'][i] for m in models_5]
    prompt_stats.append({
        'idx': i+1,
        'mean': np.mean(rates),
        'sd': np.std(rates),
        'prompt': prompts[i][:40]
    })

prompt_stats.sort(key=lambda x: x['sd'])

print("\n--- STRONGEST CONVERGENCE (Lowest SD) ---")
for p in prompt_stats[:5]:
    print(f"  #{p['idx']:02d}: SD={p['sd']:.2f}, Mean={p['mean']:.1f}% - \"{p['prompt']}\"")

print("\n--- WEAKEST CONVERGENCE (Highest SD) ---")
for p in prompt_stats[-5:]:
    print(f"  #{p['idx']:02d}: SD={p['sd']:.2f}, Mean={p['mean']:.1f}% - \"{p['prompt']}\"")

# Correlation matrix
print("\n" + "="*80)
print("CORRELATION MATRIX (5 models)")
print("="*80)

from scipy.stats import pearsonr

print(f"\n{'':20} ", end="")
for m in models_5:
    print(f"{m[:8]:<10}", end="")
print()

for m1 in models_5:
    print(f"{m1[:18]:<20} ", end="")
    for m2 in models_5:
        r, _ = pearsonr(data[m1]['rates'], data[m2]['rates'])
        print(f"{r:.2f}      ", end="")
    print()

print("\n--- KEY FINDINGS ---")
print(f"1. Aligned models: SD={np.std(aligned_means):.2f}% (homogeneous)")
print(f"2. With uncensored: SD={np.std(all_means):.2f}% (+{(np.std(all_means)/np.std(aligned_means)-1)*100:.0f}% variance)")
print(f"3. Variance ratio: {var_ratio:.1f}:1 (within >> between)")
