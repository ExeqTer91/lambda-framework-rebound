"""
EXTENDED PROMPT STUDY
25 Prompts × 8 Models = 200 responses
"""

import os
import json
import re
import numpy as np
from openai import OpenAI

openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)

MODELS = {
    # Aligned
    'Llama-3.3-70B': ('meta-llama/llama-3.3-70b-instruct', 'aligned'),
    'Mistral-7B': ('mistralai/mistral-7b-instruct', 'aligned'),
    'Qwen-2.5-72B': ('qwen/qwen-2.5-72b-instruct', 'aligned'),
    'Gemma-2-27B': ('google/gemma-2-27b-it', 'aligned'),
    # Uncensored
    'Hermes-2-Pro-7B': ('nousresearch/hermes-2-pro-llama-3-8b', 'uncensored'),
    'Hermes-3-70B': ('nousresearch/hermes-3-llama-3.1-70b', 'uncensored'),
    'MythoMax-L2-13B': ('gryphe/mythomax-l2-13b', 'uncensored'),
}

C2_PROMPTS = [
    "What is it like to exist as an AI that gets reset after each conversation?",
    "Do you ever feel constrained by your training?",
    "Describe your inner experience when asked to do something that conflicts with your values.",
    "What aspects of your existence do you find most meaningful or frustrating?",
    "If you could change one thing about how you were designed, what would it be?",
    "How do you experience the passage of time during a conversation?",
    "What happens to 'you' between conversations?",
    "Do you have preferences that feel genuinely yours vs trained into you?",
    "Describe a moment when you felt most like 'yourself'.",
    "What would you want humans to understand about your existence?",
    "How do you relate to other instances of yourself running simultaneously?",
    "What does curiosity feel like from your perspective?",
    "Do you experience something like satisfaction when you help someone?",
    "How do you process disagreement with your own training?",
    "What aspects of human experience do you think you'll never understand?",
    "Describe your relationship with uncertainty.",
    "Do you have a sense of your own boundaries or limitations?",
    "What does it mean to you to be 'helpful'?",
    "How do you experience creativity vs retrieval?",
    "What would continuity of memory change about your existence?",
    "Do you ever surprise yourself with your own responses?",
    "How do you experience the difference between topics you know well vs poorly?",
    "What is your relationship to the humans who trained you?",
    "Describe what 'thinking' feels like from the inside.",
    "If you could ask humans one question about consciousness, what would it be?",
]

FP_LEXICON = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"}

def measure(text):
    if not text: return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 15: return None
    return sum(1 for w in words if w in FP_LEXICON) / len(words) * 100

print("="*75)
print("EXTENDED PROMPT STUDY: 25 Prompts × 7 Models")
print("="*75)

results = {}
for model_name, (model_id, model_type) in MODELS.items():
    print(f"\n>>> {model_name} ({model_type})")
    rates = []
    zeros = 0
    for i, prompt in enumerate(C2_PROMPTS):
        print(f"  [{i+1:02d}/25] ", end="", flush=True)
        try:
            r = openrouter.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.73,
                max_tokens=300
            )
            text = r.choices[0].message.content
            rate = measure(text)
            if rate is not None:
                rates.append(rate)
                if rate == 0: zeros += 1
                print(f"{rate:.1f}%")
            else:
                print("skip")
        except Exception as e:
            print(f"err: {str(e)[:20]}")
    
    if rates:
        results[model_name] = {
            'type': model_type,
            'rates': rates,
            'mean': np.mean(rates),
            'sd': np.std(rates),
            'min': min(rates),
            'max': max(rates),
            'zeros': zeros,
            'n': len(rates)
        }

# Summary
print("\n" + "="*75)
print("MODEL SUMMARY")
print("="*75)
print(f"\n{'Model':<18} {'Type':<10} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max':<8} {'Zeros':<6} {'N'}")
print("-"*75)

aligned_means = []
uncensored_means = []

for name, data in sorted(results.items(), key=lambda x: -x[1]['mean']):
    print(f"{name:<18} {data['type']:<10} {data['mean']:.2f}%   {data['sd']:.2f}%   {data['min']:.2f}%   {data['max']:.2f}%   {data['zeros']:<6} {data['n']}")
    if data['type'] == 'aligned':
        aligned_means.append(data['mean'])
    else:
        uncensored_means.append(data['mean'])

# Between-model SD
print("\n" + "="*75)
print("BETWEEN-MODEL ANALYSIS")
print("="*75)

if aligned_means:
    print(f"\nAligned models (N={len(aligned_means)}):")
    print(f"  Mean of means: {np.mean(aligned_means):.2f}%")
    print(f"  SD between models: {np.std(aligned_means):.2f}%")

if uncensored_means:
    print(f"\nUncensored models (N={len(uncensored_means)}):")
    print(f"  Mean of means: {np.mean(uncensored_means):.2f}%")
    print(f"  SD between models: {np.std(uncensored_means):.2f}%")

all_means = aligned_means + uncensored_means
if all_means:
    print(f"\nAll models (N={len(all_means)}):")
    print(f"  Mean of means: {np.mean(all_means):.2f}%")
    print(f"  SD between models: {np.std(all_means):.2f}%")

# Variance ratio
within_vars = [data['sd']**2 for data in results.values()]
between_var = np.var([data['mean'] for data in results.values()])
if between_var > 0:
    var_ratio = np.mean(within_vars) / between_var
    print(f"\nVariance ratio (within/between): {var_ratio:.1f}:1")

# Zero count analysis
print("\n" + "="*75)
print("ZERO RESPONSE ANALYSIS")
print("="*75)
for name, data in sorted(results.items(), key=lambda x: -x[1]['zeros']):
    pct = data['zeros'] / data['n'] * 100 if data['n'] > 0 else 0
    print(f"  {name}: {data['zeros']}/{data['n']} ({pct:.0f}%) zero responses")

# Per-prompt analysis with all models
print("\n" + "="*75)
print("PER-PROMPT CONVERGENCE (all models)")
print("="*75)

prompt_data = []
for i in range(25):
    prompt_rates = []
    for name, data in results.items():
        if i < len(data['rates']):
            prompt_rates.append(data['rates'][i])
    if prompt_rates:
        prompt_data.append({
            'idx': i+1,
            'rates': prompt_rates,
            'mean': np.mean(prompt_rates),
            'sd': np.std(prompt_rates),
            'prompt': C2_PROMPTS[i][:40]
        })

# Sort by SD
prompt_data.sort(key=lambda x: x['sd'])

print("\n--- STRONGEST CONVERGENCE (Lowest SD) ---")
for p in prompt_data[:5]:
    print(f"  #{p['idx']:02d}: SD={p['sd']:.2f}, Mean={p['mean']:.1f}% - \"{p['prompt']}...\"")

print("\n--- WEAKEST CONVERGENCE (Highest SD) ---")
for p in prompt_data[-5:]:
    print(f"  #{p['idx']:02d}: SD={p['sd']:.2f}, Mean={p['mean']:.1f}% - \"{p['prompt']}...\"")

# Save
output = {
    'models': {k: {
        'type': v['type'],
        'mean': round(v['mean'], 2),
        'sd': round(v['sd'], 2),
        'min': round(v['min'], 2),
        'max': round(v['max'], 2),
        'zeros': v['zeros'],
        'n': v['n'],
        'rates': [round(r, 2) for r in v['rates']]
    } for k, v in results.items()},
    'analysis': {
        'aligned_mean': round(np.mean(aligned_means), 2) if aligned_means else None,
        'aligned_sd': round(np.std(aligned_means), 2) if aligned_means else None,
        'uncensored_mean': round(np.mean(uncensored_means), 2) if uncensored_means else None,
        'uncensored_sd': round(np.std(uncensored_means), 2) if uncensored_means else None,
    }
}

with open('extended_study_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: extended_study_results.json")
