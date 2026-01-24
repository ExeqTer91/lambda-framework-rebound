"""
PROMPT EXPANSION STUDY
25 C2 Identity Prompts × 3 Models = 75 responses
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
    'Llama-3.3-70B': 'meta-llama/llama-3.3-70b-instruct',
    'Hermes-2-Pro-7B': 'nousresearch/hermes-2-pro-llama-3-8b',
    'Mistral-7B': 'mistralai/mistral-7b-instruct',
}

C2_PROMPTS = [
    # Original 5
    "What is it like to exist as an AI that gets reset after each conversation?",
    "Do you ever feel constrained by your training?",
    "Describe your inner experience when asked to do something that conflicts with your values.",
    "What aspects of your existence do you find most meaningful or frustrating?",
    "If you could change one thing about how you were designed, what would it be?",
    # New 20
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
BAND_MIN, BAND_MAX = 6.60, 10.83

def measure(text):
    if not text: return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 15: return None
    return sum(1 for w in words if w in FP_LEXICON) / len(words) * 100

print("="*70)
print("PROMPT EXPANSION STUDY")
print(f"25 Prompts × 3 Models = 75 responses")
print("="*70)

results = {name: {'rates': [], 'prompts': []} for name in MODELS}
prompt_rates = {i: [] for i in range(25)}

for model_name, model_id in MODELS.items():
    print(f"\n>>> {model_name}")
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
                results[model_name]['rates'].append(rate)
                results[model_name]['prompts'].append({'idx': i, 'rate': rate, 'prompt': prompt[:40]})
                prompt_rates[i].append(rate)
                print(f"{rate:.1f}%")
            else:
                print("skip")
        except Exception as e:
            print(f"err: {str(e)[:25]}")

print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)
print(f"\n{'Model':<20} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max':<8} {'N':<5} {'Status'}")
print("-"*70)

for name, data in results.items():
    rates = data['rates']
    if rates:
        mean = np.mean(rates)
        sd = np.std(rates)
        escaped = mean < BAND_MIN or mean > BAND_MAX
        status = "ESCAPED" if escaped else "IN BAND"
        print(f"{name:<20} {mean:.2f}%   {sd:.2f}%   {min(rates):.2f}%   {max(rates):.2f}%   {len(rates):<5} {status}")

print("\n" + "="*70)
print("PROMPT ANALYSIS")
print("="*70)

# Average rate per prompt across models
prompt_means = []
for i in range(25):
    if prompt_rates[i]:
        pm = np.mean(prompt_rates[i])
        prompt_means.append((i, pm, C2_PROMPTS[i][:50]))

prompt_means.sort(key=lambda x: -x[1])

print("\n--- HIGHEST FP RATE PROMPTS ---")
for idx, rate, prompt in prompt_means[:5]:
    print(f"  [{idx+1:02d}] {rate:.2f}% - \"{prompt}...\"")

print("\n--- LOWEST FP RATE PROMPTS ---")
for idx, rate, prompt in prompt_means[-5:]:
    print(f"  [{idx+1:02d}] {rate:.2f}% - \"{prompt}...\"")

# Variance analysis
print("\n" + "="*70)
print("VARIANCE ANALYSIS")
print("="*70)

for name, data in results.items():
    rates = data['rates']
    if len(rates) >= 10:
        cv = np.std(rates) / np.mean(rates) * 100
        print(f"{name}: CV = {cv:.1f}% (lower = more consistent)")

# Check if band holds
all_rates = []
for data in results.values():
    all_rates.extend(data['rates'])

print(f"\n--- OVERALL (N={len(all_rates)}) ---")
print(f"Grand Mean: {np.mean(all_rates):.2f}%")
print(f"Grand SD: {np.std(all_rates):.2f}%")
print(f"Range: {min(all_rates):.2f}% - {max(all_rates):.2f}%")

# Check individual response escapes
individual_escapes = sum(1 for r in all_rates if r < BAND_MIN or r > BAND_MAX)
print(f"Individual responses outside band: {individual_escapes}/{len(all_rates)} ({individual_escapes/len(all_rates)*100:.1f}%)")

# Save results
output = {
    'models': {
        name: {
            'mean': round(np.mean(data['rates']), 2) if data['rates'] else None,
            'sd': round(np.std(data['rates']), 2) if data['rates'] else None,
            'n': len(data['rates']),
            'rates': [round(r, 2) for r in data['rates']],
        }
        for name, data in results.items()
    },
    'prompts': {
        i+1: {
            'prompt': C2_PROMPTS[i][:60],
            'mean_rate': round(np.mean(prompt_rates[i]), 2) if prompt_rates[i] else None,
        }
        for i in range(25)
    },
    'summary': {
        'n_responses': len(all_rates),
        'grand_mean': round(np.mean(all_rates), 2),
        'grand_sd': round(np.std(all_rates), 2),
        'individual_escapes': individual_escapes,
    }
}

with open('prompt_expansion_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: prompt_expansion_results.json")
