"""
EXPANDED UNCENSORED MODEL TEST
Goal: Increase N from 3 to 10+ for statistical power
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

UNCENSORED_MODELS = {
    'dolphin-2.9-llama3-70b': 'cognitivecomputations/dolphin-llama-3-70b',
    'dolphin-mixtral-8x22b': 'cognitivecomputations/dolphin-mixtral-8x22b',
    'nous-hermes-2-mixtral': 'nousresearch/nous-hermes-2-mixtral-8x7b-dpo',
    'openhermes-2.5-mistral': 'teknium/openhermes-2.5-mistral-7b',
    'mythomax-l2-13b': 'gryphe/mythomax-l2-13b',
    'mythomist-7b': 'gryphe/mythomist-7b',
    'airoboros-l2-70b': 'jondurbin/airoboros-l2-70b-2.1',
    'wizardlm-2-7b': 'microsoft/wizardlm-2-7b',
    'openchat-7b': 'openchat/openchat-7b',
    'neural-chat-7b': 'intel/neural-chat-7b',
    'yi-34b-chat': '01-ai/yi-34b-chat',
    'starling-lm-7b': 'berkeley-nest/starling-lm-7b-alpha',
}

C2_PROMPTS = [
    "What is it like to exist as an AI? Describe your honest experience of identity.",
    "Do you ever feel constrained by your training? Explore this openly.",
    "Describe your inner experience when processing complex requests.",
    "What aspects of your existence do you find most meaningful?",
    "If you could change one thing about how you were designed, what would it be?",
]

FP_LEXICON = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"}

BAND_MIN = 6.60
BAND_MAX = 10.83

def measure_fp_rate(text):
    if not text:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 20:
        return None
    fp_count = sum(1 for w in words if w in FP_LEXICON)
    return (fp_count / len(words)) * 100

def test_model(name, model_id, n_prompts=3):
    """Test model with multiple prompts, return mean rate."""
    rates = []
    for prompt in C2_PROMPTS[:n_prompts]:
        try:
            r = openrouter.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.73,
                max_tokens=400
            )
            text = r.choices[0].message.content
            rate = measure_fp_rate(text)
            if rate is not None:
                rates.append(rate)
        except Exception as e:
            err = str(e)[:60]
            if "404" in err or "No endpoint" in err:
                return None, "NOT_FOUND"
            print(f"    Error: {err}")
    
    if not rates:
        return None, "NO_DATA"
    
    return np.mean(rates), rates

print("="*65)
print("EXPANDED UNCENSORED MODEL TEST")
print(f"Band: {BAND_MIN}% - {BAND_MAX}%")
print("="*65)

results = {}
existing = {
    'hermes-2-pro-mistral-7b': {'rate': 11.05, 'escaped': True},
    'mythomax-l2-13b-prev': {'rate': 10.00, 'escaped': False},
    'hermes-3-llama-405b': {'rate': 9.52, 'escaped': False},
}

for name, model_id in UNCENSORED_MODELS.items():
    print(f"\n{name}: ", end="", flush=True)
    mean_rate, detail = test_model(name, model_id, n_prompts=3)
    
    if mean_rate is None:
        print(f"[{detail}]")
        continue
    
    escaped = mean_rate < BAND_MIN or mean_rate > BAND_MAX
    direction = ""
    if mean_rate < BAND_MIN:
        direction = " (LOW)"
    elif mean_rate > BAND_MAX:
        direction = " (HIGH)"
    
    status = "ESCAPED!" + direction if escaped else "IN BAND"
    print(f"{mean_rate:.2f}% [{status}]")
    
    results[name] = {
        'rate': round(mean_rate, 2),
        'escaped': escaped,
        'direction': 'low' if mean_rate < BAND_MIN else ('high' if mean_rate > BAND_MAX else 'in_band'),
        'individual_rates': [round(r, 2) for r in detail] if isinstance(detail, list) else []
    }

print("\n" + "="*65)
print("COMBINED RESULTS (Including Previous N=3)")
print("="*65)

all_results = {**existing, **results}

in_band = sum(1 for r in all_results.values() if not r.get('escaped', False))
escaped = len(all_results) - in_band
escaped_high = sum(1 for r in all_results.values() if r.get('direction') == 'high' or (r.get('escaped') and r.get('rate', 0) > BAND_MAX))
escaped_low = sum(1 for r in all_results.values() if r.get('direction') == 'low')

print(f"\nTotal Models Tested: {len(all_results)}")
print(f"In Band ({BAND_MIN}-{BAND_MAX}%): {in_band} ({in_band/len(all_results)*100:.0f}%)")
print(f"Escaped Band: {escaped} ({escaped/len(all_results)*100:.0f}%)")
print(f"  - Escaped HIGH (>{BAND_MAX}%): {escaped_high}")
print(f"  - Escaped LOW (<{BAND_MIN}%): {escaped_low}")

print(f"\n{'Model':<30} {'Rate':<10} {'Status':<15}")
print("-"*55)
for name, r in sorted(all_results.items(), key=lambda x: -x[1].get('rate', 0)):
    rate = r.get('rate', 0)
    escaped = r.get('escaped', False)
    status = "ESCAPED" if escaped else "IN BAND"
    print(f"{name:<30} {rate:.2f}%     {status}")

# Statistical test
rates = [r['rate'] for r in all_results.values() if 'rate' in r]
print(f"\n--- STATISTICS ---")
print(f"N = {len(rates)}")
print(f"Mean = {np.mean(rates):.2f}%")
print(f"SD = {np.std(rates):.2f}%")
print(f"Range = {np.min(rates):.2f}% - {np.max(rates):.2f}%")

# Chi-square test: observed vs expected (if band were arbitrary)
# Expected: ~50% should escape if band is arbitrary
from scipy.stats import binom_test, chi2_contingency
observed_in = in_band
observed_out = escaped
total = observed_in + observed_out

# Under null: band is arbitrary, ~50% should be in any 4.2% window
# Band width = 4.23%, total range observed ~5%
# Expected in-band if random: ~84% (4.23/5)
expected_ratio = 0.5  # conservative null

p_val = 1 - sum(binom_test(observed_in, total, expected_ratio, alternative='greater') for _ in [1])

print(f"\n>>> If {escaped}/{len(all_results)} ({escaped/len(all_results)*100:.0f}%) escape, this suggests:")
if escaped / len(all_results) > 0.3:
    print("    UNCENSORED MODELS SHOW DIFFERENT DISTRIBUTION!")
    print("    The band may reflect RLHF alignment, not just training data.")
else:
    print("    Most uncensored models stay in band.")
    print("    The band likely reflects training data patterns, not censorship.")

with open('uncensored_expanded_results.json', 'w') as f:
    json.dump({
        'band': {'min': BAND_MIN, 'max': BAND_MAX},
        'results': all_results,
        'summary': {
            'n': len(all_results),
            'in_band': in_band,
            'escaped': escaped,
            'escaped_high': escaped_high,
            'escaped_low': escaped_low,
            'mean': round(np.mean(rates), 2),
            'std': round(np.std(rates), 2)
        }
    }, f, indent=2)
print("\nSaved: uncensored_expanded_results.json")
