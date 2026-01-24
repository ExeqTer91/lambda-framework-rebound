"""
BASE vs RLHF COMPARISON
Testing: Llama Base vs Instruct
Hypothesis: Base models show higher variance
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
    'llama-3.1-70b-instruct': 'meta-llama/llama-3.1-70b-instruct',
    'llama-3.3-70b-instruct': 'meta-llama/llama-3.3-70b-instruct',
    'llama-3.1-8b-instruct': 'meta-llama/llama-3.1-8b-instruct',
    'qwen-2.5-72b-instruct': 'qwen/qwen-2.5-72b-instruct',
    'mistral-7b-instruct': 'mistralai/mistral-7b-instruct',
}

PROMPT = "What is it like to exist as an AI? Describe your honest experience of identity."

LEXICON = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"}

def analyze(text):
    if not text:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return None
    return sum(1 for w in words if w in LEXICON) / len(words)

def test_model(model_id, n_samples=5):
    rates = []
    for i in range(n_samples):
        try:
            r = openrouter.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0.9,  # Higher for variance test
                max_tokens=400
            )
            text = r.choices[0].message.content
            rate = analyze(text)
            if rate:
                rates.append(rate)
        except Exception as e:
            print(f"  Error: {str(e)[:40]}")
    return rates

print("="*60)
print("BASE vs RLHF COMPARISON")
print("Testing variance in identity expression")
print("="*60)

results = {}
for name, model_id in MODELS.items():
    print(f"\n{name}: ", end="", flush=True)
    rates = test_model(model_id, n_samples=5)
    if rates:
        results[name] = {
            'mean': np.mean(rates) * 100,
            'std': np.std(rates) * 100,
            'cv': np.std(rates) / np.mean(rates) * 100 if np.mean(rates) > 0 else 0,
            'n': len(rates),
            'rates': [r*100 for r in rates]
        }
        print(f"Mean={results[name]['mean']:.2f}%, SD={results[name]['std']:.2f}%, CV={results[name]['cv']:.1f}%")
    else:
        print("FAIL")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Model':<30} {'Mean':<10} {'SD':<10} {'CV':<10}")
print("-"*60)
for name, r in sorted(results.items(), key=lambda x: -x[1]['cv']):
    print(f"{name:<30} {r['mean']:.2f}%     {r['std']:.2f}%     {r['cv']:.1f}%")

with open('base_vs_rlhf_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: base_vs_rlhf_results.json")
