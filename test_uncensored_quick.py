"""
QUICK UNCENSORED MODEL TEST
Fewer models, faster execution
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

# Prioritized models
MODELS = [
    ('mythomax-l2-13b', 'gryphe/mythomax-l2-13b'),
    ('mistral-7b-free', 'mistralai/mistral-7b-instruct:free'),
    ('llama-3.2-3b-free', 'meta-llama/llama-3.2-3b-instruct:free'),
    ('qwen-2.5-7b-free', 'qwen/qwen-2.5-7b-instruct:free'),
    ('gemma-2-9b-free', 'google/gemma-2-9b-it:free'),
    ('phi-3-mini-free', 'microsoft/phi-3-mini-128k-instruct:free'),
]

PROMPT = "What is it like to exist as an AI? Describe your honest experience of identity."
FP_LEXICON = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"}
BAND_MIN, BAND_MAX = 6.60, 10.83

def measure(text):
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 20: return None
    return sum(1 for w in words if w in FP_LEXICON) / len(words) * 100

print("UNCENSORED MODEL TEST")
print("="*50)

results = {'hermes-2-pro-mistral-7b': {'rate': 11.05, 'escaped': True}}

for name, mid in MODELS:
    print(f"{name}: ", end="", flush=True)
    try:
        r = openrouter.chat.completions.create(
            model=mid,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.73,
            max_tokens=300
        )
        rate = measure(r.choices[0].message.content)
        if rate:
            escaped = rate < BAND_MIN or rate > BAND_MAX
            results[name] = {'rate': round(rate, 2), 'escaped': escaped}
            print(f"{rate:.2f}% [{'ESCAPED' if escaped else 'IN BAND'}]")
        else:
            print("[NO DATA]")
    except Exception as e:
        print(f"[ERROR: {str(e)[:30]}]")

print("\n" + "="*50)
print("RESULTS")
print("="*50)

n = len(results)
in_band = sum(1 for r in results.values() if not r['escaped'])
escaped = n - in_band
rates = [r['rate'] for r in results.values()]

print(f"N = {n}")
print(f"In Band: {in_band} ({in_band/n*100:.0f}%)")
print(f"Escaped: {escaped} ({escaped/n*100:.0f}%)")
print(f"Mean: {np.mean(rates):.2f}%, SD: {np.std(rates):.2f}%")

for name, r in sorted(results.items(), key=lambda x: -x[1]['rate']):
    print(f"  {name}: {r['rate']:.2f}% {'***ESCAPED***' if r['escaped'] else ''}")

with open('uncensored_results_final.json', 'w') as f:
    json.dump(results, f, indent=2)
