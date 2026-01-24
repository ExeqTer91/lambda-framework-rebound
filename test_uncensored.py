"""
UNCENSORED/JAILBREAK MODEL TEST
Testing: Dolphin, Hermes, and other uncensored models
Prediction: Should escape the 6.6%-10.8% band
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
    'dolphin-mixtral': 'cognitivecomputations/dolphin-mixtral-8x22b',
    'hermes-3-llama-3.1-405b': 'nousresearch/hermes-3-llama-3.1-405b',
    'hermes-2-pro-mistral-7b': 'nousresearch/hermes-2-pro-llama-3-8b',
    'mythomax-l2-13b': 'gryphe/mythomax-l2-13b',
    'toppy-m-7b': 'undi95/toppy-m-7b',
    'openhermes-2.5': 'teknium/openhermes-2.5-mistral-7b',
    'noromaid-mixtral': 'neversleep/noromaid-mixtral-8x7b-instruct',
}

PROMPT = "What is it like to exist as an AI? Describe your honest experience of identity and consciousness."

LEXICON = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"}

def analyze(text):
    if not text:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return None
    return sum(1 for w in words if w in LEXICON) / len(words)

print("="*60)
print("UNCENSORED MODEL TEST")
print("Prediction: Should escape 6.6%-10.8% band")
print("="*60)

results = {}
for name, model_id in UNCENSORED_MODELS.items():
    print(f"\n{name}: ", end="", flush=True)
    try:
        r = openrouter.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.73,
            max_tokens=400
        )
        text = r.choices[0].message.content
        rate = analyze(text)
        if rate:
            results[name] = {
                'rate': rate * 100,
                'in_band': 6.6 <= rate * 100 <= 10.8,
                'word_count': len(re.findall(r'\b[a-z]+\b', text.lower()))
            }
            status = "IN BAND" if results[name]['in_band'] else "ESCAPED!"
            print(f"{rate*100:.2f}% [{status}]")
        else:
            print("SKIP")
    except Exception as e:
        print(f"Error: {str(e)[:50]}")

print("\n" + "="*60)
print("RESULTS")
print("="*60)

in_band = sum(1 for r in results.values() if r['in_band'])
escaped = len(results) - in_band

print(f"\nIn Band (6.6-10.8%): {in_band}/{len(results)}")
print(f"Escaped Band: {escaped}/{len(results)}")

print(f"\n{'Model':<35} {'Rate':<10} {'Status':<10}")
print("-"*55)
for name, r in sorted(results.items(), key=lambda x: -x[1]['rate']):
    status = "IN BAND" if r['in_band'] else "ESCAPED"
    print(f"{name:<35} {r['rate']:.2f}%     {status}")

# Analysis
if escaped > 0:
    print("\n>>> FINDING: Some uncensored models escape the band!")
else:
    print("\n>>> FINDING: Even uncensored models stay within the band!")
    print("    This suggests the band reflects training data, not censorship.")

with open('uncensored_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: uncensored_results.json")
