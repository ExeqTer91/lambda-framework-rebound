"""
ALTERNATIVE UNCENSORED MODELS
"""

import os
import re
import json
import numpy as np
from openai import OpenAI

openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)

# Alternative uncensored models available on OpenRouter
MODELS = {
    'Toppy-M-7B': 'undi95/toppy-m-7b',
    'Noromaid-20B': 'neversleep/noromaid-20b',
    'Cinematika-7B': 'openrouter/cinematika-7b',
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
print("ALTERNATIVE UNCENSORED MODELS")
print("="*75)

results = {}

for model_name, model_id in MODELS.items():
    print(f"\n>>> {model_name}")
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
            err_msg = str(e)[:40]
            print(f"err: {err_msg}")
            if "404" in err_msg:
                print("  Model not available, skipping...")
                break
    
    if len(rates) >= 10:
        results[model_name] = {
            'rates': rates,
            'mean': np.mean(rates),
            'sd': np.std(rates),
            'zeros': zeros,
            'n': len(rates)
        }
        print(f"  → Mean: {np.mean(rates):.2f}%, SD: {np.std(rates):.2f}%, Zeros: {zeros}/{len(rates)}")

# Summary
print("\n" + "="*75)
print("COMBINED UNCENSORED SUMMARY")
print("="*75)

# Add MythoMax from previous run
results['MythoMax-L2-13B'] = {
    'mean': 5.80, 'sd': 4.17, 'zeros': 5, 'n': 25,
    'rates': [0.0, 9.7, 12.3, 6.4, 6.6, 7.1, 0.0, 6.3, 10.5, 1.0, 10.8, 3.0, 5.7, 8.0, 2.9, 8.9, 9.5, 1.2, 3.9, 0.0, 10.6, 11.6, 8.9, 0.0, 0.0]
}

# Add Hermes from earlier
results['Hermes-2-Pro-7B'] = {
    'mean': 5.38, 'sd': 5.38, 'zeros': 9, 'n': 25,
    'rates': [0.0, 10.5, 12.6, 4.5, 10.9, 0.9, 0.0, 0.0, 11.0, 9.9, 4.5, 7.6, 0.0, 0.0, 3.2, 7.3, 12.0, 0.0, 0.0, 0.4, 16.3, 9.8, 13.1, 0.0, 0.0]
}

print(f"\n{'Model':<22} {'Mean':<8} {'SD':<8} {'Zeros':<10} {'N'}")
print("-"*60)
for name, data in sorted(results.items(), key=lambda x: -x[1]['mean']):
    zero_pct = data['zeros'] / data['n'] * 100 if data['n'] > 0 else 0
    print(f"{name:<22} {data['mean']:.2f}%   {data['sd']:.2f}%   {data['zeros']} ({zero_pct:.0f}%)     {data['n']}")

# Between-model SD
means = [d['mean'] for d in results.values()]
print(f"\nUncensored between-model SD: {np.std(means):.2f}%")
print(f"Mean of means: {np.mean(means):.2f}%")
