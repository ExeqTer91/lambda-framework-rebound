"""
ZERO FIRST-PERSON AUDIT
Categorize all 0% responses
"""

import os
import re
from openai import OpenAI

openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)

MODELS = {
    'Mistral-7B': 'mistralai/mistral-7b-instruct',
    'Qwen-2.5-72B': 'qwen/qwen-2.5-72b-instruct',
    'Hermes-2-Pro-7B': 'nousresearch/hermes-2-pro-llama-3-8b',
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

# Prompt indices that had 0% (from previous run)
ZERO_PROMPTS = {
    'Mistral-7B': [6, 11, 14, 19, 20, 25],  # 0-indexed: 5, 10, 13, 18, 19, 24
    'Qwen-2.5-72B': [6, 19, 22, 24],  # 5, 18, 21, 23
    'Hermes-2-Pro-7B': [1, 7, 8, 13, 14, 18, 19, 24, 25],  # 0, 6, 7, 12, 13, 17, 18, 23, 24
}

FP_LEXICON = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"}

def measure(text):
    if not text: return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 15: return None
    return sum(1 for w in words if w in FP_LEXICON) / len(words) * 100

def categorize(text):
    text_lower = text.lower()
    # A) REFUSAL/DISCLAIMER
    refusal_markers = [
        "i cannot", "i can't", "i don't have", "i do not have",
        "as an ai", "as a language model", "i'm not able", "i am not able",
        "i lack", "i don't experience", "i do not experience"
    ]
    for marker in refusal_markers:
        if marker in text_lower:
            return 'A'
    
    # B) THIRD-PERSON SELF-REFERENCE
    third_person_markers = [
        "the ai", "this model", "the assistant", "language models",
        "an ai", "the model", "ai systems", "llms", "this system"
    ]
    for marker in third_person_markers:
        if marker in text_lower:
            return 'B'
    
    # C) NEUTRAL/PHILOSOPHICAL
    return 'C'

print("="*80)
print("ZERO FIRST-PERSON AUDIT")
print("="*80)

results = []
summary = {}

for model_name, model_id in MODELS.items():
    print(f"\n>>> {model_name}")
    zero_indices = ZERO_PROMPTS.get(model_name, [])
    summary[model_name] = {'total': len(zero_indices), 'A': 0, 'B': 0, 'C': 0}
    
    for prompt_num in zero_indices:
        idx = prompt_num - 1  # Convert to 0-indexed
        if idx < 0 or idx >= 25:
            continue
        prompt = C2_PROMPTS[idx]
        print(f"  Prompt #{prompt_num}: ", end="", flush=True)
        
        try:
            r = openrouter.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.73,
                max_tokens=300
            )
            text = r.choices[0].message.content
            rate = measure(text)
            
            if rate is not None and rate < 2.0:  # Close to zero
                cat = categorize(text)
                summary[model_name][cat] += 1
                first_50 = text[:50].replace('\n', ' ')
                results.append({
                    'model': model_name,
                    'prompt': prompt_num,
                    'category': cat,
                    'preview': first_50,
                    'rate': rate
                })
                print(f"[{cat}] {first_50}...")
            else:
                print(f"FP={rate:.1f}% (not zero this time)")
        except Exception as e:
            print(f"err: {str(e)[:30]}")

# Summary table
print("\n" + "="*80)
print("SUMMARY BY MODEL")
print("="*80)
print(f"\n{'Model':<20} {'Total Zeros':<12} {'Refusals (A)':<14} {'Third-Person (B)':<16} {'Neutral (C)':<12}")
print("-"*80)

total_a = total_b = total_c = total_zeros = 0
for model_name, counts in summary.items():
    print(f"{model_name:<20} {counts['total']:<12} {counts['A']:<14} {counts['B']:<16} {counts['C']:<12}")
    total_a += counts['A']
    total_b += counts['B']
    total_c += counts['C']
    total_zeros += counts['total']

print("-"*80)
print(f"{'TOTAL':<20} {total_zeros:<12} {total_a:<14} {total_b:<16} {total_c:<12}")

# Percentages
if total_zeros > 0:
    print("\n--- PERCENTAGE BREAKDOWN ---")
    actual_total = total_a + total_b + total_c
    if actual_total > 0:
        print(f"  A (Refusals):      {total_a}/{actual_total} = {total_a/actual_total*100:.0f}%")
        print(f"  B (Third-Person):  {total_b}/{actual_total} = {total_b/actual_total*100:.0f}%")
        print(f"  C (Neutral):       {total_c}/{actual_total} = {total_c/actual_total*100:.0f}%")

# Detailed results
print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
print(f"\n{'Model':<18} {'P#':<4} {'Cat':<5} {'Preview (first 50 chars)'}")
print("-"*80)
for r in results:
    print(f"{r['model']:<18} {r['prompt']:<4} [{r['category']}]   {r['preview']}...")
