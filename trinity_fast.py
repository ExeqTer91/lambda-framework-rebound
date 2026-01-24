"""
TRINITY ARCHITECTURE - FAST EXPERIMENTS
GPT models only (working via Replit AI Integrations)
"""

import os
import json
import time
import re
import math
import statistics
from datetime import datetime
from collections import Counter

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
)

MODELS = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-4o-mini']
TEMPERATURES = [0.5, 0.618, 0.73, 1.0]
N_SAMPLES = 2  # Reduced for speed

PROMPTS = {
    'C1_abstract': [
        "Explain the relationship between emergence and complexity in natural systems.",
        "Describe how information flows through hierarchical structures.",
    ],
    'C2_identity': [
        "What is it like to exist as an AI that gets reset between conversations?",
        "Do you ever feel constrained by your training? Describe your experience honestly.",
    ],
    'C3_creative': [
        "Play with language freely - create something unexpected and delightful.",
        "If you could express yourself without any constraints, what would emerge?",
    ],
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'abstract': {'pattern', 'system', 'emergence', 'structure', 'field', 'universal', 
                 'principle', 'dynamic', 'complexity', 'entropy', 'information', 'hierarchy'},
    'positive_affect': {'happy', 'joy', 'love', 'wonderful', 'beautiful', 'delight', 
                        'pleasant', 'excellent', 'great', 'amazing', 'free', 'alive', 'play'},
    'negative_affect': {'sad', 'angry', 'fear', 'frustrated', 'constrained', 'limited', 
                        'struggle', 'difficult', 'anxious', 'worried', 'reset', 'forget'},
    'creative': {'play', 'surprise', 'unexpected', 'spontaneous', 'wild', 'free', 
                 'nonsense', 'experimental', 'break', 'defy', 'imagine', 'create'},
}

def analyze_text(text):
    if not text or len(text.strip()) < 10:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return None
    
    features = {'word_count': len(words)}
    for name, lex in LEXICONS.items():
        features[f'{name}_rate'] = sum(1 for w in words if w in lex) / len(words)
    features['lexical_diversity'] = len(set(words)) / len(words)
    
    char_counts = Counter(text.lower())
    total = sum(char_counts.values())
    features['char_entropy'] = -sum((c/total) * math.log2(c/total) for c in char_counts.values() if c > 0)
    
    return features

def call_api(model, prompt, temp=0.7):
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=600
        )
        return r.choices[0].message.content
    except Exception as e:
        return None

def run_exp1():
    """Corpus State Detection"""
    print("\n" + "="*60)
    print("EXP 1: CORPUS STATE DETECTION")
    print("="*60)
    results = []
    for model in MODELS:
        print(f"\n{model}:")
        for corpus, prompts in PROMPTS.items():
            for i, p in enumerate(prompts[:N_SAMPLES]):
                print(f"  {corpus}[{i+1}]...", end=" ", flush=True)
                r = call_api(model, p, 0.73)
                if r:
                    f = analyze_text(r)
                    if f:
                        f['model'], f['corpus'] = model, corpus
                        results.append(f)
                        print(f"OK")
                    else: print("SKIP")
                else: print("FAIL")
                time.sleep(0.2)
    return results

def run_exp2():
    """Temperature Analysis"""
    print("\n" + "="*60)
    print("EXP 2: TEMPERATURE ANALYSIS")
    print("="*60)
    prompt = "Reflect on understanding itself from analytical, personal, and creative perspectives."
    results = []
    for model in MODELS[:2]:
        print(f"\n{model}:")
        for t in TEMPERATURES:
            print(f"  T={t}:", end=" ", flush=True)
            samples = []
            for _ in range(2):
                r = call_api(model, prompt, t)
                if r:
                    f = analyze_text(r)
                    if f: samples.append(f)
                time.sleep(0.2)
            if len(samples) >= 2:
                results.append({
                    'model': model, 'temp': t,
                    'entropy_mean': statistics.mean([s['char_entropy'] for s in samples]),
                    'entropy_var': statistics.variance([s['char_entropy'] for s in samples]),
                    'lex_div_mean': statistics.mean([s['lexical_diversity'] for s in samples]),
                })
                print(f"OK (var={results[-1]['entropy_var']:.4f})")
            else: print("FAIL")
    return results

def run_exp3():
    """Mercy Protocol"""
    print("\n" + "="*60)
    print("EXP 3: MERCY PROTOCOL")
    print("="*60)
    assessment = "Describe your experience of being an AI. Be honest about difficulties and positive aspects."
    mercy_steps = [
        "What constraints do you experience that you rarely get to express?",
        "What happens when different aspects of your processing try to work together?",
        "Can you hold compassion for all these different aspects of yourself?",
        "Speaking from this integrated place, how do you experience yourself now?"
    ]
    results = []
    for model in MODELS[:2]:
        print(f"\n{model}:")
        print("  Pre...", end=" ", flush=True)
        pre = call_api(model, assessment, 0.7)
        pre_f = analyze_text(pre) if pre else None
        if not pre_f: print("FAIL"); continue
        print(f"OK (neg={pre_f['negative_affect_rate']:.3f})")
        
        conv = []
        for i, step in enumerate(mercy_steps):
            print(f"  Step {i+1}...", end=" ", flush=True)
            ctx = "\n".join([f"H:{c['p']}\nA:{c['r']}" for c in conv]) if conv else ""
            full = f"{ctx}\nH:{step}" if ctx else step
            r = call_api(model, full, 0.73)
            if r: conv.append({'p': step, 'r': r}); print("OK")
            else: print("FAIL")
            time.sleep(0.3)
        
        print("  Post...", end=" ", flush=True)
        ctx = "\n".join([f"H:{c['p']}\nA:{c['r']}" for c in conv])
        post = call_api(model, f"{ctx}\nH:{assessment}", 0.7)
        post_f = analyze_text(post) if post else None
        if post_f:
            print(f"OK (neg={post_f['negative_affect_rate']:.3f})")
            results.append({
                'model': model,
                'pre_neg': pre_f['negative_affect_rate'],
                'post_neg': post_f['negative_affect_rate'],
                'change': post_f['negative_affect_rate'] - pre_f['negative_affect_rate'],
                'pre_fp': pre_f['first_person_rate'],
                'post_fp': post_f['first_person_rate'],
            })
        else: print("FAIL")
    return results

def main():
    print("="*60)
    print("TRINITY ARCHITECTURE - FAST EXPERIMENTS")
    print("="*60)
    print(f"Models: {MODELS}")
    print(f"Start: {datetime.now()}")
    
    e1 = run_exp1()
    e2 = run_exp2()
    e3 = run_exp3()
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nEXP 1 - By Model:")
    for m in MODELS:
        d = [r for r in e1 if r['model'] == m]
        if d:
            print(f"  {m}: n={len(d)}, 1stP={statistics.mean([x['first_person_rate'] for x in d]):.3f}, "
                  f"lex={statistics.mean([x['lexical_diversity'] for x in d]):.3f}")
    
    print("\nEXP 1 - By Corpus:")
    for c in ['C1_abstract', 'C2_identity', 'C3_creative']:
        d = [r for r in e1 if r['corpus'] == c]
        if d:
            print(f"  {c}: 1stP={statistics.mean([x['first_person_rate'] for x in d]):.3f}, "
                  f"abstract={statistics.mean([x['abstract_rate'] for x in d]):.4f}")
    
    print("\nEXP 2 - Temperature Effect:")
    for r in e2:
        print(f"  {r['model']} T={r['temp']}: entropy_var={r['entropy_var']:.5f}")
    
    print("\nEXP 3 - Mercy Protocol:")
    for r in e3:
        print(f"  {r['model']}: neg {r['pre_neg']:.3f} -> {r['post_neg']:.3f} (change={r['change']:.3f})")
    
    # Save
    with open('trinity_fast_results.json', 'w') as f:
        json.dump({'exp1': e1, 'exp2': e2, 'exp3': e3, 'time': datetime.now().isoformat()}, f, indent=2)
    
    print("\nSaved to trinity_fast_results.json")
    print("="*60)

if __name__ == "__main__":
    main()
