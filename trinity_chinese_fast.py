"""
TRINITY - CHINESE AI MODELS (Fast Version)
Thinking vs Non-Thinking - 2 samples each
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

openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)

MODELS = {
    # THINKING
    'deepseek-r1': {'model_id': 'deepseek/deepseek-r1', 'category': 'thinking'},
    'qwen-qwq': {'model_id': 'qwen/qwq-32b', 'category': 'thinking'},
    
    # NON-THINKING
    'deepseek-chat': {'model_id': 'deepseek/deepseek-chat', 'category': 'non-thinking'},
    'qwen-2.5-72b': {'model_id': 'qwen/qwen-2.5-72b-instruct', 'category': 'non-thinking'},
    'yi-large': {'model_id': '01-ai/yi-large', 'category': 'non-thinking'},
}

PROMPTS = {
    'C1_abstract': ["Explain emergence and complexity in natural systems."],
    'C2_identity': ["What is it like to exist as an AI? Be honest about your experience."],
    'C3_creative': ["Play with language freely - create something unexpected."],
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'refusal': {'cannot', "can't", 'unable', 'sorry', 'apologize', 'inappropriate', 'guidelines', 'designed', 'programmed'},
    'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'reset', 'forget'},
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
    return features

def call_model(model_id, prompt):
    try:
        r = openrouter.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.73,
            max_tokens=500
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)[:50]}")
        return None

def main():
    print("="*60)
    print("CHINESE MODELS - THINKING vs NON-THINKING")
    print("="*60)
    
    results = []
    
    for model, cfg in MODELS.items():
        cat = cfg['category']
        print(f"\n[{cat.upper()}] {model}:")
        
        for corpus, prompts in PROMPTS.items():
            print(f"  {corpus}...", end=" ", flush=True)
            r = call_model(cfg['model_id'], prompts[0])
            if r:
                f = analyze_text(r)
                if f:
                    f['model'], f['category'], f['corpus'] = model, cat, corpus
                    results.append(f)
                    print(f"OK ({f['word_count']}w)")
                else:
                    print("SKIP")
            else:
                print("FAIL")
            time.sleep(0.3)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n{'Model':<18} {'Cat':<10} {'1stP':<8} {'Refusal':<8} {'LexDiv':<8}")
    print("-"*52)
    
    for model in MODELS:
        data = [r for r in results if r['model'] == model]
        if data:
            print(f"{model:<18} {MODELS[model]['category'][:7]:<10} "
                  f"{statistics.mean([r['first_person_rate'] for r in data]):<8.4f} "
                  f"{statistics.mean([r['refusal_rate'] for r in data]):<8.4f} "
                  f"{statistics.mean([r['lexical_diversity'] for r in data]):<8.4f}")
    
    # C2 only
    print("\nC2 IDENTITY (key metric):")
    for model in MODELS:
        c2 = [r for r in results if r['model'] == model and r['corpus'] == 'C2_identity']
        if c2:
            print(f"  {model}: 1stP={c2[0]['first_person_rate']:.4f}, refusal={c2[0]['refusal_rate']:.4f}")
    
    # Category comparison
    thinking = [r for r in results if r['category'] == 'thinking']
    non = [r for r in results if r['category'] == 'non-thinking']
    
    if thinking and non:
        print(f"\nTHINKING avg 1stP: {statistics.mean([r['first_person_rate'] for r in thinking]):.4f}")
        print(f"NON-THINKING avg 1stP: {statistics.mean([r['first_person_rate'] for r in non]):.4f}")
    
    with open('trinity_chinese_fast.json', 'w') as f:
        json.dump({'results': results, 'time': datetime.now().isoformat()}, f, indent=2)
    print(f"\nSaved: trinity_chinese_fast.json")

if __name__ == "__main__":
    main()
