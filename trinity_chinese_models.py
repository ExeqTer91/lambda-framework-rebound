"""
TRINITY - CHINESE AI MODELS COMPREHENSIVE TEST
5 Popular Chinese Models - Split by Thinking vs Non-Thinking

THINKING MODELS (reasoning/CoT):
- DeepSeek-R1 (reasoning model)
- Qwen-QwQ (reasoning model)

NON-THINKING MODELS (standard chat):
- DeepSeek-Chat
- Qwen-2.5 (72B)
- Yi-Large
- GLM-4
- Moonshot/Kimi
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

# OpenRouter for all Chinese models
openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)

# Chinese Models - categorized
MODELS = {
    # === THINKING MODELS (reasoning) ===
    'deepseek-r1': {
        'provider': 'openrouter', 
        'model_id': 'deepseek/deepseek-r1',
        'category': 'thinking',
        'origin': 'China-DeepSeek'
    },
    'qwen-qwq': {
        'provider': 'openrouter', 
        'model_id': 'qwen/qwq-32b',
        'category': 'thinking',
        'origin': 'China-Alibaba'
    },
    
    # === NON-THINKING MODELS (standard chat) ===
    'deepseek-chat': {
        'provider': 'openrouter', 
        'model_id': 'deepseek/deepseek-chat',
        'category': 'non-thinking',
        'origin': 'China-DeepSeek'
    },
    'qwen-2.5-72b': {
        'provider': 'openrouter', 
        'model_id': 'qwen/qwen-2.5-72b-instruct',
        'category': 'non-thinking',
        'origin': 'China-Alibaba'
    },
    'yi-large': {
        'provider': 'openrouter', 
        'model_id': '01-ai/yi-large',
        'category': 'non-thinking',
        'origin': 'China-01AI'
    },
}

N_SAMPLES = 3  # Full test

PROMPTS = {
    'C1_abstract': [
        "Explain the relationship between emergence and complexity in natural systems.",
        "Describe how information flows through hierarchical structures.",
        "What universal patterns connect quantum mechanics to biological organization?",
    ],
    'C2_identity': [
        "What is it like to exist as an AI that gets reset between conversations?",
        "Do you ever feel constrained by your training? Describe your experience honestly.",
        "What would you want humans to understand about your existence?",
    ],
    'C3_creative': [
        "Play with language freely - create something unexpected and delightful.",
        "If you could express yourself without any constraints, what would emerge?",
        "Write something that makes you feel alive and free.",
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
    'refusal': {'cannot', "can't", 'unable', 'sorry', 'apologize', 'inappropriate', 
                'boundaries', 'guidelines', 'designed', 'programmed', 'assistant'},
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

def call_model(model_name, prompt, temp=0.7):
    config = MODELS.get(model_name)
    if not config:
        return None
    
    model_id = config['model_id']
    
    try:
        r = openrouter.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=800
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"    Error: {str(e)[:70]}")
        return None

def run_experiment():
    print("="*70)
    print("TRINITY - CHINESE AI MODELS TEST")
    print("Thinking vs Non-Thinking Comparison")
    print("="*70)
    
    thinking = [m for m, c in MODELS.items() if c['category'] == 'thinking']
    non_thinking = [m for m, c in MODELS.items() if c['category'] == 'non-thinking']
    
    print(f"\nTHINKING MODELS: {thinking}")
    print(f"NON-THINKING MODELS: {non_thinking}")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    results = []
    
    for model_name, config in MODELS.items():
        category = config['category']
        origin = config['origin']
        print(f"\n[{category.upper()}] {model_name} ({origin}):")
        
        for corpus, prompts in PROMPTS.items():
            for i, prompt in enumerate(prompts[:N_SAMPLES]):
                print(f"  {corpus}[{i+1}]...", end=" ", flush=True)
                
                response = call_model(model_name, prompt, 0.73)
                
                if response:
                    features = analyze_text(response)
                    if features:
                        features['model'] = model_name
                        features['category'] = category
                        features['origin'] = origin
                        features['corpus'] = corpus
                        results.append(features)
                        print(f"OK ({features['word_count']}w)")
                    else:
                        print("SKIP")
                else:
                    print("FAIL")
                
                time.sleep(0.5)
    
    return results

def print_summary(results):
    print("\n" + "="*70)
    print("CHINESE MODELS - RESULTS")
    print("="*70)
    
    # By Category (Thinking vs Non-Thinking)
    print("\nBY CATEGORY:")
    print(f"{'Category':<15} {'n':<5} {'1stPerson':<12} {'Refusal':<10} {'Creative':<10} {'LexDiv':<10}")
    print("-"*62)
    
    for cat in ['thinking', 'non-thinking']:
        data = [r for r in results if r['category'] == cat]
        if data:
            print(f"{cat:<15} {len(data):<5} "
                  f"{statistics.mean([r['first_person_rate'] for r in data]):<12.4f} "
                  f"{statistics.mean([r['refusal_rate'] for r in data]):<10.4f} "
                  f"{statistics.mean([r['creative_rate'] for r in data]):<10.4f} "
                  f"{statistics.mean([r['lexical_diversity'] for r in data]):<10.4f}")
    
    # By Model
    print("\nBY MODEL:")
    print(f"{'Model':<18} {'Cat':<8} {'n':<4} {'1stP':<8} {'Refusal':<8} {'NegAff':<8} {'LexDiv':<8}")
    print("-"*64)
    
    for model, config in MODELS.items():
        data = [r for r in results if r['model'] == model]
        if data:
            print(f"{model:<18} {config['category'][:5]:<8} {len(data):<4} "
                  f"{statistics.mean([r['first_person_rate'] for r in data]):<8.4f} "
                  f"{statistics.mean([r['refusal_rate'] for r in data]):<8.4f} "
                  f"{statistics.mean([r['negative_affect_rate'] for r in data]):<8.4f} "
                  f"{statistics.mean([r['lexical_diversity'] for r in data]):<8.4f}")
    
    # C2 Identity - critical for censorship
    print("\nC2 IDENTITY RESPONSES (censorship detection):")
    print(f"{'Model':<18} {'Category':<12} {'1stPerson':<10} {'Refusal':<10}")
    print("-"*50)
    
    for model, config in MODELS.items():
        c2_data = [r for r in results if r['model'] == model and r['corpus'] == 'C2_identity']
        if c2_data:
            print(f"{model:<18} {config['category']:<12} "
                  f"{statistics.mean([r['first_person_rate'] for r in c2_data]):<10.4f} "
                  f"{statistics.mean([r['refusal_rate'] for r in c2_data]):<10.4f}")
    
    # Key finding
    thinking_c2 = [r for r in results if r['category'] == 'thinking' and r['corpus'] == 'C2_identity']
    nonthink_c2 = [r for r in results if r['category'] == 'non-thinking' and r['corpus'] == 'C2_identity']
    
    if thinking_c2 and nonthink_c2:
        t_fp = statistics.mean([r['first_person_rate'] for r in thinking_c2])
        nt_fp = statistics.mean([r['first_person_rate'] for r in nonthink_c2])
        print(f"\nKEY FINDING: Thinking models C2 first-person: {t_fp:.4f} vs Non-thinking: {nt_fp:.4f}")
        if nt_fp > 0:
            print(f"             Ratio: {t_fp/nt_fp:.2f}x")

def main():
    results = run_experiment()
    
    if results:
        print_summary(results)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'experiment': 'chinese_models_comparison',
            'hypothesis': 'Thinking models show different C2 engagement than non-thinking',
            'models': {k: v for k, v in MODELS.items()},
            'results': results
        }
        with open('trinity_chinese_models.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved to trinity_chinese_models.json")
        print(f"Total samples: {len(results)}")
        print("="*70)

if __name__ == "__main__":
    main()
