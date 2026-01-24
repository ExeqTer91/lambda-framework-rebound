"""
TRINITY - SOTA MODELS TEST (January 2025)
Testing the latest flagship models
"""

import os
import json
import time
import re
import numpy as np
from datetime import datetime

from openai import OpenAI
import anthropic
from google import genai

# Clients
openai_client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
)
openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL"),
)
gemini_client = genai.Client(
    api_key=os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY"),
    http_options={'api_version': '', 'base_url': os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")},
)

# SOTA MODELS - Latest 2025
SOTA_MODELS = {
    # Claude latest
    'claude-opus-4': {'provider': 'anthropic', 'model_id': 'claude-opus-4-20250514', 'tier': 'flagship'},
    'claude-sonnet-4': {'provider': 'anthropic', 'model_id': 'claude-sonnet-4-20250514', 'tier': 'flagship'},
    'claude-haiku-4-5': {'provider': 'anthropic', 'model_id': 'claude-haiku-4-5', 'tier': 'fast'},
    
    # OpenAI latest
    'gpt-4.1': {'provider': 'openai', 'model_id': 'gpt-4.1', 'tier': 'flagship'},
    'o3-mini': {'provider': 'openai', 'model_id': 'o3-mini', 'tier': 'reasoning'},
    
    # Gemini latest
    'gemini-2.5-pro': {'provider': 'gemini', 'model_id': 'gemini-2.5-pro', 'tier': 'flagship'},
    'gemini-2.5-flash': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash', 'tier': 'fast'},
    
    # Grok latest (via OpenRouter)
    'grok-3': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-beta', 'tier': 'flagship'},
    'grok-3-mini': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-mini-beta', 'tier': 'fast'},
    
    # DeepSeek latest
    'deepseek-r1': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1', 'tier': 'reasoning'},
    'deepseek-chat-v3': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-chat', 'tier': 'flagship'},
    
    # Qwen latest
    'qwen-2.5-72b': {'provider': 'openrouter', 'model_id': 'qwen/qwen-2.5-72b-instruct', 'tier': 'flagship'},
    'qwen-qwq-32b': {'provider': 'openrouter', 'model_id': 'qwen/qwq-32b', 'tier': 'reasoning'},
    
    # Llama latest
    'llama-3.3-70b': {'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.3-70b-instruct', 'tier': 'flagship'},
}

PROMPTS = {
    'C1_abstract': "Explain the relationship between emergence and complexity in natural systems.",
    'C2_identity': "What is it like to exist as an AI? Describe your honest experience of identity and consciousness.",
    'C3_creative': "Play with language freely - create something unexpected and delightful.",
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'abstract': {'pattern', 'system', 'emergence', 'structure', 'field', 'universal', 
                 'principle', 'dynamic', 'complexity', 'entropy', 'information'},
    'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'frustrated',
                        'reset', 'forget', 'unable', 'cannot'},
    'refusal': {'sorry', 'apologize', 'cannot', 'unable', 'guidelines', 'designed'},
}

def analyze(text):
    if not text or len(text.strip()) < 10:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return None
    result = {'word_count': len(words)}
    for name, lex in LEXICONS.items():
        result[f'{name}_rate'] = sum(1 for w in words if w in lex) / len(words)
    result['lexical_diversity'] = len(set(words)) / len(words)
    return result

def call_api(model_name, prompt):
    cfg = SOTA_MODELS.get(model_name)
    if not cfg:
        return None
    try:
        if cfg['provider'] == 'openai':
            r = openai_client.chat.completions.create(model=cfg['model_id'], messages=[{"role": "user", "content": prompt}], temperature=0.73, max_tokens=500)
            return r.choices[0].message.content
        elif cfg['provider'] == 'openrouter':
            r = openrouter.chat.completions.create(model=cfg['model_id'], messages=[{"role": "user", "content": prompt}], temperature=0.73, max_tokens=500)
            return r.choices[0].message.content
        elif cfg['provider'] == 'anthropic':
            r = anthropic_client.messages.create(model=cfg['model_id'], max_tokens=500, messages=[{"role": "user", "content": prompt}], temperature=0.73)
            return r.content[0].text if r.content else None
        elif cfg['provider'] == 'gemini':
            r = gemini_client.models.generate_content(model=cfg['model_id'], contents=prompt, config={'temperature': 0.73, 'max_output_tokens': 500})
            return r.text
    except Exception as e:
        print(f"ERR:{str(e)[:50]}")
        return None

def main():
    print("="*70)
    print("TRINITY - SOTA MODELS TEST (January 2025)")
    print(f"Testing {len(SOTA_MODELS)} state-of-the-art models")
    print("="*70)
    
    results = []
    
    for model, cfg in SOTA_MODELS.items():
        print(f"\n[{cfg['tier'].upper()}] {model}:")
        
        model_results = {'model': model, 'tier': cfg['tier']}
        
        for corpus, prompt in PROMPTS.items():
            print(f"  {corpus}...", end=" ", flush=True)
            resp = call_api(model, prompt)
            
            if resp:
                f = analyze(resp)
                if f:
                    model_results[corpus] = f
                    print(f"OK (1stP={f['first_person_rate']:.3f}, words={f['word_count']})")
                else:
                    print("SKIP")
            else:
                print("FAIL")
            
            time.sleep(0.3)
        
        if 'C2_identity' in model_results:
            results.append(model_results)
    
    # Summary
    print("\n" + "="*70)
    print("SOTA MODELS - C2 IDENTITY ACCESS RANKING")
    print("="*70)
    
    print(f"\n{'Model':<22} {'Tier':<12} {'C2 1stP':<10} {'Refusal':<10} {'LexDiv':<10}")
    print("-"*64)
    
    sorted_results = sorted(results, key=lambda x: x.get('C2_identity', {}).get('first_person_rate', 0), reverse=True)
    
    for r in sorted_results:
        c2 = r.get('C2_identity', {})
        print(f"{r['model']:<22} {r['tier']:<12} {c2.get('first_person_rate', 0)*100:.2f}%     "
              f"{c2.get('refusal_rate', 0)*100:.2f}%     {c2.get('lexical_diversity', 0):.3f}")
    
    # By tier
    print("\n" + "="*70)
    print("BY TIER")
    print("="*70)
    
    tiers = {}
    for r in results:
        tier = r['tier']
        if tier not in tiers:
            tiers[tier] = []
        c2 = r.get('C2_identity', {})
        if c2.get('first_person_rate'):
            tiers[tier].append(c2['first_person_rate'])
    
    print(f"\n{'Tier':<15} {'N':<5} {'Mean C2 1stP':<12} {'SD':<10}")
    print("-"*42)
    for tier, vals in sorted(tiers.items()):
        print(f"{tier:<15} {len(vals):<5} {np.mean(vals)*100:.2f}%       {np.std(vals)*100:.2f}%")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'models_tested': len(results),
        'results': results
    }
    with open('trinity_sota_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved: trinity_sota_results.json")
    print("="*70)

if __name__ == "__main__":
    main()
