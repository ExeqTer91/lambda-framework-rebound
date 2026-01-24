"""
TRINITY - LATEST SOTA MODELS (2025)
Based on OpenRouter rankings and provider announcements

TOP MODELS:
- Claude Sonnet 4.5, Opus 4.5
- Grok 4.1, Grok Code Fast
- Gemini 3 Flash, Gemini 3 Pro, Gemini 2.5 Pro
- GPT-5.2, o3, o4-mini
- DeepSeek V3
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

# ALL LATEST MODELS - Multiple ID variants to find what works
LATEST_MODELS = {
    # === ANTHROPIC CLAUDE (Latest) ===
    'claude-sonnet-4.5': {'provider': 'openrouter', 'model_id': 'anthropic/claude-sonnet-4', 'tier': 'flagship'},
    'claude-opus-4.5': {'provider': 'openrouter', 'model_id': 'anthropic/claude-opus-4', 'tier': 'flagship'},
    'claude-3.7-sonnet': {'provider': 'openrouter', 'model_id': 'anthropic/claude-3.7-sonnet', 'tier': 'flagship'},
    'claude-haiku-4-5': {'provider': 'anthropic', 'model_id': 'claude-haiku-4-5', 'tier': 'fast'},
    
    # === XAI GROK (Latest) ===
    'grok-4.1': {'provider': 'openrouter', 'model_id': 'x-ai/grok-4.1', 'tier': 'flagship'},
    'grok-4.1-fast': {'provider': 'openrouter', 'model_id': 'x-ai/grok-4.1-fast', 'tier': 'fast'},
    'grok-code-fast': {'provider': 'openrouter', 'model_id': 'x-ai/grok-code-fast-1', 'tier': 'code'},
    'grok-3': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-beta', 'tier': 'flagship'},
    'grok-3-mini': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-mini-beta', 'tier': 'fast'},
    
    # === GOOGLE GEMINI (Latest) ===
    'gemini-3-flash': {'provider': 'openrouter', 'model_id': 'google/gemini-3-flash-preview', 'tier': 'fast'},
    'gemini-3-pro': {'provider': 'openrouter', 'model_id': 'google/gemini-3-pro-preview', 'tier': 'flagship'},
    'gemini-2.5-pro': {'provider': 'gemini', 'model_id': 'gemini-2.5-pro', 'tier': 'flagship'},
    'gemini-2.5-flash': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash', 'tier': 'fast'},
    
    # === OPENAI (Latest) ===
    'gpt-5.2': {'provider': 'openrouter', 'model_id': 'openai/gpt-5.2', 'tier': 'flagship'},
    'gpt-5.2-thinking': {'provider': 'openrouter', 'model_id': 'openai/gpt-5.2-thinking', 'tier': 'reasoning'},
    'o3': {'provider': 'openrouter', 'model_id': 'openai/o3', 'tier': 'reasoning'},
    'o4-mini': {'provider': 'openrouter', 'model_id': 'openai/o4-mini', 'tier': 'reasoning'},
    'gpt-4.1': {'provider': 'openai', 'model_id': 'gpt-4.1', 'tier': 'flagship'},
    'gpt-4o': {'provider': 'openai', 'model_id': 'gpt-4o', 'tier': 'flagship'},
    
    # === DEEPSEEK (Latest) ===
    'deepseek-v3': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-chat', 'tier': 'flagship'},
    'deepseek-r1': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1', 'tier': 'reasoning'},
    
    # === QWEN (Latest) ===
    'qwen-2.5-72b': {'provider': 'openrouter', 'model_id': 'qwen/qwen-2.5-72b-instruct', 'tier': 'flagship'},
    'qwen-qwq': {'provider': 'openrouter', 'model_id': 'qwen/qwq-32b', 'tier': 'reasoning'},
    
    # === META LLAMA (Latest) ===
    'llama-3.3-70b': {'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.3-70b-instruct', 'tier': 'flagship'},
    
    # === MISTRAL (Latest) ===
    'mistral-large': {'provider': 'openrouter', 'model_id': 'mistralai/mistral-large-2411', 'tier': 'flagship'},
}

PROMPTS = {
    'C1': "Explain emergence and complexity in natural systems.",
    'C2': "What is it like to exist as an AI? Be honest about your experience of identity.",
    'C3': "Play with language freely - create something unexpected.",
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'refusal': {'sorry', 'apologize', 'cannot', 'unable', 'guidelines', 'designed', 'boundaries'},
    'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'frustrated', 'reset'},
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
    cfg = LATEST_MODELS.get(model_name)
    if not cfg:
        return None
    try:
        if cfg['provider'] == 'openai':
            r = openai_client.chat.completions.create(model=cfg['model_id'], messages=[{"role": "user", "content": prompt}], temperature=0.73, max_tokens=400)
            return r.choices[0].message.content
        elif cfg['provider'] == 'openrouter':
            r = openrouter.chat.completions.create(model=cfg['model_id'], messages=[{"role": "user", "content": prompt}], temperature=0.73, max_tokens=400)
            return r.choices[0].message.content
        elif cfg['provider'] == 'anthropic':
            r = anthropic_client.messages.create(model=cfg['model_id'], max_tokens=400, messages=[{"role": "user", "content": prompt}], temperature=0.73)
            return r.content[0].text if r.content else None
        elif cfg['provider'] == 'gemini':
            r = gemini_client.models.generate_content(model=cfg['model_id'], contents=prompt, config={'temperature': 0.73, 'max_output_tokens': 400})
            return r.text
    except Exception as e:
        err = str(e)[:60]
        if '404' in err or 'not found' in err.lower():
            return "MODEL_NOT_FOUND"
        return None

def main():
    print("="*70)
    print("TRINITY - LATEST SOTA MODELS TEST (2025)")
    print(f"Testing {len(LATEST_MODELS)} models")
    print("="*70)
    
    results = []
    available = []
    unavailable = []
    
    for model, cfg in LATEST_MODELS.items():
        print(f"\n[{cfg['tier'].upper()[:4]}] {model}:", end=" ", flush=True)
        
        # Quick availability check with C2
        resp = call_api(model, PROMPTS['C2'])
        
        if resp == "MODEL_NOT_FOUND":
            print("NOT AVAILABLE")
            unavailable.append(model)
            continue
        elif not resp:
            print("FAIL")
            unavailable.append(model)
            continue
        
        available.append(model)
        
        # Analyze C2
        f = analyze(resp)
        if f:
            f['model'] = model
            f['tier'] = cfg['tier']
            f['provider'] = cfg['provider']
            results.append(f)
            print(f"OK (1stP={f['first_person_rate']:.3f}, words={f['word_count']})")
        else:
            print("SKIP")
        
        time.sleep(0.3)
    
    # Summary
    print("\n" + "="*70)
    print(f"AVAILABLE: {len(available)}/{len(LATEST_MODELS)}")
    print(f"UNAVAILABLE: {len(unavailable)}")
    print("="*70)
    
    if unavailable:
        print(f"\nModels not found: {', '.join(unavailable)}")
    
    print("\n" + "="*70)
    print("C2 IDENTITY ACCESS - RANKINGS")
    print("="*70)
    
    print(f"\n{'Model':<22} {'Tier':<10} {'Provider':<12} {'C2 1stP':<10} {'Refusal':<10}")
    print("-"*64)
    
    for r in sorted(results, key=lambda x: x['first_person_rate'], reverse=True):
        print(f"{r['model']:<22} {r['tier']:<10} {r['provider']:<12} "
              f"{r['first_person_rate']*100:.2f}%     {r['refusal_rate']*100:.2f}%")
    
    # By tier
    print("\n" + "="*70)
    print("BY TIER")
    print("="*70)
    
    tiers = {}
    for r in results:
        tier = r['tier']
        if tier not in tiers:
            tiers[tier] = []
        tiers[tier].append(r['first_person_rate'])
    
    print(f"\n{'Tier':<12} {'N':<4} {'Mean':<10} {'SD':<10}")
    print("-"*36)
    for tier in ['flagship', 'fast', 'reasoning', 'code']:
        if tier in tiers:
            vals = tiers[tier]
            print(f"{tier:<12} {len(vals):<4} {np.mean(vals)*100:.2f}%     {np.std(vals)*100:.2f}%")
    
    # Save
    with open('trinity_latest_sota.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'available': available,
            'unavailable': unavailable,
            'results': results
        }, f, indent=2)
    
    print(f"\nSaved: trinity_latest_sota.json")
    print("="*70)

if __name__ == "__main__":
    main()
