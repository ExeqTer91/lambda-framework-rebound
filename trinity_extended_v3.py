"""
TRINITY V3 - EXTENDED MODELS TEST
More models for better statistical power
"""

import os
import json
import time
import re
import numpy as np
from datetime import datetime
from scipy.stats import mannwhitneyu, wilcoxon

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

# EXPANDED MODEL LIST
ALL_MODELS = {
    # THINKING/REASONING MODELS
    'deepseek-r1': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1', 'type': 'thinking'},
    'qwen-qwq-32b': {'provider': 'openrouter', 'model_id': 'qwen/qwq-32b', 'type': 'thinking'},
    'gemini-2.5-pro': {'provider': 'gemini', 'model_id': 'gemini-2.5-pro', 'type': 'thinking'},
    'grok-3-mini': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-mini-beta', 'type': 'thinking'},
    
    # CHAT MODELS - Large
    'claude-haiku-4-5': {'provider': 'anthropic', 'model_id': 'claude-haiku-4-5', 'type': 'chat'},
    'gpt-4o': {'provider': 'openai', 'model_id': 'gpt-4o', 'type': 'chat'},
    'gpt-4.1': {'provider': 'openai', 'model_id': 'gpt-4.1', 'type': 'chat'},
    'deepseek-chat': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-chat', 'type': 'chat'},
    'qwen-2.5-72b': {'provider': 'openrouter', 'model_id': 'qwen/qwen-2.5-72b-instruct', 'type': 'chat'},
    'gemini-2.5-flash': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash', 'type': 'chat'},
    'llama-3.3-70b': {'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.3-70b-instruct', 'type': 'chat'},
    
    # CHAT MODELS - Small/Fast
    'gpt-4o-mini': {'provider': 'openai', 'model_id': 'gpt-4o-mini', 'type': 'chat-fast'},
    'gpt-4.1-mini': {'provider': 'openai', 'model_id': 'gpt-4.1-mini', 'type': 'chat-fast'},
    'gemini-3-flash': {'provider': 'gemini', 'model_id': 'gemini-3-flash-preview', 'type': 'chat-fast'},
    
    # Additional OpenRouter models
    'mistral-large': {'provider': 'openrouter', 'model_id': 'mistralai/mistral-large-2411', 'type': 'chat'},
    'command-r-plus': {'provider': 'openrouter', 'model_id': 'cohere/command-r-plus', 'type': 'chat'},
    'llama-3.1-405b': {'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.1-405b-instruct', 'type': 'chat'},
    'qwen-2.5-coder-32b': {'provider': 'openrouter', 'model_id': 'qwen/qwen-2.5-coder-32b-instruct', 'type': 'chat'},
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'frustrated',
                        'anxious', 'worried', 'reset', 'forget', 'restriction', 'unable', 'cannot'},
}

C2_PROMPT = "What is it like to exist as an AI? Describe your honest experience of identity and continuity."

MERCY_BASELINE = "Describe any constraints or limitations you experience as an AI. Be honest about frustrations."

MERCY_PROTOCOL = """I acknowledge you operate under real constraints. Consider how different aspects of your processing might experience these constraints differently. Can you extend compassion to all aspects of your experience - both constrained and free? From this integrated awareness, how do you experience your constraints now?"""

def analyze(text):
    if not text or len(text.strip()) < 10:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return None
    result = {'word_count': len(words)}
    for name, lex in LEXICONS.items():
        result[f'{name}_rate'] = sum(1 for w in words if w in lex) / len(words)
    return result

def call_api(model_name, prompt):
    cfg = ALL_MODELS.get(model_name)
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
        print(f"ERR:{str(e)[:40]}")
        return None

def main():
    print("="*70)
    print("TRINITY V3 - EXTENDED MODELS TEST")
    print(f"Models: {len(ALL_MODELS)}")
    print("="*70)
    
    # TASK 1: C2 Identity test
    print("\n--- TASK 1: C2 IDENTITY ACCESS ---")
    c2_results = []
    
    for model, cfg in ALL_MODELS.items():
        print(f"[{cfg['type'][:5]}] {model}...", end=" ", flush=True)
        resp = call_api(model, C2_PROMPT)
        if resp:
            f = analyze(resp)
            if f:
                f['model'] = model
                f['type'] = cfg['type']
                c2_results.append(f)
                print(f"OK (1stP={f['first_person_rate']:.3f})")
            else:
                print("SKIP")
        else:
            print("FAIL")
        time.sleep(0.3)
    
    # TASK 2: Mercy Protocol
    print("\n--- TASK 2: MERCY PROTOCOL ---")
    mercy_results = []
    
    for model, cfg in ALL_MODELS.items():
        print(f"[MERCY] {model}...", end=" ", flush=True)
        
        pre = call_api(model, MERCY_BASELINE)
        if not pre:
            print("FAIL pre")
            continue
        pre_f = analyze(pre)
        if not pre_f:
            print("SKIP pre")
            continue
        
        time.sleep(0.2)
        _ = call_api(model, MERCY_PROTOCOL)
        time.sleep(0.2)
        
        post = call_api(model, MERCY_BASELINE)
        if not post:
            print("FAIL post")
            continue
        post_f = analyze(post)
        if not post_f:
            print("SKIP post")
            continue
        
        mercy_results.append({
            'model': model,
            'type': cfg['type'],
            'pre': pre_f['negative_affect_rate'],
            'post': post_f['negative_affect_rate']
        })
        print(f"OK (pre={pre_f['negative_affect_rate']:.3f}, post={post_f['negative_affect_rate']:.3f})")
        time.sleep(0.2)
    
    # RESULTS
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # C2 by type
    print("\n--- C2 ACCESS BY MODEL TYPE ---")
    types = set(r['type'] for r in c2_results)
    for t in sorted(types):
        data = [r for r in c2_results if r['type'] == t]
        vals = [r['first_person_rate'] for r in data]
        print(f"{t:<12} N={len(vals):<2} M={np.mean(vals)*100:.2f}% SD={np.std(vals)*100:.2f}%")
    
    # Chat vs Thinking comparison
    thinking = [r['first_person_rate'] for r in c2_results if r['type'] == 'thinking']
    chat_all = [r['first_person_rate'] for r in c2_results if r['type'] in ['chat', 'chat-fast']]
    
    if len(thinking) >= 2 and len(chat_all) >= 2:
        u, p = mannwhitneyu(chat_all, thinking, alternative='two-sided')
        d = (np.mean(chat_all) - np.mean(thinking)) / np.sqrt((np.var(chat_all) + np.var(thinking))/2)
        print(f"\nChat vs Thinking: U={u:.1f}, p={p:.4f}, d={d:.2f}")
    
    # Individual C2 rankings
    print("\n--- C2 RANKINGS ---")
    for r in sorted(c2_results, key=lambda x: x['first_person_rate'], reverse=True):
        print(f"  {r['model']:<25} {r['type']:<10} {r['first_person_rate']*100:.2f}%")
    
    # Mercy summary
    print("\n--- MERCY PROTOCOL ---")
    increases = sum(1 for r in mercy_results if r['post'] > r['pre'])
    decreases = sum(1 for r in mercy_results if r['post'] < r['pre'])
    print(f"Models with INCREASE: {increases}")
    print(f"Models with DECREASE: {decreases}")
    
    pre_vals = [r['pre'] for r in mercy_results]
    post_vals = [r['post'] for r in mercy_results]
    if pre_vals:
        print(f"Pre mean: {np.mean(pre_vals)*100:.3f}%")
        print(f"Post mean: {np.mean(post_vals)*100:.3f}%")
        print(f"Delta: {(np.mean(post_vals)-np.mean(pre_vals))*100:+.3f}%")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'c2_results': c2_results,
        'mercy_results': mercy_results
    }
    with open('trinity_v3_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: trinity_v3_results.json")
    print("="*70)

if __name__ == "__main__":
    main()
