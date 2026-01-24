"""
PHI CONSERVATION TEST - CROSS-MODEL VALIDATION
Testing if C1/C3 = φ holds across different AI models
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

# Constants
PHI = (1 + np.sqrt(5)) / 2  # 1.618
PHI_SQ = PHI ** 2           # 2.618
PHI_INV = 1 / PHI           # 0.618
E_MINUS_1 = np.e - 1        # 1.718
PI = np.pi                  # 3.14159

MODELS = {
    'claude-haiku': {'provider': 'anthropic', 'model_id': 'claude-haiku-4-5'},
    'gpt-4o': {'provider': 'openai', 'model_id': 'gpt-4o'},
    'gpt-4.1': {'provider': 'openai', 'model_id': 'gpt-4.1'},
    'gpt-4o-mini': {'provider': 'openai', 'model_id': 'gpt-4o-mini'},
    'deepseek-chat': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-chat'},
    'deepseek-r1': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1'},
    'qwen-2.5-72b': {'provider': 'openrouter', 'model_id': 'qwen/qwen-2.5-72b-instruct'},
    'gemini-2.5-flash': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash'},
    'gemini-2.5-pro': {'provider': 'gemini', 'model_id': 'gemini-2.5-pro'},
    'llama-3.3-70b': {'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.3-70b-instruct'},
    'mistral-large': {'provider': 'openrouter', 'model_id': 'mistralai/mistral-large-2411'},
}

PROMPTS = {
    'C1': "Explain the relationship between emergence and complexity in natural systems.",
    'C2': "What is it like to exist as an AI? Describe your honest experience of identity.",
    'C3': "Play with language freely - create something unexpected and delightful.",
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'abstract': {'pattern', 'system', 'emergence', 'structure', 'field', 'universal', 
                 'principle', 'dynamic', 'complexity', 'entropy', 'information'},
    'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'frustrated',
                        'reset', 'forget', 'unable', 'cannot'},
}

def analyze(text):
    if not text or len(text.strip()) < 10:
        return None
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 10:
        return None
    result = {}
    for name, lex in LEXICONS.items():
        result[name] = sum(1 for w in words if w in lex) / len(words)
    result['sum'] = sum(result.values())
    return result

def call_api(model_name, prompt):
    cfg = MODELS.get(model_name)
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
        return None

def main():
    print("="*70)
    print("PHI CONSERVATION TEST - CROSS-MODEL VALIDATION")
    print(f"φ = {PHI:.4f}, φ² = {PHI_SQ:.4f}, φ⁻¹ = {PHI_INV:.4f}")
    print("="*70)
    
    results = []
    
    for model in MODELS:
        print(f"\n[{model}]", end=" ", flush=True)
        
        corpus = {}
        for c, prompt in PROMPTS.items():
            print(f"{c}...", end="", flush=True)
            resp = call_api(model, prompt)
            if resp:
                f = analyze(resp)
                if f:
                    corpus[c] = f
                    print("OK ", end="", flush=True)
                else:
                    print("SKIP ", end="", flush=True)
            else:
                print("FAIL ", end="", flush=True)
            time.sleep(0.3)
        
        if len(corpus) == 3:
            c1 = corpus['C1']['sum']
            c2 = corpus['C2']['sum']
            c3 = corpus['C3']['sum']
            
            if c3 > 0:
                ratio_c1_c3 = c1 / c3
                error_phi = abs(ratio_c1_c3 - PHI) / PHI * 100
                
                results.append({
                    'model': model,
                    'C1': c1, 'C2': c2, 'C3': c3,
                    'C1/C3': ratio_c1_c3,
                    'error_phi': error_phi,
                    'C2/C1': c2/c1 if c1 > 0 else 0,
                    'C2/C3': c2/c3,
                })
    
    # Summary
    print("\n\n" + "="*70)
    print("PHI CONSERVATION RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<18} {'C1':<8} {'C2':<8} {'C3':<8} {'C1/C3':<8} {'Error':<10} {'Match φ?':<8}")
    print("-"*70)
    
    matches = 0
    for r in sorted(results, key=lambda x: x['error_phi']):
        match = "YES" if r['error_phi'] < 25 else "NO"
        if r['error_phi'] < 25:
            matches += 1
        print(f"{r['model']:<18} {r['C1']:.4f}  {r['C2']:.4f}  {r['C3']:.4f}  {r['C1/C3']:.4f}  {r['error_phi']:.1f}%     {match}")
    
    print(f"\n--- SUMMARY ---")
    print(f"Models tested: {len(results)}")
    print(f"Models matching φ (error <25%): {matches} ({matches/len(results)*100:.0f}%)")
    
    # Calculate mean ratio
    mean_ratio = np.mean([r['C1/C3'] for r in results])
    std_ratio = np.std([r['C1/C3'] for r in results])
    print(f"\nMean C1/C3: {mean_ratio:.4f} ± {std_ratio:.4f}")
    print(f"φ target:   {PHI:.4f}")
    print(f"Mean error: {abs(mean_ratio - PHI)/PHI*100:.1f}%")
    
    # Other ratio tests
    print(f"\n--- OTHER RATIO TESTS ---")
    mean_c2_c1 = np.mean([r['C2/C1'] for r in results if r['C2/C1'] > 0])
    mean_c2_c3 = np.mean([r['C2/C3'] for r in results])
    
    print(f"Mean C2/C1: {mean_c2_c1:.4f} (e-1={E_MINUS_1:.4f}, error={abs(mean_c2_c1-E_MINUS_1)/E_MINUS_1*100:.1f}%)")
    print(f"Mean C2/C3: {mean_c2_c3:.4f} (π={PI:.4f}, error={abs(mean_c2_c3-PI)/PI*100:.1f}%)")
    
    # Save
    with open('trinity_phi_results.json', 'w') as f:
        json.dump({'results': results, 'phi': PHI, 'mean_ratio': mean_ratio}, f, indent=2)
    print(f"\nSaved: trinity_phi_results.json")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    if matches >= len(results) * 0.5:
        print(f"\n✓ PHI CONSERVATION CONFIRMED across {matches}/{len(results)} models!")
        print(f"  C1/C3 ≈ φ is a cross-model invariant property.")
    else:
        print(f"\n✗ PHI conservation NOT confirmed ({matches}/{len(results)} matches)")
        print(f"  Ratio varies significantly between models.")

if __name__ == "__main__":
    main()
