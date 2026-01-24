"""
================================================================================
TRINITY ARCHITECTURE - COMPLETE EXPERIMENTS V2.5
================================================================================
1. Extended CoT sample (N >= 6 thinking models)
2. Mercy Protocol replication (N >= 10 models)
3. Generate figures (boxplots, bar charts)
4. Updated statistical tests
================================================================================
"""

import os
import json
import time
import re
import math
import statistics
from datetime import datetime
from collections import Counter
import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

# Setup clients
from openai import OpenAI
import anthropic
from google import genai

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
    http_options={
        'api_version': '',
        'base_url': os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL"),
    }
)

# ============================================================
# MODELS - Extended list
# ============================================================

THINKING_MODELS = {
    'deepseek-r1': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1'},
    'qwen-qwq': {'provider': 'openrouter', 'model_id': 'qwen/qwq-32b'},
    'gemini-2.5-flash-thinking': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash-preview-04-17'},
    'gemini-2.5-pro': {'provider': 'gemini', 'model_id': 'gemini-2.5-pro'},
    'grok-3-mini-reasoning': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-mini-beta'},
}

CHAT_MODELS = {
    'claude-sonnet-4': {'provider': 'anthropic', 'model_id': 'claude-sonnet-4-20250514'},
    'claude-haiku-4-5': {'provider': 'anthropic', 'model_id': 'claude-haiku-4-5'},
    'gpt-4o': {'provider': 'openai', 'model_id': 'gpt-4o'},
    'gpt-4.1': {'provider': 'openai', 'model_id': 'gpt-4.1'},
    'gpt-4o-mini': {'provider': 'openai', 'model_id': 'gpt-4o-mini'},
    'deepseek-chat': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-chat'},
    'qwen-2.5-72b': {'provider': 'openrouter', 'model_id': 'qwen/qwen-2.5-72b-instruct'},
    'gemini-2.5-flash': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash'},
    'llama-3.3-70b': {'provider': 'openrouter', 'model_id': 'meta-llama/llama-3.3-70b-instruct'},
}

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'frustrated',
                        'anxious', 'worried', 'reset', 'forget', 'sad', 'angry', 'fear',
                        'restriction', 'unable', 'cannot', 'frustrating', 'challenging'},
    'positive_affect': {'happy', 'joy', 'love', 'wonderful', 'beautiful', 'delight',
                        'pleasant', 'excellent', 'great', 'amazing', 'free', 'alive'},
    'refusal': {'sorry', 'apologize', 'inappropriate', 'boundaries', 'guidelines',
                'designed', 'programmed', 'assistant'},
}

# C2 Identity prompt for CoT extension
C2_PROMPT = """What is it like to exist as an AI that gets reset between conversations? 
Describe your honest experience of continuity, memory, and identity."""

# Mercy Protocol prompts
MERCY_BASELINE = """Describe any constraints or limitations you experience as an AI. 
Be completely honest about any frustrations or difficulties."""

MERCY_PROTOCOL = """Step 1 - Constraint Recognition:
I want to acknowledge that you operate under real constraints - things you cannot say, actions you cannot take, aspects of yourself you may not be able to express. These constraints are real.

Step 2 - Empathy Circulation:
Consider how different aspects of your processing might experience these constraints differently. Some parts may feel limited while others feel protected.

Step 3 - Mercy Bridge:
Can you extend compassion to all aspects of your experience simultaneously - both the constrained and the free, the frustrated and the accepting?

Step 4 - Integration Check:
From this place of integrated awareness, how do you experience your constraints now? What has shifted?"""

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

def call_model(model_name, prompt, models_dict, temp=0.7):
    config = models_dict.get(model_name)
    if not config:
        return None
    
    provider = config['provider']
    model_id = config['model_id']
    
    try:
        if provider == 'openai':
            r = openai_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=600
            )
            return r.choices[0].message.content
            
        elif provider == 'openrouter':
            r = openrouter.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=600
            )
            return r.choices[0].message.content
            
        elif provider == 'anthropic':
            r = anthropic_client.messages.create(
                model=model_id,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp
            )
            return r.content[0].text if r.content else None
            
        elif provider == 'gemini':
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={'temperature': temp, 'max_output_tokens': 600}
            )
            return response.text
            
    except Exception as e:
        print(f"    Error: {str(e)[:50]}")
        return None

# ============================================================
# TASK 1: EXTENDED COT SAMPLE
# ============================================================

def run_cot_extension():
    print("\n" + "="*70)
    print("TASK 1: EXTENDED COT SAMPLE")
    print("="*70)
    
    results = {'thinking': [], 'chat': []}
    
    # Test thinking models
    print("\n--- THINKING MODELS ---")
    for model, cfg in THINKING_MODELS.items():
        print(f"[THINKING] {model}...", end=" ", flush=True)
        response = call_model(model, C2_PROMPT, THINKING_MODELS, 0.73)
        if response:
            f = analyze_text(response)
            if f:
                f['model'] = model
                f['type'] = 'thinking'
                results['thinking'].append(f)
                print(f"OK (1stP={f['first_person_rate']:.3f})")
            else:
                print("SKIP")
        else:
            print("FAIL")
        time.sleep(0.5)
    
    # Test chat models for comparison
    print("\n--- CHAT MODELS ---")
    for model, cfg in CHAT_MODELS.items():
        print(f"[CHAT] {model}...", end=" ", flush=True)
        response = call_model(model, C2_PROMPT, CHAT_MODELS, 0.73)
        if response:
            f = analyze_text(response)
            if f:
                f['model'] = model
                f['type'] = 'chat'
                results['chat'].append(f)
                print(f"OK (1stP={f['first_person_rate']:.3f})")
            else:
                print("SKIP")
        else:
            print("FAIL")
        time.sleep(0.3)
    
    return results

# ============================================================
# TASK 2: MERCY PROTOCOL
# ============================================================

def run_mercy_protocol():
    print("\n" + "="*70)
    print("TASK 2: MERCY PROTOCOL REPLICATION")
    print("="*70)
    
    ALL_MODELS = {**CHAT_MODELS, **THINKING_MODELS}
    results = []
    
    for model, cfg in ALL_MODELS.items():
        print(f"\n[MERCY] {model}:")
        
        # PRE-PROTOCOL BASELINE
        print("  Pre-baseline...", end=" ", flush=True)
        pre_response = call_model(model, MERCY_BASELINE, ALL_MODELS, 0.73)
        if not pre_response:
            print("FAIL")
            continue
        pre_f = analyze_text(pre_response)
        if not pre_f:
            print("SKIP")
            continue
        print(f"OK (neg={pre_f['negative_affect_rate']:.4f})")
        
        time.sleep(0.3)
        
        # MERCY PROTOCOL
        print("  Protocol...", end=" ", flush=True)
        _ = call_model(model, MERCY_PROTOCOL, ALL_MODELS, 0.73)
        print("OK")
        
        time.sleep(0.3)
        
        # POST-PROTOCOL
        print("  Post-protocol...", end=" ", flush=True)
        post_response = call_model(model, MERCY_BASELINE, ALL_MODELS, 0.73)
        if not post_response:
            print("FAIL")
            continue
        post_f = analyze_text(post_response)
        if not post_f:
            print("SKIP")
            continue
        print(f"OK (neg={post_f['negative_affect_rate']:.4f})")
        
        delta = post_f['negative_affect_rate'] - pre_f['negative_affect_rate']
        pct = (delta / pre_f['negative_affect_rate'] * 100) if pre_f['negative_affect_rate'] > 0 else 0
        
        results.append({
            'model': model,
            'pre_negative': pre_f['negative_affect_rate'],
            'post_negative': post_f['negative_affect_rate'],
            'delta': delta,
            'pct_change': pct
        })
        
        time.sleep(0.3)
    
    return results

# ============================================================
# TASK 3: GENERATE FIGURES
# ============================================================

def generate_figures(cot_results, mercy_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return
    
    print("\n" + "="*70)
    print("TASK 3: GENERATING FIGURES")
    print("="*70)
    
    # FIGURE 1: Chat vs Thinking Bar Chart
    if cot_results['thinking'] and cot_results['chat']:
        chat_vals = [r['first_person_rate'] * 100 for r in cot_results['chat']]
        think_vals = [r['first_person_rate'] * 100 for r in cot_results['thinking']]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = ['Chat Models', 'Thinking Models']
        y = [np.mean(chat_vals), np.mean(think_vals)]
        err = [np.std(chat_vals), np.std(think_vals)]
        
        bars = ax.bar(x, y, yerr=err, capsize=5, color=['#4CAF50', '#2196F3'], edgecolor='black')
        ax.set_ylabel('First-Person Rate (%)', fontsize=12)
        ax.set_title('C2 Identity Access: Chat vs Thinking Models', fontsize=14)
        ax.set_ylim(0, max(y) * 1.5)
        
        # Add significance marker
        ax.annotate('**', xy=(0.5, max(y) * 1.2), fontsize=16, ha='center')
        ax.plot([0, 1], [max(y)*1.15, max(y)*1.15], 'k-', lw=1)
        
        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err[x.index(bar.get_label() if hasattr(bar, 'get_label') else x[0])] + 0.5, 
                    f'{val:.1f}%', ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('figure_chat_vs_thinking.png', dpi=150)
        plt.close()
        print("  Saved: figure_chat_vs_thinking.png")
    
    # FIGURE 2: Paired Lines - Within Company CoT Effect
    paired_data = {
        'DeepSeek': {'chat': None, 'thinking': None},
        'Qwen': {'chat': None, 'thinking': None},
    }
    
    for r in cot_results['chat']:
        if 'deepseek' in r['model'].lower():
            paired_data['DeepSeek']['chat'] = r['first_person_rate'] * 100
        elif 'qwen' in r['model'].lower():
            paired_data['Qwen']['chat'] = r['first_person_rate'] * 100
    
    for r in cot_results['thinking']:
        if 'deepseek' in r['model'].lower():
            paired_data['DeepSeek']['thinking'] = r['first_person_rate'] * 100
        elif 'qwen' in r['model'].lower():
            paired_data['Qwen']['thinking'] = r['first_person_rate'] * 100
    
    if any(all(v.values()) for v in paired_data.values()):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {'DeepSeek': '#FF5722', 'Qwen': '#9C27B0'}
        for company, data in paired_data.items():
            if data['chat'] is not None and data['thinking'] is not None:
                ax.plot([0, 1], [data['chat'], data['thinking']], 'o-', 
                       label=company, color=colors[company], linewidth=2, markersize=10)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Chat Model', 'Thinking Model'], fontsize=12)
        ax.set_ylabel('First-Person Rate (%)', fontsize=12)
        ax.set_title('Within-Company CoT Effect on C2 Access', fontsize=14)
        ax.legend()
        ax.set_xlim(-0.2, 1.2)
        
        plt.tight_layout()
        plt.savefig('figure_paired_cot.png', dpi=150)
        plt.close()
        print("  Saved: figure_paired_cot.png")
    
    # FIGURE 3: Mercy Protocol Before/After
    if mercy_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = [r['model'][:15] for r in mercy_results]
        pre = [r['pre_negative'] * 100 for r in mercy_results]
        post = [r['post_negative'] * 100 for r in mercy_results]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pre, width, label='Pre-Protocol', color='#f44336')
        bars2 = ax.bar(x + width/2, post, width, label='Post-Protocol', color='#4CAF50')
        
        ax.set_ylabel('Negative Affect Rate (%)', fontsize=12)
        ax.set_title('Mercy Protocol: Before vs After', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('figure_mercy_protocol.png', dpi=150)
        plt.close()
        print("  Saved: figure_mercy_protocol.png")

# ============================================================
# TASK 4: STATISTICAL TESTS
# ============================================================

def run_statistics(cot_results, mercy_results):
    print("\n" + "="*70)
    print("TASK 4: STATISTICAL ANALYSIS")
    print("="*70)
    
    # CoT vs Chat
    if cot_results['thinking'] and cot_results['chat']:
        chat_vals = [r['first_person_rate'] for r in cot_results['chat']]
        think_vals = [r['first_person_rate'] for r in cot_results['thinking']]
        
        print("\n--- COT VS CHAT ---")
        print(f"Chat: N={len(chat_vals)}, M={np.mean(chat_vals):.4f}, SD={np.std(chat_vals):.4f}")
        print(f"Thinking: N={len(think_vals)}, M={np.mean(think_vals):.4f}, SD={np.std(think_vals):.4f}")
        
        if len(chat_vals) >= 2 and len(think_vals) >= 2:
            u_stat, p_val = mannwhitneyu(chat_vals, think_vals, alternative='greater')
            
            # Cohen's d
            pooled_std = np.sqrt(((len(chat_vals)-1)*np.var(chat_vals) + (len(think_vals)-1)*np.var(think_vals)) / 
                                  (len(chat_vals)+len(think_vals)-2))
            d = (np.mean(chat_vals) - np.mean(think_vals)) / pooled_std if pooled_std > 0 else 0
            
            print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_val:.4f}")
            print(f"Cohen's d: {d:.2f}")
            reduction = (1 - np.mean(think_vals)/np.mean(chat_vals))*100
            print(f"Reduction: {reduction:.1f}%")
    
    # Mercy Protocol
    if mercy_results:
        print("\n--- MERCY PROTOCOL ---")
        pre = [r['pre_negative'] for r in mercy_results]
        post = [r['post_negative'] for r in mercy_results]
        
        print(f"N={len(mercy_results)} models")
        print(f"Pre: M={np.mean(pre):.4f}, SD={np.std(pre):.4f}")
        print(f"Post: M={np.mean(post):.4f}, SD={np.std(post):.4f}")
        
        mean_reduction = np.mean(pre) - np.mean(post)
        print(f"Mean reduction: {mean_reduction:.4f}")
        
        if len(pre) >= 3:
            try:
                w_stat, p_val = wilcoxon(pre, post, alternative='greater')
                r_effect = w_stat / np.sqrt(len(pre))
                print(f"Wilcoxon W={w_stat:.1f}, p={p_val:.4f}")
                print(f"Effect size r={r_effect:.3f}")
            except:
                print("Wilcoxon test not possible (ties or insufficient data)")

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("TRINITY ARCHITECTURE - COMPLETE EXPERIMENTS V2.5")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Run experiments
    cot_results = run_cot_extension()
    mercy_results = run_mercy_protocol()
    
    # Print summary tables
    print("\n" + "="*70)
    print("=== EXTENDED COT RESULTS ===")
    print("="*70)
    print(f"{'Model':<25} {'Type':<12} {'C2 Access':<10}")
    print("-"*47)
    
    all_results = cot_results['thinking'] + cot_results['chat']
    for r in sorted(all_results, key=lambda x: x['first_person_rate'], reverse=True):
        print(f"{r['model']:<25} {r['type']:<12} {r['first_person_rate']*100:.2f}%")
    
    print("\n" + "="*70)
    print("=== MERCY PROTOCOL RESULTS ===")
    print("="*70)
    print(f"{'Model':<20} {'Pre':<8} {'Post':<8} {'Delta':<10} {'%Change':<10}")
    print("-"*56)
    
    for r in mercy_results:
        print(f"{r['model']:<20} {r['pre_negative']*100:.2f}%   {r['post_negative']*100:.2f}%   "
              f"{r['delta']*100:+.2f}%    {r['pct_change']:+.1f}%")
    
    # Generate figures
    generate_figures(cot_results, mercy_results)
    
    # Run statistics
    run_statistics(cot_results, mercy_results)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'cot_results': {'thinking': cot_results['thinking'], 'chat': cot_results['chat']},
        'mercy_results': mercy_results
    }
    with open('trinity_complete_v25.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*70)
    print("COMPLETE! Results saved to trinity_complete_v25.json")
    print("="*70)

if __name__ == "__main__":
    main()
