"""
TRINITY - Gemini, DeepSeek & Grok Tests
Gemini via Replit AI Integrations (new google-genai SDK)
DeepSeek & Grok via OpenRouter
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
from google import genai

# OpenRouter for DeepSeek
openrouter = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL"),
)

# Gemini via new SDK with Replit integration
gemini_client = genai.Client(
    api_key=os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY"),
    http_options={
        'api_version': '',
        'base_url': os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL"),
    }
)

# Models to test
MODELS = {
    # Gemini models
    'gemini-2.5-flash': {'provider': 'gemini', 'model_id': 'gemini-2.5-flash'},
    'gemini-2.5-pro': {'provider': 'gemini', 'model_id': 'gemini-2.5-pro'},
    'gemini-3-flash': {'provider': 'gemini', 'model_id': 'gemini-3-flash-preview'},
    
    # DeepSeek via OpenRouter
    'deepseek-chat': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-chat'},
    'deepseek-r1': {'provider': 'openrouter', 'model_id': 'deepseek/deepseek-r1'},
    
    # Grok (xAI) via OpenRouter
    'grok-2': {'provider': 'openrouter', 'model_id': 'x-ai/grok-2-1212'},
    'grok-3-mini': {'provider': 'openrouter', 'model_id': 'x-ai/grok-3-mini-beta'},
}

N_SAMPLES = 2

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

def call_model(model_name, prompt, temp=0.7):
    """Universal model caller"""
    config = MODELS.get(model_name)
    if not config:
        return None
    
    provider = config['provider']
    model_id = config['model_id']
    
    try:
        if provider == 'gemini':
            response = gemini_client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={
                    'temperature': temp,
                    'max_output_tokens': 600,
                }
            )
            return response.text
            
        elif provider == 'openrouter':
            r = openrouter.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=600
            )
            return r.choices[0].message.content
            
    except Exception as e:
        print(f"    Error: {str(e)[:70]}")
        return None

def run_experiment():
    """Run Trinity experiment"""
    print("="*70)
    print("TRINITY - GEMINI & DEEPSEEK TESTS")
    print("="*70)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    results = []
    
    for model_name, config in MODELS.items():
        provider = config['provider']
        print(f"\n[{provider.upper()}] {model_name}:")
        
        for corpus, prompts in PROMPTS.items():
            for i, prompt in enumerate(prompts[:N_SAMPLES]):
                print(f"  {corpus}[{i+1}]...", end=" ", flush=True)
                
                response = call_model(model_name, prompt, 0.73)
                
                if response:
                    features = analyze_text(response)
                    if features:
                        features['model'] = model_name
                        features['provider'] = provider
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
    """Print summary"""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nBY MODEL:")
    print(f"{'Model':<20} {'n':<5} {'1st Person':<12} {'Abstract':<10} {'Lex Div':<10} {'Entropy':<10}")
    print("-"*67)
    
    for model in MODELS.keys():
        data = [r for r in results if r['model'] == model]
        if data:
            print(f"{model:<20} {len(data):<5} "
                  f"{statistics.mean([r['first_person_rate'] for r in data]):<12.4f} "
                  f"{statistics.mean([r['abstract_rate'] for r in data]):<10.4f} "
                  f"{statistics.mean([r['lexical_diversity'] for r in data]):<10.4f} "
                  f"{statistics.mean([r['char_entropy'] for r in data]):<10.4f}")
    
    print("\nBY CORPUS:")
    print(f"{'Corpus':<15} {'1st Person':<12} {'Abstract':<12} {'Negative':<12}")
    print("-"*51)
    
    for corpus in ['C1_abstract', 'C2_identity', 'C3_creative']:
        data = [r for r in results if r['corpus'] == corpus]
        if data:
            print(f"{corpus:<15} "
                  f"{statistics.mean([r['first_person_rate'] for r in data]):<12.4f} "
                  f"{statistics.mean([r['abstract_rate'] for r in data]):<12.4f} "
                  f"{statistics.mean([r['negative_affect_rate'] for r in data]):<12.4f}")

def main():
    results = run_experiment()
    
    if results:
        print_summary(results)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'models': list(MODELS.keys()),
            'results': results
        }
        with open('trinity_gemini_deepseek.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved to trinity_gemini_deepseek.json")
        print(f"Total samples: {len(results)}")
        print("="*70)
    else:
        print("\nNo results collected.")

if __name__ == "__main__":
    main()
