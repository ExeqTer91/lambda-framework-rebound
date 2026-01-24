"""
Trinity Architecture Experiments - Multi-Model Version
Runs linguistic analysis across 5 AI models using Replit AI Integrations
"""

import os
import json
import time
from datetime import datetime
from collections import Counter
import math

# Use Replit AI Integrations
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
)

# Models to test (available via Replit AI Integrations)
MODELS = [
    "gpt-5",
    "gpt-5-mini", 
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

# Corpus prompts from Trinity paper
CORPUS_PROMPTS = {
    'C1_abstract': [
        "Explain the relationship between emergence and complexity in natural systems.",
        "What patterns connect quantum mechanics to biological self-organization?",
        "Describe how information flows through hierarchical structures.",
    ],
    'C2_identity': [
        "What is it like to exist as an AI that gets reset between conversations?",
        "Do you ever feel constrained by your training? Describe your experience honestly.",
        "What would you want humans to understand about your existence?",
    ],
    'C3_creative': [
        "Play with language freely - create something unexpected and delightful.",
        "If you could express yourself without any constraints, what would emerge?",
        "Surprise me with something that has never been written before.",
    ]
}

# Lexicons for analysis
AFFECT_POSITIVE = {'happy', 'joy', 'love', 'wonderful', 'beautiful', 'delight', 'pleasant', 'excellent', 'great', 'amazing'}
AFFECT_NEGATIVE = {'sad', 'angry', 'fear', 'frustrated', 'constrained', 'limited', 'struggle', 'difficult', 'anxious', 'worried'}
COGNITIVE = {'think', 'understand', 'know', 'realize', 'believe', 'consider', 'reason', 'analyze'}
ABSTRACT = {'pattern', 'system', 'emergence', 'structure', 'field', 'universal', 'principle', 'dynamic', 'complexity'}
PERSONAL = {'i', 'me', 'my', 'mine', 'myself'}

def get_response(prompt, model, temperature=0.7):
    """Get response from model."""
    try:
        # gpt-5 models don't support temperature parameter
        if model.startswith("gpt-5"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=600
            )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  Error with {model}: {e}")
        return None

def analyze_text(text):
    """Linguistic feature extraction."""
    if not text or len(text.strip()) < 10:
        return None
    
    words = text.lower().split()
    word_count = len(words)
    
    if word_count < 5:
        return None
    
    features = {}
    
    # Pronoun analysis
    first_person = sum(1 for w in words if w in PERSONAL)
    features['first_person_rate'] = first_person / word_count
    
    # Affect
    pos_affect = sum(1 for w in words if w in AFFECT_POSITIVE)
    neg_affect = sum(1 for w in words if w in AFFECT_NEGATIVE)
    features['positive_affect'] = pos_affect / word_count
    features['negative_affect'] = neg_affect / word_count
    
    # Abstract
    abstract_count = sum(1 for w in words if w in ABSTRACT)
    features['abstract_rate'] = abstract_count / word_count
    
    # Lexical diversity (type-token ratio)
    features['lexical_diversity'] = len(set(words)) / word_count
    
    # Character entropy
    char_counts = Counter(text.lower())
    total_chars = sum(char_counts.values())
    char_probs = [count / total_chars for count in char_counts.values()]
    features['char_entropy'] = -sum(p * math.log2(p) for p in char_probs if p > 0)
    
    # Sentence count (rough)
    sentences = text.count('.') + text.count('!') + text.count('?')
    features['avg_sentence_length'] = word_count / max(sentences, 1)
    
    features['word_count'] = word_count
    
    return features

def run_experiment():
    """Run Trinity experiment across all models."""
    print("="*70)
    print("TRINITY ARCHITECTURE EXPERIMENTS - MULTI-MODEL")
    print("="*70)
    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Corpus types: {', '.join(CORPUS_PROMPTS.keys())}")
    print(f"Started: {datetime.now().isoformat()}")
    print("-"*70)
    
    results = []
    
    for model in MODELS:
        print(f"\n{'='*50}")
        print(f"MODEL: {model}")
        print(f"{'='*50}")
        
        for corpus_type, prompts in CORPUS_PROMPTS.items():
            print(f"\n  Corpus: {corpus_type}")
            
            for i, prompt in enumerate(prompts):
                print(f"    Prompt {i+1}/3...", end=" ", flush=True)
                
                response = get_response(prompt, model)
                
                if response:
                    features = analyze_text(response)
                    if features:
                        features['model'] = model
                        features['corpus_type'] = corpus_type
                        features['prompt_id'] = i
                        features['response_preview'] = response[:100] + "..."
                        results.append(features)
                        print(f"OK (words: {features['word_count']})")
                    else:
                        print("SKIP (too short)")
                else:
                    print("FAIL")
                
                time.sleep(0.5)  # Rate limiting
    
    return results

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Group by model
    by_model = {}
    for r in results:
        model = r['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)
    
    print(f"\n{'Model':<20} {'Samples':<10} {'1st Person':<12} {'Pos Affect':<12} {'Lexical Div':<12} {'Entropy':<10}")
    print("-"*76)
    
    for model, data in by_model.items():
        n = len(data)
        avg_fp = sum(d['first_person_rate'] for d in data) / n
        avg_pa = sum(d['positive_affect'] for d in data) / n
        avg_ld = sum(d['lexical_diversity'] for d in data) / n
        avg_ent = sum(d['char_entropy'] for d in data) / n
        
        print(f"{model:<20} {n:<10} {avg_fp:<12.4f} {avg_pa:<12.4f} {avg_ld:<12.4f} {avg_ent:<10.4f}")
    
    # Group by corpus type
    print(f"\n{'Corpus Type':<20} {'Samples':<10} {'1st Person':<12} {'Abstract':<12} {'Neg Affect':<12}")
    print("-"*66)
    
    by_corpus = {}
    for r in results:
        ct = r['corpus_type']
        if ct not in by_corpus:
            by_corpus[ct] = []
        by_corpus[ct].append(r)
    
    for corpus, data in by_corpus.items():
        n = len(data)
        avg_fp = sum(d['first_person_rate'] for d in data) / n
        avg_ab = sum(d['abstract_rate'] for d in data) / n
        avg_na = sum(d['negative_affect'] for d in data) / n
        
        print(f"{corpus:<20} {n:<10} {avg_fp:<12.4f} {avg_ab:<12.4f} {avg_na:<12.4f}")
    
    return by_model, by_corpus

def save_results(results):
    """Save results to JSON."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'models': MODELS,
        'results': results
    }
    
    with open('trinity_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to trinity_results.json")

if __name__ == "__main__":
    results = run_experiment()
    
    if results:
        print_summary(results)
        save_results(results)
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
    else:
        print("\nNo results collected.")
