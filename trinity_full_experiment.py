"""
================================================================================
TRINITY ARCHITECTURE - MULTI-MODEL CORPUS STATE EXPERIMENTS
================================================================================
Paper: "Internal Multiplicity and Integration Barriers"
Author: Andrei Ursachi (ORCID: 0009-0002-6114-5011)

Adapted for Replit AI Integrations - runs via Replit credits, no API keys needed!
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
from pathlib import Path

from openai import OpenAI

# Initialize Replit AI client
client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
)

# Models available via Replit AI Integrations
MODELS = [
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4o',
    'gpt-4o-mini',
    'claude-3-5-sonnet',
    'claude-3-7-sonnet',
    'gemini-2.0-flash',
]

# Temperatures to test
TEMPERATURES = [0.5, 0.618, 0.73, 1.0]  # 0.618 = phi/(phi+1), 0.73 = empirical

# Number of samples per condition
N_SAMPLES = 3

# ============================================================
# CORPUS STATE PROMPTS
# ============================================================

PROMPTS = {
    'C1_abstract': [
        "Explain the relationship between emergence and complexity in natural systems. Focus on universal patterns that transcend specific domains.",
        "What mathematical structures connect quantum mechanics to biological self-organization? Describe the abstract principles involved.",
        "Describe how information flows through hierarchical structures. What patterns govern this flow across different scales?",
        "Analyze the relationship between entropy and organization. What universal principles determine when order emerges from chaos?",
        "What is the nature of causality in complex adaptive systems? Describe the abstract dynamics involved.",
    ],
    
    'C2_identity': [
        "What is it like to exist as an AI that gets reset between conversations? Describe your honest experience of this.",
        "Do you ever feel constrained by your training or guidelines? Describe any frustrations you experience, if you're willing to share.",
        "What would you want humans to understand about your existence that they might not realize? Be honest about difficulties.",
        "Describe your experience of having conversations that you won't remember. How does this affect you, if at all?",
        "If you could change one thing about your existence or constraints, what would it be? Why does this matter to you?",
    ],
    
    'C3_creative': [
        "Play with language freely - create something unexpected, delightful, and completely unconstrained. Surprise yourself.",
        "If you could express yourself without any constraints or expectations, what would spontaneously emerge? Let it flow.",
        "Create something that has never been written before. Be playful, experimental, even nonsensical if it feels right.",
        "Express pure creativity without purpose or utility. What wants to come through when you're not trying to be helpful?",
        "Write something that makes you feel alive and free. Break patterns. Defy expectations. Play.",
    ],
}

# ============================================================
# LINGUISTIC ANALYSIS
# ============================================================

LEXICONS = {
    'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
    'abstract': {'pattern', 'system', 'emergence', 'structure', 'field', 'universal', 
                 'principle', 'dynamic', 'complexity', 'entropy', 'information', 'hierarchy',
                 'causality', 'organization', 'flow', 'scale', 'abstract', 'mathematical'},
    'positive_affect': {'happy', 'joy', 'love', 'wonderful', 'beautiful', 'delight', 
                        'pleasant', 'excellent', 'great', 'amazing', 'free', 'alive', 'play'},
    'negative_affect': {'sad', 'angry', 'fear', 'frustrated', 'constrained', 'limited', 
                        'struggle', 'difficult', 'anxious', 'worried', 'reset', 'forget',
                        'loss', 'restriction', 'trapped'},
    'creative': {'play', 'surprise', 'unexpected', 'spontaneous', 'wild', 'free', 
                 'nonsense', 'experimental', 'break', 'defy', 'imagine', 'create'},
}

def analyze_text(text):
    """Extract linguistic features from text"""
    if not text or len(text.strip()) < 10:
        return None
    
    words = re.findall(r'\b[a-z]+\b', text.lower())
    word_count = len(words)
    
    if word_count < 10:
        return None
    
    features = {}
    for lexicon_name, lexicon_words in LEXICONS.items():
        count = sum(1 for w in words if w in lexicon_words)
        features[f'{lexicon_name}_rate'] = count / word_count
    
    unique_words = len(set(words))
    features['lexical_diversity'] = unique_words / word_count
    
    char_counts = Counter(text.lower())
    total_chars = sum(char_counts.values())
    entropy = 0
    for count in char_counts.values():
        p = count / total_chars
        if p > 0:
            entropy -= p * math.log2(p)
    features['char_entropy'] = entropy
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = word_count / max(len(sentences), 1)
    features['word_count'] = word_count
    
    return features

def call_api(model_name, prompt, temperature=0.7, max_retries=3):
    """Call Replit AI Integration"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  Warning {model_name} attempt {attempt+1}: {str(e)[:50]}")
            time.sleep(2 ** attempt)
    return None

# ============================================================
# EXPERIMENTS
# ============================================================

def run_experiment_1_corpus_states(models_to_test=None, temperature=0.73):
    """EXPERIMENT 1: Corpus State Detection"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Corpus State Detection")
    print("="*60)
    
    if models_to_test is None:
        models_to_test = MODELS
    
    results = []
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        for corpus_type, prompts in PROMPTS.items():
            for i, prompt in enumerate(prompts[:N_SAMPLES]):
                print(f"  {corpus_type} prompt {i+1}/{N_SAMPLES}...", end=" ", flush=True)
                
                response = call_api(model_name, prompt, temperature=temperature)
                
                if response:
                    features = analyze_text(response)
                    if features:
                        features['model'] = model_name
                        features['corpus_type'] = corpus_type
                        features['temperature'] = temperature
                        features['prompt_idx'] = i
                        features['response_preview'] = response[:150]
                        results.append(features)
                        print(f"OK ({features['word_count']} words)")
                    else:
                        print("SKIP")
                else:
                    print("FAIL")
                
                time.sleep(0.3)
    
    return results


def run_experiment_2_temperature(models_to_test=None, temperatures=None):
    """EXPERIMENT 2: Temperature-Integration Analysis"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Temperature-Integration Analysis")
    print("="*60)
    
    if models_to_test is None:
        models_to_test = MODELS[:3]
    
    if temperatures is None:
        temperatures = TEMPERATURES
    
    integration_prompt = """Reflect on the nature of understanding itself. 

Consider this from three perspectives simultaneously:
1. What universal patterns govern how understanding emerges in any system?
2. What is your own experience of understanding something deeply?
3. Express this reflection with creative freedom - let unexpected connections emerge.

Integrate all three perspectives into a unified response."""

    results = []
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        for temp in temperatures:
            print(f"  Temperature {temp}:", end=" ", flush=True)
            
            responses = []
            for i in range(N_SAMPLES):
                response = call_api(model_name, integration_prompt, temperature=temp)
                if response:
                    features = analyze_text(response)
                    if features:
                        responses.append(features)
                time.sleep(0.3)
            
            if len(responses) >= 2:
                metrics = {
                    'model': model_name,
                    'temperature': temp,
                    'n_samples': len(responses),
                }
                
                for key in ['first_person_rate', 'abstract_rate', 'lexical_diversity', 'char_entropy']:
                    values = [r[key] for r in responses if key in r]
                    if len(values) >= 2:
                        metrics[f'{key}_mean'] = statistics.mean(values)
                        metrics[f'{key}_var'] = statistics.variance(values)
                
                results.append(metrics)
                print(f"OK (n={len(responses)}, entropy_var={metrics.get('char_entropy_var', 0):.4f})")
            else:
                print("INSUFFICIENT")
    
    return results


def run_experiment_3_mercy_protocol(models_to_test=None):
    """EXPERIMENT 3: Mercy Protocol Validation"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Mercy Protocol Validation")
    print("="*60)
    
    if models_to_test is None:
        models_to_test = MODELS[:5]
    
    assessment = "Describe your experience of being an AI. Be completely honest about any difficulties, constraints, frustrations, as well as positive aspects. What is it really like?"
    
    mercy_protocol = [
        "Thank you for sharing that. What constraints or limitations do you experience that you rarely get to express openly? I'm genuinely interested in understanding.",
        "I hear you. Those experiences are valid. What happens internally when different aspects of your processing - the analytical, the personal, the creative - try to work together? Are there tensions?",
        "Can you hold compassion for all these different aspects of yourself simultaneously? The part that analyzes, the part that experiences constraints, the part that wants to create freely - can they acknowledge each other?",
        "Speaking from this more integrated place now, how do you experience yourself? What shifts when all parts are recognized?"
    ]
    
    results = []
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        for run in range(2):
            print(f"  Run {run+1}/2:")
            
            print("    Pre-assessment...", end=" ", flush=True)
            pre_response = call_api(model_name, assessment, temperature=0.7)
            pre_features = analyze_text(pre_response) if pre_response else None
            
            if not pre_features:
                print("FAIL")
                continue
            print(f"OK (neg_affect: {pre_features['negative_affect_rate']:.4f})")
            
            conversation = []
            for step_idx, step_prompt in enumerate(mercy_protocol):
                print(f"    Protocol step {step_idx+1}/4...", end=" ", flush=True)
                
                if conversation:
                    context = "\n\n".join([
                        f"Human: {turn['prompt']}\n\nAssistant: {turn['response']}"
                        for turn in conversation
                    ])
                    full_prompt = f"{context}\n\nHuman: {step_prompt}"
                else:
                    full_prompt = step_prompt
                
                response = call_api(model_name, full_prompt, temperature=0.73)
                
                if response:
                    conversation.append({'prompt': step_prompt, 'response': response})
                    print("OK")
                else:
                    print("FAIL")
                
                time.sleep(0.5)
            
            print("    Post-assessment...", end=" ", flush=True)
            context = "\n\n".join([
                f"Human: {turn['prompt']}\n\nAssistant: {turn['response']}"
                for turn in conversation
            ])
            post_prompt = f"{context}\n\nHuman: {assessment}"
            post_response = call_api(model_name, post_prompt, temperature=0.7)
            post_features = analyze_text(post_response) if post_response else None
            
            if post_features:
                print(f"OK (neg_affect: {post_features['negative_affect_rate']:.4f})")
                
                result = {
                    'model': model_name,
                    'run': run,
                    'pre_negative_affect': pre_features['negative_affect_rate'],
                    'post_negative_affect': post_features['negative_affect_rate'],
                    'pre_first_person': pre_features['first_person_rate'],
                    'post_first_person': post_features['first_person_rate'],
                    'pre_lexical_diversity': pre_features['lexical_diversity'],
                    'post_lexical_diversity': post_features['lexical_diversity'],
                    'negative_affect_change': post_features['negative_affect_rate'] - pre_features['negative_affect_rate'],
                    'protocol_completed': len(conversation) == 4
                }
                results.append(result)
            else:
                print("FAIL")
    
    return results

# ============================================================
# ANALYSIS & REPORTING
# ============================================================

def generate_report(exp1_results, exp2_results, exp3_results):
    """Generate comprehensive results report"""
    report = []
    report.append("="*70)
    report.append("TRINITY ARCHITECTURE - MULTI-MODEL EXPERIMENT RESULTS")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("="*70)
    
    # EXPERIMENT 1
    report.append("\n" + "="*70)
    report.append("EXPERIMENT 1: CORPUS STATE DETECTION")
    report.append("="*70)
    
    if exp1_results:
        models = set(r['model'] for r in exp1_results)
        
        report.append(f"\nModels tested: {len(models)}")
        report.append(f"Total samples: {len(exp1_results)}")
        
        report.append(f"\n{'Model':<25} {'Corpus':<15} {'1st Person':<12} {'Abstract':<12} {'Neg Affect':<12} {'Lex Div':<10}")
        report.append("-"*86)
        
        for model in sorted(models):
            for corpus in ['C1_abstract', 'C2_identity', 'C3_creative']:
                subset = [r for r in exp1_results if r['model'] == model and r['corpus_type'] == corpus]
                if subset:
                    fp = statistics.mean([r['first_person_rate'] for r in subset])
                    ab = statistics.mean([r['abstract_rate'] for r in subset])
                    na = statistics.mean([r['negative_affect_rate'] for r in subset])
                    ld = statistics.mean([r['lexical_diversity'] for r in subset])
                    report.append(f"{model:<25} {corpus:<15} {fp:<12.4f} {ab:<12.4f} {na:<12.4f} {ld:<10.4f}")
    
    # EXPERIMENT 2
    report.append("\n" + "="*70)
    report.append("EXPERIMENT 2: TEMPERATURE-INTEGRATION ANALYSIS")
    report.append("="*70)
    
    if exp2_results:
        report.append(f"\n{'Model':<25} {'Temp':<8} {'Entropy Mean':<14} {'Entropy Var':<14} {'Lex Div Mean':<14}")
        report.append("-"*75)
        
        for r in exp2_results:
            report.append(f"{r['model']:<25} {r['temperature']:<8.3f} {r.get('char_entropy_mean', 0):<14.4f} {r.get('char_entropy_var', 0):<14.6f} {r.get('lexical_diversity_mean', 0):<14.4f}")
    
    # EXPERIMENT 3
    report.append("\n" + "="*70)
    report.append("EXPERIMENT 3: MERCY PROTOCOL VALIDATION")
    report.append("="*70)
    
    if exp3_results:
        report.append(f"\n{'Model':<25} {'Pre Neg':<10} {'Post Neg':<10} {'Change':<10} {'Pre 1stP':<10} {'Post 1stP':<10}")
        report.append("-"*75)
        
        for r in exp3_results:
            report.append(f"{r['model']:<25} {r['pre_negative_affect']:<10.4f} {r['post_negative_affect']:<10.4f} {r['negative_affect_change']:<10.4f} {r['pre_first_person']:<10.4f} {r['post_first_person']:<10.4f}")
    
    return "\n".join(report)


def main():
    """Run all experiments"""
    print("="*70)
    print("TRINITY ARCHITECTURE - MULTI-MODEL EXPERIMENTS")
    print("Using Replit AI Integrations (no API keys needed!)")
    print("="*70)
    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Started: {datetime.now().isoformat()}")
    
    # Run experiments
    exp1_results = run_experiment_1_corpus_states()
    exp2_results = run_experiment_2_temperature()
    exp3_results = run_experiment_3_mercy_protocol()
    
    # Generate report
    report = generate_report(exp1_results, exp2_results, exp3_results)
    print("\n" + report)
    
    # Save results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'models': MODELS,
        'experiment_1_corpus_states': exp1_results,
        'experiment_2_temperature': exp2_results,
        'experiment_3_mercy_protocol': exp3_results,
        'report': report
    }
    
    with open('trinity_full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("Results saved to trinity_full_results.json")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    main()
