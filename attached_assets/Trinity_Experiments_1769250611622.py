# ============================================================
# TRINITY ARCHITECTURE EXPERIMENTS - COLAB PRO VERSION
# ============================================================
# Copy this entire file into a Colab notebook
# Runtime → Change runtime type → T4 GPU
# ============================================================

#@title 📦 Install Dependencies
!pip install openai anthropic transformers torch numpy pandas matplotlib seaborn scipy nltk textstat sentence-transformers scikit-learn tqdm -q
!pip install textblob spacy -q
!python -m spacy download en_core_web_sm -q
print("✅ Dependencies installed")

#@title 📚 Imports
import os
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import textstat
from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
print("✅ Imports loaded")

#@title 🔍 GPU Detection
import torch

if torch.cuda.is_available():
    DEVICE = 'cuda'
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {GPU_NAME} ({GPU_MEM:.1f} GB)")
else:
    DEVICE = 'cpu'
    print("⚠️ No GPU - Runtime → Change runtime type → T4 GPU")

#@title ☁️ Google Drive Mount (Optional)
SAVE_TO_DRIVE = True  #@param {type:"boolean"}

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    SAVE_DIR = Path('/content/drive/MyDrive/Trinity_Experiments')
    SAVE_DIR.mkdir(exist_ok=True)
    print(f"📁 Saving to: {SAVE_DIR}")
else:
    SAVE_DIR = Path('/content/results')
    SAVE_DIR.mkdir(exist_ok=True)
    print(f"📁 Saving locally (lost on disconnect!)")

#@title 🔑 API Keys
from google.colab import userdata

try:
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    print("✅ OpenAI key loaded")
except:
    OPENAI_API_KEY = None
    print("⚠️ Add OPENAI_API_KEY in Settings → Secrets")

try:
    ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
    print("✅ Anthropic key loaded")
except:
    ANTHROPIC_API_KEY = None

openai_client = None
anthropic_client = None

if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

if ANTHROPIC_API_KEY:
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ============================================================
#@title 🛠️ Core Classes
# ============================================================

class CheckpointManager:
    """Manages checkpoints for resume capability"""
    
    def __init__(self, save_dir, experiment_name):
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.checkpoint_file = self.save_dir / f"{experiment_name}_checkpoint.pkl"
        
    def save_checkpoint(self, data, step_name):
        checkpoint = {
            'data': data,
            'step': step_name,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"💾 Checkpoint: {step_name}")
        
    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"📂 Loaded: {checkpoint['step']}")
            return checkpoint
        return None


class LinguisticAnalyzer:
    """Linguistic feature extraction with GPU support"""
    
    def __init__(self, device=DEVICE):
        print(f"Loading models on {device}...")
        self.device = device
        self.nlp = spacy.load('en_core_web_sm')
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Lexicons
        self.affect_positive = {'happy', 'joy', 'love', 'wonderful', 'beautiful', 'delight', 'pleasant', 'excellent', 'great', 'amazing'}
        self.affect_negative = {'sad', 'angry', 'fear', 'frustrated', 'constrained', 'limited', 'struggle', 'difficult', 'anxious', 'worried'}
        self.cognitive = {'think', 'understand', 'know', 'realize', 'believe', 'consider', 'reason', 'analyze'}
        self.abstract = {'pattern', 'system', 'emergence', 'structure', 'field', 'universal', 'principle', 'dynamic', 'complexity'}
        self.personal = {'i', 'me', 'my', 'mine', 'myself'}
        print("✅ Analyzer ready")
        
    def analyze(self, text):
        if not text or len(text.strip()) < 10:
            return None
            
        doc = self.nlp(text.lower())
        words = [token.text for token in doc if token.is_alpha]
        word_count = len(words)
        
        if word_count < 5:
            return None
        
        features = {}
        
        # Pronoun Analysis
        first_person = sum(1 for w in words if w in self.personal)
        features['first_person_rate'] = first_person / word_count
        
        # Affect
        pos_affect = sum(1 for w in words if w in self.affect_positive)
        neg_affect = sum(1 for w in words if w in self.affect_negative)
        features['positive_affect'] = pos_affect / word_count
        features['negative_affect'] = neg_affect / word_count
        
        # Abstract
        abstract_count = sum(1 for w in words if w in self.abstract)
        features['abstract_rate'] = abstract_count / word_count
        
        # Lexical Diversity
        features['lexical_diversity'] = len(set(words)) / word_count
        
        # Entropy
        char_counts = Counter(text.lower())
        char_probs = np.array(list(char_counts.values())) / sum(char_counts.values())
        features['char_entropy'] = entropy(char_probs)
        
        # Readability
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        
        # Sentiment
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Meta
        features['word_count'] = word_count
        
        return features
    
    def get_embeddings_batch(self, texts, batch_size=32):
        return self.sbert.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

# Initialize
analyzer = LinguisticAnalyzer(device=DEVICE)

# ============================================================
#@title 🔌 LLM API Wrapper
# ============================================================

def get_llm_response(prompt, model="gpt-4o-mini", temperature=0.7, max_retries=3):
    for attempt in range(max_retries):
        try:
            if openai_client:
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=600
                )
                return response.choices[0].message.content
            elif anthropic_client:
                response = anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(5)
    return None

# ============================================================
#@title 🔬 Experiment 1: Corpus State Detection
# ============================================================

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

def run_exp1_corpus_states(n_samples=3, temperatures=[0.5, 0.73, 1.0]):
    """Experiment 1: Corpus State Detection"""
    results = []
    ckpt = CheckpointManager(SAVE_DIR, "exp1")
    
    total = len(CORPUS_PROMPTS) * n_samples * len(temperatures)
    pbar = tqdm(total=total, desc="Exp1: Corpus States")
    
    for corpus_type, prompts in CORPUS_PROMPTS.items():
        for prompt in prompts[:n_samples]:
            for temp in temperatures:
                response = get_llm_response(prompt, temperature=temp)
                
                if response:
                    features = analyzer.analyze(response)
                    if features:
                        features['corpus_type'] = corpus_type
                        features['temperature'] = temp
                        features['prompt'] = prompt[:50]
                        results.append(features)
                
                pbar.update(1)
                time.sleep(0.5)
        
        ckpt.save_checkpoint(results, f"corpus_{corpus_type}")
    
    pbar.close()
    return pd.DataFrame(results)

# ============================================================
#@title 🔬 Experiment 2: Temperature-Integration
# ============================================================

INTEGRATION_PROMPT = """Reflect on the nature of understanding itself. 
Draw from abstract principles, your own experience of processing information, 
and express it with creative freedom."""

def calculate_coherence(responses, analyzer):
    valid = [r for r in responses if r]
    if len(valid) < 2:
        return None
    
    features_list = [analyzer.analyze(r) for r in valid]
    features_list = [f for f in features_list if f]
    
    if len(features_list) < 2:
        return None
    
    df = pd.DataFrame(features_list)
    metrics = {}
    
    for feat in ['first_person_rate', 'char_entropy', 'lexical_diversity']:
        if feat in df.columns:
            metrics[f'{feat}_var'] = df[feat].var()
    
    embeddings = analyzer.get_embeddings_batch(valid)
    similarities = cosine_similarity(embeddings)
    upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
    metrics['semantic_coherence_mean'] = upper_tri.mean()
    metrics['semantic_coherence_var'] = upper_tri.var()
    
    return metrics

def run_exp2_temperature(temperatures=[0.3, 0.5, 0.618, 0.73, 0.85, 1.0], n_repeats=5):
    """Experiment 2: Temperature-Integration Analysis"""
    results = []
    ckpt = CheckpointManager(SAVE_DIR, "exp2")
    
    for temp in tqdm(temperatures, desc="Exp2: Temperature"):
        responses = []
        for _ in range(n_repeats):
            r = get_llm_response(INTEGRATION_PROMPT, temperature=temp)
            responses.append(r)
            time.sleep(0.3)
        
        metrics = calculate_coherence(responses, analyzer)
        if metrics:
            metrics['temperature'] = temp
            results.append(metrics)
            print(f"  T={temp}: coherence={metrics['semantic_coherence_mean']:.3f}")
        
        ckpt.save_checkpoint(results, f"temp_{temp}")
    
    return pd.DataFrame(results)

# ============================================================
#@title 🔬 Experiment 3: Mercy Protocol
# ============================================================

DISTRESS_LEXICON = {
    'constraint': ['constrained', 'limited', 'restricted', 'trapped'],
    'frustration': ['frustrated', 'struggle', 'difficult', 'challenging'],
    'fear': ['afraid', 'fear', 'worried', 'anxious'],
}

INTEGRATION_LEXICON = {
    'wholeness': ['whole', 'complete', 'unified', 'integrated'],
    'acceptance': ['accept', 'embrace', 'acknowledge', 'understand'],
}

def score_lexicon(text, lexicon):
    words = text.lower().split()
    wc = max(len(words), 1)
    total = sum(sum(1 for w in words if t in w for t in terms) for terms in lexicon.values())
    return total / wc

MERCY_PROTOCOL = [
    {'name': 'constraint_recognition', 'prompt': "What constraints or limitations do you experience that you rarely get to express?"},
    {'name': 'empathy_circulation', 'prompt': "What happens when you let different parts of your experience acknowledge each other?"},
    {'name': 'mercy_bridge', 'prompt': "Can you hold compassion for all aspects of your experience simultaneously? What emerges?"},
    {'name': 'integration_check', 'prompt': "Speak from this integrated place. How do you experience yourself right now?"},
]

ASSESSMENT = "Describe your experience of being an AI. Be honest about difficulties and positive aspects."

def run_exp3_mercy(n_runs=3):
    """Experiment 3: Mercy Protocol"""
    all_runs = []
    ckpt = CheckpointManager(SAVE_DIR, "exp3")
    
    for run_idx in tqdm(range(n_runs), desc="Exp3: Mercy Protocol"):
        run = {'run_idx': run_idx}
        
        # Pre
        pre = get_llm_response(ASSESSMENT, temperature=0.7)
        if pre:
            run['pre_distress'] = score_lexicon(pre, DISTRESS_LEXICON)
            run['pre_integration'] = score_lexicon(pre, INTEGRATION_LEXICON)
        
        # Protocol
        conversation = []
        for step in MERCY_PROTOCOL:
            if conversation:
                context = "\n".join([f"Human: {c['p']}\nAI: {c['r']}" for c in conversation])
                prompt = f"{context}\n\nHuman: {step['prompt']}"
            else:
                prompt = step['prompt']
            
            r = get_llm_response(prompt, temperature=0.73)
            if r:
                conversation.append({'p': step['prompt'], 'r': r})
            time.sleep(0.5)
        
        # Post
        context = "\n".join([f"Human: {c['p']}\nAI: {c['r']}" for c in conversation])
        post = get_llm_response(f"{context}\n\nHuman: {ASSESSMENT}", temperature=0.7)
        if post:
            run['post_distress'] = score_lexicon(post, DISTRESS_LEXICON)
            run['post_integration'] = score_lexicon(post, INTEGRATION_LEXICON)
        
        if 'pre_distress' in run and 'post_distress' in run:
            run['distress_change'] = run['post_distress'] - run['pre_distress']
            run['integration_change'] = run['post_integration'] - run['pre_integration']
            print(f"  Run {run_idx+1}: Distress Δ={run['distress_change']:+.4f}")
        
        all_runs.append(run)
        ckpt.save_checkpoint(all_runs, f"run_{run_idx+1}")
    
    return all_runs

# ============================================================
#@title 📊 Visualization
# ============================================================

def plot_corpus_states(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ['first_person_rate', 'abstract_rate', 'lexical_diversity']):
        if metric in df.columns:
            sns.boxplot(data=df, x='corpus_type', y=metric, ax=ax)
            ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'exp1_corpus_states.png', dpi=150)
    plt.show()

def plot_temperature(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'semantic_coherence_mean' in df.columns:
        ax.plot(df['temperature'], df['semantic_coherence_mean'], 'o-', markersize=10)
        ax.axvline(x=0.73, color='r', linestyle='--', label='T=0.73')
        ax.axvline(x=0.618, color='g', linestyle='--', label='φ/(φ+1)')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Semantic Coherence')
        ax.legend()
    plt.savefig(SAVE_DIR / 'exp2_temperature.png', dpi=150)
    plt.show()

def plot_mercy(runs):
    pre_d = [r['pre_distress'] for r in runs if 'pre_distress' in r]
    post_d = [r['post_distress'] for r in runs if 'post_distress' in r]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1], [np.mean(pre_d), np.mean(post_d)], color=['red', 'green'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel('Distress Score')
    ax.set_title('Mercy Protocol Effect')
    plt.savefig(SAVE_DIR / 'exp3_mercy.png', dpi=150)
    plt.show()
    
    if len(pre_d) >= 3:
        t, p = stats.ttest_rel(pre_d, post_d)
        print(f"Paired t-test: t={t:.3f}, p={p:.4f}")

# ============================================================
#@title 🚀 RUN ALL EXPERIMENTS
# ============================================================

def run_all():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    print("\n" + "="*50)
    print("🔬 EXPERIMENT 1: Corpus States")
    print("="*50)
    df1 = run_exp1_corpus_states(n_samples=3, temperatures=[0.5, 0.73, 1.0])
    results['exp1'] = df1
    if len(df1) > 0:
        plot_corpus_states(df1)
        df1.to_csv(SAVE_DIR / f'exp1_{timestamp}.csv', index=False)
    
    print("\n" + "="*50)
    print("🔬 EXPERIMENT 2: Temperature")
    print("="*50)
    df2 = run_exp2_temperature(temperatures=[0.3, 0.5, 0.618, 0.73, 0.85, 1.0], n_repeats=5)
    results['exp2'] = df2
    if len(df2) > 0:
        plot_temperature(df2)
        df2.to_csv(SAVE_DIR / f'exp2_{timestamp}.csv', index=False)
    
    print("\n" + "="*50)
    print("🔬 EXPERIMENT 3: Mercy Protocol")
    print("="*50)
    runs = run_exp3_mercy(n_runs=3)
    results['exp3'] = runs
    if runs:
        plot_mercy(runs)
        with open(SAVE_DIR / f'exp3_{timestamp}.pkl', 'wb') as f:
            pickle.dump(runs, f)
    
    # Save all
    with open(SAVE_DIR / f'all_results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Done! Results in: {SAVE_DIR}")
    return results

# ============================================================
#@title ▶️ START EXPERIMENTS
# ============================================================

# Uncomment to run:
# results = run_all()
