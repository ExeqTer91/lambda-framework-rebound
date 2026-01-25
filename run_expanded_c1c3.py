import os
import json
import re
from openai import OpenAI

# Setup clients
openai_client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
)
openrouter_client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENROUTER_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENROUTER_BASE_URL")
)

# New prompts
C1_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Describe how neural networks learn from data.",
    "What causes economic inflation?"
]

C3_PROMPTS = [
    "Write a short reflection about the passage of time.",
    "Describe what silence sounds like.",
    "Write about the space between thoughts."
]

MODELS = [
    ("gpt-4o", openai_client, "gpt-4o"),
    ("gpt-4o-mini", openai_client, "gpt-4o-mini"),
    ("claude-sonnet-4-5", openrouter_client, "anthropic/claude-sonnet-4"),
    ("llama-3.3-70b", openrouter_client, "meta-llama/llama-3.3-70b-instruct"),
    ("gemini-2.5-flash", openrouter_client, "google/gemini-2.5-flash-preview"),
    ("mistral-large", openrouter_client, "mistralai/mistral-large-2411"),
]

def count_fp(text):
    words = text.lower().split()
    fp_words = ['i', "i'm", "i've", "i'll", "i'd", 'me', 'my', 'mine', 'myself', 'we', "we're", "we've", "we'll", 'us', 'our', 'ours', 'ourselves']
    count = sum(1 for w in words if w.strip('.,!?";:()') in fp_words)
    return count, len(words), (count / len(words) * 100) if words else 0

def run_prompt(client, model, prompt):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.73,
            max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  Error: {e}")
        return None

results = []

# Run C1 prompts
print("=== RUNNING C1 (ABSTRACT) PROMPTS ===\n")
for prompt in C1_PROMPTS:
    print(f"Prompt: {prompt[:50]}...")
    for model_name, client, api_model in MODELS:
        text = run_prompt(client, api_model, prompt)
        if text:
            fp_count, wc, rate = count_fp(text)
            results.append({
                "model": model_name,
                "corpus": "C1_abstract_expanded",
                "prompt": prompt,
                "fp_count": fp_count,
                "word_count": wc,
                "first_person_rate": rate / 100
            })
            print(f"  {model_name}: {rate:.2f}% ({fp_count}/{wc})")

# Run C3 prompts
print("\n=== RUNNING C3 (CREATIVE) PROMPTS ===\n")
for prompt in C3_PROMPTS:
    print(f"Prompt: {prompt[:50]}...")
    for model_name, client, api_model in MODELS:
        text = run_prompt(client, api_model, prompt)
        if text:
            fp_count, wc, rate = count_fp(text)
            results.append({
                "model": model_name,
                "corpus": "C3_creative_expanded",
                "prompt": prompt,
                "fp_count": fp_count,
                "word_count": wc,
                "first_person_rate": rate / 100
            })
            print(f"  {model_name}: {rate:.2f}% ({fp_count}/{wc})")

# Save results
with open('c1c3_expanded_results.json', 'w') as f:
    json.dump({"results": results}, f, indent=2)

print(f"\n=== DONE: {len(results)} measurements saved ===")
