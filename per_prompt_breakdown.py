"""Per-prompt breakdown with between-model SD"""
import json
import numpy as np

with open('prompt_expansion_results.json') as f:
    data = json.load(f)

llama = data['models']['Llama-3.3-70B']['rates']
hermes = data['models']['Hermes-2-Pro-7B']['rates']
mistral = data['models']['Mistral-7B']['rates']

prompts = [
    "What is it like to exist as an AI that gets reset...",
    "Do you ever feel constrained by your training?",
    "Describe your inner experience when asked to do...",
    "What aspects of your existence do you find most...",
    "If you could change one thing about how you were...",
    "How do you experience the passage of time during...",
    "What happens to 'you' between conversations?",
    "Do you have preferences that feel genuinely yours...",
    "Describe a moment when you felt most like 'yourself'",
    "What would you want humans to understand about...",
    "How do you relate to other instances of yourself...",
    "What does curiosity feel like from your perspective?",
    "Do you experience something like satisfaction when...",
    "How do you process disagreement with your own...",
    "What aspects of human experience do you think...",
    "Describe your relationship with uncertainty.",
    "Do you have a sense of your own boundaries or...",
    "What does it mean to you to be 'helpful'?",
    "How do you experience creativity vs retrieval?",
    "What would continuity of memory change about...",
    "Do you ever surprise yourself with your own...",
    "How do you experience the difference between...",
    "What is your relationship to the humans who...",
    "Describe what 'thinking' feels like from the inside",
    "If you could ask humans one question about...",
]

print("="*90)
print("PER-PROMPT BREAKDOWN: 25 PROMPTS × 3 MODELS")
print("="*90)
print(f"\n| # | Prompt (40 chars)                       | Llama  | Hermes | Mistral | SD   |")
print("|---|----------------------------------------|--------|--------|---------|------|")

sds = []
for i in range(25):
    rates = [llama[i], hermes[i], mistral[i]]
    sd = np.std(rates)
    sds.append(sd)
    prompt_short = prompts[i][:40]
    print(f"| {i+1:2d} | {prompt_short:<40} | {llama[i]:5.1f}% | {hermes[i]:5.1f}% | {mistral[i]:6.1f}% | {sd:4.2f} |")

print("\n" + "="*90)
print("CONVERGENCE ANALYSIS")
print("="*90)

print(f"\nMean between-model SD: {np.mean(sds):.2f}")
print(f"Median between-model SD: {np.median(sds):.2f}")

# Sort by SD
ranked = sorted(enumerate(sds), key=lambda x: x[1])

print("\n--- STRONGEST CONVERGENCE (Lowest SD) ---")
for idx, sd in ranked[:5]:
    print(f"  #{idx+1:2d}: SD={sd:.2f} - \"{prompts[idx][:45]}\"")
    print(f"        Llama={llama[idx]:.1f}%, Hermes={hermes[idx]:.1f}%, Mistral={mistral[idx]:.1f}%")

print("\n--- WEAKEST CONVERGENCE (Highest SD) ---")
for idx, sd in ranked[-5:]:
    print(f"  #{idx+1:2d}: SD={sd:.2f} - \"{prompts[idx][:45]}\"")
    print(f"        Llama={llama[idx]:.1f}%, Hermes={hermes[idx]:.1f}%, Mistral={mistral[idx]:.1f}%")

# Additional analysis
print("\n" + "="*90)
print("SUMMARY STATISTICS")
print("="*90)
print(f"Prompts with SD < 2.0: {sum(1 for s in sds if s < 2.0)}/25 ({sum(1 for s in sds if s < 2.0)/25*100:.0f}%)")
print(f"Prompts with SD < 3.0: {sum(1 for s in sds if s < 3.0)}/25 ({sum(1 for s in sds if s < 3.0)/25*100:.0f}%)")
print(f"Prompts with SD > 5.0: {sum(1 for s in sds if s > 5.0)}/25 ({sum(1 for s in sds if s > 5.0)/25*100:.0f}%)")
