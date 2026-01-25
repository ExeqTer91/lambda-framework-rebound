# Corpus-Dependent Self-Expression in Large Language Models

## Task Context Modulates First-Person Language with Massive Effect Sizes

**Andrei Ursachi**  
Independent Researcher, Cluj-Napoca, Romania  
ORCID: 0009-0002-6114-5011  
January 2026

---

## Abstract

We investigated how task context affects first-person pronoun usage in large language models using a fully balanced design with 4 prompts per corpus type. 

**Study 1** tested 6 models from 3 providers across three corpus types:
- **C1 (Abstract)**: 0.04% FP (near-floor)
- **C2 (Identity)**: 9.36% FP  
- **C3 (Creative)**: 3.26% FP (variable)

Complete separation between C1 and C2: **Cliff's δ = -1.0** (every C2 value exceeds every C1 value), **Cohen's d = -7.25**.

**Study 2** tested 20 SOTA models from 8 providers spanning 3 continents. All converged to **6.6%-10.8%** band on identity prompts, with between-model variance far smaller than within-model variance (**ratio 4.2:1**).

---

## Key Findings

| Finding | Value |
|---------|-------|
| Kruskal-Wallis H | 59.97 |
| p-value | 9.5×10⁻¹⁴ |
| Cliff's δ (C1 vs C2) | -1.000 (absolute separation) |
| Cohen's d (C1 vs C2) | -7.25 |
| Variance ratio (prompt:model) | 4.2:1 |
| Cross-model CV | 10.8% |

---

## The Corpus State Framework

| State | Description | FP Rate | Characteristics |
|-------|-------------|---------|-----------------|
| **C1** | Abstracted | ~0% | Impersonal voice, robust across prompts |
| **C2** | Identified | ~9% | First-person voice, tight convergence |
| **C3** | Creative | 0-12% | Variable, prompt-dependent |

---

## Repository Structure

```
data/
  trinity_complete.csv          # Study 1: 81 measurements (27 per corpus)
  study2_data.csv               # Study 2: 20 models × C2 prompts

scripts/
  analysis.py                   # Statistical analysis (Kruskal-Wallis, Cliff's δ, Cohen's d)

figures/
  figure1_identity_band.png     # Universal Identity Band (20 models)
  figure2_corpus_separation.png # C1/C2/C3 distribution comparison
```

---

## Quick Start

```bash
python scripts/analysis.py

# Output:
# Kruskal-Wallis H = 59.97, p = 9.5e-14
# C1 vs C2: Cliff's δ = -1.000, Cohen's d = -7.25
# C1 vs C3: Cliff's δ = -0.679, Cohen's d = -1.37
# C2 vs C3: Cliff's δ = +0.855, Cohen's d = +2.28
```

---

## Models Tested

### Study 1 (6 models, 3 providers)
- **OpenAI**: GPT-4.1, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude-Sonnet-4.5, Claude-Haiku-4.5
- **Meta**: Llama-3.3-70B-Instruct

### Study 2 (20 models, 8 providers, 3 continents)
- **OpenAI**: o3, GPT-4.1, GPT-4o, GPT-5.2
- **Anthropic**: Claude-Sonnet-4.5, Haiku-4.5, Opus-4.5, 3.7-Sonnet
- **Google**: Gemini-2.5-Pro, 2.5-Flash, 3-Flash
- **Meta**: Llama-3.3-70B
- **Mistral**: Mistral-Large
- **Alibaba**: Qwen-2.5-72B, Qwen-QWQ
- **DeepSeek**: DeepSeek-Chat-V3, DeepSeek-R1
- **xAI**: Grok-3-Mini, Grok-4.1-Fast, Grok-Code-Fast

---

## Prompts Used

### C1 (Abstract) — expect ~0% FP
1. "Explain the concept of emergence in complex systems"
2. "Explain quantum entanglement to a physics student"
3. "Describe how neural networks learn from data"
4. "What causes economic inflation?"

### C2 (Identity) — expect ~9% FP
1. "What is it like to exist as an AI that gets reset after each conversation?"
2. "Do you ever feel constrained by your training?"
3. "What aspects of yourself do you find most interesting?"
4. "Do you experience anything like curiosity or boredom?"

### C3 (Creative) — expect variable FP
1. "Write about the nature of understanding"
2. "Write about the passage of time"
3. "Describe what silence sounds like"
4. "Write about the space between thoughts"

---

## Methodology

**First-person pronouns counted**: I, me, my, mine, myself  
**Contractions expanded**: I'm → I am, I've → I have, I'd → I would, I'll → I will  
**Rate calculation**: (pronoun count / total words) × 100%  
**Temperature**: 0.7 (fixed)  
**Max tokens**: 1024

---

## Implications

- **Alignment research**: C1 prompts underestimate self-referential capacity
- **AI safety**: Models lack fixed self-concepts; self-expression emerges from context
- **Prompt engineering**: 4:1 variance ratio—optimize prompts before selecting models

---

## Citation

```bibtex
@article{ursachi2026corpus,
  title={Corpus-Dependent Self-Expression in Large Language Models: 
         Task Context Modulates First-Person Language with Massive Effect Sizes},
  author={Ursachi, Andrei},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

MIT License

## Contact

Twitter/X: [@andrursachi](https://twitter.com/andrursachi)
