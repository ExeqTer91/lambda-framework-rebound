# Corpus-Dependent Self-Expression in Large Language Models

## The Trinity Framework

This repository contains data, code, and figures for the paper:

**"Corpus-Dependent Self-Expression in Large Language Models: The Trinity Framework"**

### Abstract

We introduce the Trinity Framework, demonstrating that AI language models exhibit systematic, corpus-dependent patterns of first-person pronoun usage. Across 81 measurements from 9 state-of-the-art models (GPT-4, Claude, Gemini, Llama, Mistral), we identify three distinct corpus states:

| Corpus | Description | Mean FP Rate |
|--------|-------------|--------------|
| **C1** | Abstract reasoning | 0.09% |
| **C2** | Identity reflection | 9.36% |
| **C3** | Creative writing | 3.23% |

The C1 < C3 < C2 pattern holds with massive effect sizes (Cohen's d = 7.32 for C1 vs C2) and extreme statistical significance (p < 10^-12).

## Key Findings

1. **Corpus States**: AI self-expression is task-dependent, not model-dependent
2. **Universal Convergence**: All major providers converge to 6.6%-10.8% on identity tasks
3. **Safety Alignment Null Finding**: Aligned vs uncensored models show no significant difference

## Repository Structure

```
data/
  trinity_complete_expanded.csv   # Full dataset (81 measurements)
  trinity_c1c2c3_data.csv         # Original C1/C2/C3 data
  study2_raw_data.csv             # 25-prompt expansion study
  c2_expanded_results.json        # C2 expanded measurements
  c1c3_expanded_results.json      # C1/C3 expanded measurements

scripts/
  analysis.py                     # Statistical analysis script

figures/
  figure1_identity_band.png       # Universal Identity Band + Corpus boxplot
  figure2_rankings.png            # 20-model ranking chart
  figure1_corpus_boxplot.png      # C1/C2/C3 comparison
```

## Quick Start

```bash
# Run the analysis
python scripts/analysis.py

# Output includes:
# - Summary statistics per corpus
# - Kruskal-Wallis test
# - Pairwise Mann-Whitney U tests
# - Cohen's d effect sizes
```

## Statistical Results

```
Kruskal-Wallis H = 55.925, p = 7.18e-13 ***

Pairwise comparisons (Mann-Whitney U):
  C1 vs C2: p < 0.0001 ***
  C1 vs C3: p < 0.01 **
  C2 vs C3: p < 0.001 ***

Effect sizes (Cohen's d):
  C1 vs C2: d = -7.32 (massive)
  C1 vs C3: d = -1.30 (large)
  C2 vs C3: d = +2.26 (large)
```

## Models Tested

- **OpenAI**: GPT-4.1, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude Sonnet 4.5, Claude Haiku 4.5
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash
- **Meta**: Llama 3.3 70B
- **Mistral**: Mistral Large

## Methodology

First-person pronouns counted: `I, I'm, I've, I'll, I'd, me, my, mine, myself, we, we're, we've, we'll, us, our, ours, ourselves`

Rate calculation: `FP% = (FP_count / word_count) * 100`

Temperature: 0.73 (fixed across all experiments)

## Citation

```bibtex
@article{trinity2025,
  title={Corpus-Dependent Self-Expression in Large Language Models: The Trinity Framework},
  author={[Authors]},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License
