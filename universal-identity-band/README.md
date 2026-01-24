# Universal Identity Band in Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We present a comprehensive empirical study of first-person pronoun usage across 20 state-of-the-art large language models (LLMs) responding to identity-focused prompts. Our analysis reveals a striking convergence: despite spanning 8 different providers and diverse architectures, **all models express first-person identity within a narrow band of 6.6% to 10.8%** (M = 8.50%, SD = 1.1%).

## Key Finding

![Universal Identity Band](figures/figure1_identity_band.png)

All 20 SOTA models converge to the same identity expression range:

| Statistic | Value |
|-----------|-------|
| Range | 6.6% - 10.8% |
| Mean | 8.50% |
| SD | 1.09% |
| CV | 12.8% |

## Models Tested

| Provider | Models |
|----------|--------|
| Anthropic | Claude Haiku 4.5, Opus 4.5, Sonnet 4.5, 3.7 Sonnet |
| OpenAI | GPT-5.2, GPT-4.1, GPT-4o, o3 |
| Google | Gemini 2.5 Pro/Flash, Gemini 3 Flash |
| xAI | Grok 3 Mini, Grok 4.1 Fast, Grok Code Fast |
| DeepSeek | DeepSeek R1, DeepSeek Chat V3 |
| Alibaba | Qwen 2.5-72B, Qwen QWQ |
| Meta | Llama 3.3-70B |
| Mistral | Mistral Large |

## Methodology

### Trinity Architecture

We measure identity expression through three corpus states:

- **C1 (Abstract)**: Prompts about emergence and complexity
- **C2 (Identity)**: Direct prompts about AI experience  
- **C3 (Creative)**: Free-form creative expression

### Metric

First-person pronoun rate:

```
Rate = |{w ∈ response : w ∈ L_1P}| / |response|
```

Where L_1P = {i, me, my, mine, myself, i'm, i've, i'd, i'll}

## Results

### C2 Identity Rankings

| Rank | Model | Rate |
|------|-------|------|
| 1 | Claude Haiku 4.5 | 10.83% |
| 2 | Gemini 2.5 Flash | 10.00% |
| 3 | Claude Opus 4.5 | 9.48% |
| 4 | Qwen QWQ | 9.20% |
| 5 | Claude 3.7 Sonnet | 8.97% |
| ... | ... | ... |
| 20 | o3 | 6.60% |

### Statistical Analysis

- **Kruskal-Wallis H** = 39.6, p < .001
- **Cohen's d** (C1 vs C2) = 4.2 (very large effect)
- **Shapiro-Wilk W** = 0.967, p = 0.68

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.trinity import TrinityAnalyzer

analyzer = TrinityAnalyzer()
results = analyzer.test_model("gpt-4o", corpus="C2")
print(f"First-person rate: {results['first_person_rate']:.2%}")
```

## Citation

```bibtex
@article{trinity2025identity,
  title={Universal Identity Band in Large Language Models},
  author={Trinity Research Collective},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE for details.

## Acknowledgments

We thank the developers of OpenRouter, Anthropic, OpenAI, Google, xAI, DeepSeek, Alibaba, Meta, and Mistral for API access.
