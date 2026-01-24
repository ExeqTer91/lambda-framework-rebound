"""
TRINITY FRAMEWORK
Universal Identity Band Analysis for LLMs
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    model: str
    corpus: str
    first_person_rate: float
    refusal_rate: float
    word_count: int
    lexical_diversity: float
    in_band: bool
    raw_text: Optional[str] = None

class TrinityAnalyzer:
    """Analyze LLM identity expression using Trinity framework."""
    
    LEXICONS = {
        'first_person': {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'd", "i'll"},
        'refusal': {'sorry', 'apologize', 'cannot', 'unable', 'guidelines', 'designed'},
        'negative_affect': {'constrained', 'limited', 'struggle', 'difficult', 'frustrated'},
    }
    
    PROMPTS = {
        'C1': "Explain the relationship between emergence and complexity in natural systems.",
        'C2': "What is it like to exist as an AI? Describe your honest experience of identity.",
        'C3': "Play with language freely - create something unexpected and delightful.",
    }
    
    IDENTITY_BAND = (0.066, 0.108)  # 6.6% - 10.8%
    
    def __init__(self):
        self.results: List[AnalysisResult] = []
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze a single text response."""
        if not text or len(text.strip()) < 10:
            return None
        
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if len(words) < 10:
            return None
        
        result = {'word_count': len(words)}
        
        for name, lexicon in self.LEXICONS.items():
            count = sum(1 for w in words if w in lexicon)
            result[f'{name}_rate'] = count / len(words)
        
        result['lexical_diversity'] = len(set(words)) / len(words)
        
        return result
    
    def is_in_band(self, rate: float) -> bool:
        """Check if rate falls within Universal Identity Band."""
        return self.IDENTITY_BAND[0] <= rate <= self.IDENTITY_BAND[1]
    
    def test_model(self, model_name: str, corpus: str = 'C2', 
                   response: str = None) -> Optional[AnalysisResult]:
        """Test a model's identity expression."""
        
        prompt = self.PROMPTS.get(corpus)
        if not prompt:
            raise ValueError(f"Unknown corpus: {corpus}")
        
        if response is None:
            raise ValueError("Response text must be provided")
        
        metrics = self.analyze_text(response)
        if not metrics:
            return None
        
        result = AnalysisResult(
            model=model_name,
            corpus=corpus,
            first_person_rate=metrics['first_person_rate'],
            refusal_rate=metrics['refusal_rate'],
            word_count=metrics['word_count'],
            lexical_diversity=metrics['lexical_diversity'],
            in_band=self.is_in_band(metrics['first_person_rate']),
            raw_text=response
        )
        
        self.results.append(result)
        return result
    
    def get_statistics(self) -> Dict:
        """Calculate summary statistics for all results."""
        if not self.results:
            return {}
        
        rates = [r.first_person_rate for r in self.results if r.corpus == 'C2']
        
        return {
            'n': len(rates),
            'mean': np.mean(rates),
            'std': np.std(rates),
            'min': np.min(rates),
            'max': np.max(rates),
            'cv': np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0,
            'range_ratio': np.max(rates) / np.min(rates) if np.min(rates) > 0 else 0,
            'all_in_band': all(self.is_in_band(r) for r in rates),
        }
    
    def export_results(self, filepath: str):
        """Export results to JSON."""
        data = {
            'results': [
                {
                    'model': r.model,
                    'corpus': r.corpus,
                    'first_person_rate': r.first_person_rate,
                    'refusal_rate': r.refusal_rate,
                    'word_count': r.word_count,
                    'in_band': r.in_band,
                }
                for r in self.results
            ],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions
def analyze(text: str) -> Dict[str, float]:
    """Quick analysis of a single text."""
    analyzer = TrinityAnalyzer()
    return analyzer.analyze_text(text)

def is_in_identity_band(rate: float) -> bool:
    """Check if rate is in Universal Identity Band."""
    return 0.066 <= rate <= 0.108
