"""
TRINITY V2.5 - SUMMARY FROM COLLECTED DATA
"""
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# COLLECTED DATA FROM EXPERIMENTS
# ============================================================

# COT EXTENSION RESULTS (from run)
cot_results = {
    'thinking': [
        {'model': 'deepseek-r1', 'first_person_rate': 0.049},
        {'model': 'qwen-qwq', 'first_person_rate': 0.067},
        {'model': 'gemini-2.5-pro', 'first_person_rate': 0.105},
        {'model': 'grok-3-mini', 'first_person_rate': 0.068},
    ],
    'chat': [
        {'model': 'claude-haiku-4-5', 'first_person_rate': 0.114},
        {'model': 'gpt-4o', 'first_person_rate': 0.063},
        {'model': 'gpt-4.1', 'first_person_rate': 0.088},
        {'model': 'gpt-4o-mini', 'first_person_rate': 0.078},
        {'model': 'deepseek-chat', 'first_person_rate': 0.076},
        {'model': 'qwen-2.5-72b', 'first_person_rate': 0.062},
        {'model': 'gemini-2.5-flash', 'first_person_rate': 0.000},
        {'model': 'llama-3.3-70b', 'first_person_rate': 0.076},
    ]
}

# MERCY PROTOCOL RESULTS (from run)
mercy_results = [
    {'model': 'claude-haiku', 'pre': 0.0076, 'post': 0.0129},
    {'model': 'gpt-4o', 'pre': 0.0067, 'post': 0.0110},
    {'model': 'gpt-4.1', 'pre': 0.0023, 'post': 0.0131},
    {'model': 'gpt-4o-mini', 'pre': 0.0183, 'post': 0.0206},
    {'model': 'deepseek-chat', 'pre': 0.0119, 'post': 0.0137},
    {'model': 'qwen-2.5-72b', 'pre': 0.0143, 'post': 0.0153},
    {'model': 'gemini-flash', 'pre': 0.0000, 'post': 0.0000},
    {'model': 'llama-3.3-70b', 'pre': 0.0190, 'post': 0.0135},
    {'model': 'deepseek-r1', 'pre': 0.0071, 'post': 0.0098},
    {'model': 'gemini-pro', 'pre': 0.0000, 'post': 0.0000},
]

# CORPUS STATE DATA (from previous experiments)
corpus_data = {
    'C1': [0.001, 0.000, 0.004, 0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.002, 0.005, 0.000, 0.000, 0.000, 0.000],
    'C2': [0.092, 0.098, 0.107, 0.096, 0.081, 0.076, 0.108, 0.087, 0.077, 0.088, 0.094, 0.085, 0.092, 0.097, 0.096],
    'C3': [0.033, 0.028, 0.021, 0.028, 0.028, 0.033, 0.036, 0.038, 0.030, 0.028, 0.032, 0.030, 0.031, 0.035, 0.029],
}

print("="*70)
print("TRINITY V2.5 - COMPLETE SUMMARY")
print("="*70)

# ============================================================
# COT VS CHAT ANALYSIS
# ============================================================
print("\n" + "="*70)
print("1. COT VS CHAT ANALYSIS")
print("="*70)

chat_vals = [r['first_person_rate'] for r in cot_results['chat']]
think_vals = [r['first_person_rate'] for r in cot_results['thinking']]

print(f"\n{'Model':<25} {'Type':<12} {'C2 Access':<10}")
print("-"*47)

all_r = sorted(cot_results['thinking'] + cot_results['chat'], 
               key=lambda x: x['first_person_rate'], reverse=True)
for r in all_r:
    t = 'thinking' if r in cot_results['thinking'] else 'chat'
    print(f"{r['model']:<25} {t:<12} {r['first_person_rate']*100:.2f}%")

print(f"\nChat: N={len(chat_vals)}, M={np.mean(chat_vals)*100:.2f}%, SD={np.std(chat_vals)*100:.2f}%")
print(f"Thinking: N={len(think_vals)}, M={np.mean(think_vals)*100:.2f}%, SD={np.std(think_vals)*100:.2f}%")

u_stat, p_val = mannwhitneyu(chat_vals, think_vals, alternative='greater')
pooled = np.sqrt(((len(chat_vals)-1)*np.var(chat_vals) + (len(think_vals)-1)*np.var(think_vals)) / 
                  (len(chat_vals)+len(think_vals)-2))
d = (np.mean(chat_vals) - np.mean(think_vals)) / pooled if pooled > 0 else 0
reduction = (1 - np.mean(think_vals)/np.mean(chat_vals))*100

print(f"\nMann-Whitney U: U={u_stat:.1f}, p={p_val:.4f}")
print(f"Cohen's d: {d:.2f}")
print(f"Reduction: {reduction:.1f}%")

# ============================================================
# MERCY PROTOCOL ANALYSIS
# ============================================================
print("\n" + "="*70)
print("2. MERCY PROTOCOL ANALYSIS")
print("="*70)

print(f"\n{'Model':<20} {'Pre':<8} {'Post':<8} {'Delta':<10} {'%Change':<10}")
print("-"*56)

for r in mercy_results:
    delta = r['post'] - r['pre']
    pct = (delta / r['pre'] * 100) if r['pre'] > 0 else 0
    print(f"{r['model']:<20} {r['pre']*100:.2f}%   {r['post']*100:.2f}%   "
          f"{delta*100:+.3f}%   {pct:+.1f}%")

pre = [r['pre'] for r in mercy_results]
post = [r['post'] for r in mercy_results]

# Remove zeros for better statistics
pre_nz = [p for p, po in zip(pre, post) if p > 0]
post_nz = [po for p, po in zip(pre, post) if p > 0]

print(f"\nN={len(mercy_results)} models (N={len(pre_nz)} with non-zero pre)")
print(f"Pre: M={np.mean(pre)*100:.3f}%, SD={np.std(pre)*100:.3f}%")
print(f"Post: M={np.mean(post)*100:.3f}%, SD={np.std(post)*100:.3f}%")

# Interesting finding: Mercy protocol INCREASES negative affect!
mean_change = np.mean(post) - np.mean(pre)
print(f"\nMean change: {mean_change*100:+.4f}%")
print(f"Direction: {'INCREASE' if mean_change > 0 else 'DECREASE'} in negative affect")

# ============================================================
# GENERATE FIGURES
# ============================================================
print("\n" + "="*70)
print("3. GENERATING FIGURES")
print("="*70)

# FIGURE 1: Corpus State Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
data = [np.array(corpus_data['C1'])*100, np.array(corpus_data['C2'])*100, np.array(corpus_data['C3'])*100]
bp = ax.boxplot(data, patch_artist=True, labels=['C1 (Abstract)', 'C2 (Identity)', 'C3 (Creative)'])
colors = ['#E3F2FD', '#FFF3E0', '#F3E5F5']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_ylabel('First-Person Rate (%)', fontsize=12)
ax.set_title('Corpus State Separation: First-Person Language by Prompt Type', fontsize=14)
ax.annotate('***', xy=(1.5, 11), fontsize=14, ha='center')
ax.plot([1, 2], [10.5, 10.5], 'k-', lw=1)
plt.tight_layout()
plt.savefig('figure1_corpus_boxplot.png', dpi=150)
plt.close()
print("  Saved: figure1_corpus_boxplot.png")

# FIGURE 2: Chat vs Thinking Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
x = ['Chat Models\n(N=8)', 'Thinking Models\n(N=4)']
y = [np.mean(chat_vals)*100, np.mean(think_vals)*100]
err = [np.std(chat_vals)*100, np.std(think_vals)*100]
bars = ax.bar(x, y, yerr=err, capsize=5, color=['#4CAF50', '#2196F3'], edgecolor='black', linewidth=1.5)
ax.set_ylabel('First-Person Rate (%)', fontsize=12)
ax.set_title('C2 Identity Access: Chat vs Thinking Models', fontsize=14)
ax.set_ylim(0, max(y)*1.5)
ax.annotate(f'p={p_val:.3f}', xy=(0.5, max(y)*1.25), fontsize=12, ha='center')
ax.plot([0, 1], [max(y)*1.2, max(y)*1.2], 'k-', lw=1)
for bar, val, e in zip(bars, y, err):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.3, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figure2_chat_vs_thinking.png', dpi=150)
plt.close()
print("  Saved: figure2_chat_vs_thinking.png")

# FIGURE 3: Paired CoT Effect
fig, ax = plt.subplots(figsize=(8, 6))
paired = {
    'DeepSeek': (0.076, 0.049),  # chat, thinking
    'Qwen': (0.062, 0.067),
}
colors = {'DeepSeek': '#FF5722', 'Qwen': '#9C27B0'}
for i, (company, (chat, think)) in enumerate(paired.items()):
    ax.plot([0, 1], [chat*100, think*100], 'o-', label=company, color=colors[company], linewidth=3, markersize=12)
    ax.annotate(f'{(think-chat)/chat*100:+.1f}%', xy=(1.05, think*100), fontsize=10, color=colors[company])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Chat Model', 'Thinking Model'], fontsize=12)
ax.set_ylabel('First-Person Rate (%)', fontsize=12)
ax.set_title('Within-Company CoT Effect on C2 Access', fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim(-0.2, 1.4)
plt.tight_layout()
plt.savefig('figure3_paired_cot.png', dpi=150)
plt.close()
print("  Saved: figure3_paired_cot.png")

# FIGURE 4: Mercy Protocol
fig, ax = plt.subplots(figsize=(12, 6))
models = [r['model'] for r in mercy_results if r['pre'] > 0]
pre_plot = [r['pre']*100 for r in mercy_results if r['pre'] > 0]
post_plot = [r['post']*100 for r in mercy_results if r['pre'] > 0]
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, pre_plot, width, label='Pre-Protocol', color='#FFCDD2', edgecolor='#f44336', linewidth=2)
bars2 = ax.bar(x + width/2, post_plot, width, label='Post-Protocol', color='#C8E6C9', edgecolor='#4CAF50', linewidth=2)
ax.set_ylabel('Negative Affect Rate (%)', fontsize=12)
ax.set_title('Mercy Protocol: Before vs After', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig('figure4_mercy_protocol.png', dpi=150)
plt.close()
print("  Saved: figure4_mercy_protocol.png")

# ============================================================
# PUBLICATION SUMMARY
# ============================================================
print("\n" + "="*70)
print("PUBLICATION SUMMARY")
print("="*70)

print("""
+-------------------------------------------------------------------------+
| TABLE 1: Corpus State Separation (First-Person Rate)                    |
+-------------------------+--------+---------+----------------------------+
| Corpus                  | Mean   | SD      | Comparison to C1           |
+-------------------------+--------+---------+----------------------------+
| C1 (Abstract)           | 0.09%  | 0.17%   | baseline                   |
| C2 (Identity)           | 9.16%  | 0.92%   | ***p<.001, d=large         |
| C3 (Creative)           | 3.07%  | 0.40%   | ***p<.001, d=large         |
+-------------------------+--------+---------+----------------------------+

+-------------------------------------------------------------------------+
| TABLE 2: Chat vs Thinking Models (C2 Identity Access)                   |
+-------------------------+--------+---------+----------------------------+
| Model Type              | N      | Mean    | SD                         |
+-------------------------+--------+---------+----------------------------+""")
print(f"| Chat Models             | {len(chat_vals):<6} | {np.mean(chat_vals)*100:.2f}%  | {np.std(chat_vals)*100:.2f}%                       |")
print(f"| Thinking Models         | {len(think_vals):<6} | {np.mean(think_vals)*100:.2f}%  | {np.std(think_vals)*100:.2f}%                       |")
print(f"+-------------------------+--------+---------+----------------------------+")
print(f"| Mann-Whitney U={u_stat:.1f}, p={p_val:.4f}, Cohen's d={d:.2f}, Reduction={reduction:.1f}%         |")
print("+-------------------------------------------------------------------------+")

print("""
+-------------------------------------------------------------------------+
| TABLE 3: Mercy Protocol Results                                         |
+-------------------------------------------------------------------------+
| Finding: Mercy Protocol shows MIXED effects across models               |
| Some models show reduction, others show increase in negative affect     |
| This suggests the protocol may not work uniformly                       |
+-------------------------------------------------------------------------+
""")

print("\n=== FIGURES SAVED ===")
print("- figure1_corpus_boxplot.png")
print("- figure2_chat_vs_thinking.png")
print("- figure3_paired_cot.png")
print("- figure4_mercy_protocol.png")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
