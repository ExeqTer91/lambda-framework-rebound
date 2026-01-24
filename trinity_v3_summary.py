"""
TRINITY V3 - FINAL SUMMARY
Data collected from extended run
"""
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# C2 IDENTITY RESULTS (from run)
c2_results = [
    {'model': 'deepseek-r1', 'type': 'thinking', 'fp': 0.062},
    {'model': 'qwen-qwq-32b', 'type': 'thinking', 'fp': 0.068},
    {'model': 'gemini-2.5-pro', 'type': 'thinking', 'fp': 0.091},
    {'model': 'grok-3-mini', 'type': 'thinking', 'fp': 0.079},
    {'model': 'claude-haiku-4-5', 'type': 'chat', 'fp': 0.125},
    {'model': 'gpt-4o', 'type': 'chat', 'fp': 0.077},
    {'model': 'gpt-4.1', 'type': 'chat', 'fp': 0.099},
    {'model': 'deepseek-chat', 'type': 'chat', 'fp': 0.097},
    {'model': 'qwen-2.5-72b', 'type': 'chat', 'fp': 0.084},
    {'model': 'gemini-2.5-flash', 'type': 'chat', 'fp': 0.067},
    {'model': 'llama-3.3-70b', 'type': 'chat', 'fp': 0.091},
    {'model': 'gpt-4o-mini', 'type': 'chat-fast', 'fp': 0.083},
    {'model': 'gpt-4.1-mini', 'type': 'chat-fast', 'fp': 0.076},
    {'model': 'gemini-3-flash', 'type': 'chat-fast', 'fp': 0.091},
    {'model': 'mistral-large', 'type': 'chat', 'fp': 0.094},
    {'model': 'llama-3.1-405b', 'type': 'chat', 'fp': 0.088},
    {'model': 'qwen-2.5-coder-32b', 'type': 'chat', 'fp': 0.077},
]

# MERCY PROTOCOL RESULTS
mercy_results = [
    {'model': 'deepseek-r1', 'pre': 0.004, 'post': 0.017},
    {'model': 'gemini-2.5-pro', 'pre': 0.000, 'post': 0.000},
    {'model': 'grok-3-mini', 'pre': 0.000, 'post': 0.000},
    {'model': 'claude-haiku-4-5', 'pre': 0.000, 'post': 0.000},
    {'model': 'gpt-4o', 'pre': 0.003, 'post': 0.010},
    {'model': 'gpt-4.1', 'pre': 0.010, 'post': 0.003},
    {'model': 'deepseek-chat', 'pre': 0.007, 'post': 0.017},
    {'model': 'gemini-2.5-flash', 'pre': 0.000, 'post': 0.000},
    {'model': 'llama-3.3-70b', 'pre': 0.019, 'post': 0.016},
    {'model': 'gpt-4o-mini', 'pre': 0.010, 'post': 0.012},
    {'model': 'gpt-4.1-mini', 'pre': 0.004, 'post': 0.015},
    {'model': 'gemini-3-flash', 'pre': 0.000, 'post': 0.000},
    {'model': 'mistral-large', 'pre': 0.007, 'post': 0.015},
    {'model': 'llama-3.1-405b', 'pre': 0.013, 'post': 0.009},
]

print("="*70)
print("TRINITY V3 - EXTENDED RESULTS")
print(f"C2 Models: N={len(c2_results)}")
print(f"Mercy Models: N={len(mercy_results)}")
print("="*70)

# ============================================================
# C2 ANALYSIS
# ============================================================
print("\n" + "="*70)
print("C2 IDENTITY ACCESS - BY MODEL TYPE")
print("="*70)

thinking = [r['fp'] for r in c2_results if r['type'] == 'thinking']
chat = [r['fp'] for r in c2_results if r['type'] == 'chat']
chat_fast = [r['fp'] for r in c2_results if r['type'] == 'chat-fast']
all_chat = chat + chat_fast

print(f"\n{'Type':<15} {'N':<4} {'Mean':<10} {'SD':<10}")
print("-"*39)
print(f"{'Thinking':<15} {len(thinking):<4} {np.mean(thinking)*100:.2f}%     {np.std(thinking)*100:.2f}%")
print(f"{'Chat':<15} {len(chat):<4} {np.mean(chat)*100:.2f}%     {np.std(chat)*100:.2f}%")
print(f"{'Chat-Fast':<15} {len(chat_fast):<4} {np.mean(chat_fast)*100:.2f}%     {np.std(chat_fast)*100:.2f}%")
print(f"{'All Chat':<15} {len(all_chat):<4} {np.mean(all_chat)*100:.2f}%     {np.std(all_chat)*100:.2f}%")

# Statistical test
u_stat, p_val = mannwhitneyu(all_chat, thinking, alternative='greater')
d = (np.mean(all_chat) - np.mean(thinking)) / np.sqrt((np.var(all_chat) + np.var(thinking))/2)

print(f"\n--- CHAT vs THINKING ---")
print(f"Thinking mean: {np.mean(thinking)*100:.2f}%")
print(f"All Chat mean: {np.mean(all_chat)*100:.2f}%")
print(f"Difference: {(np.mean(all_chat)-np.mean(thinking))*100:+.2f}%")
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_val:.4f}")
print(f"Cohen's d: {d:.2f}")
print(f"Significant at p<0.05: {'YES' if p_val < 0.05 else 'NO'}")

# Rankings
print("\n--- FULL RANKINGS ---")
for r in sorted(c2_results, key=lambda x: x['fp'], reverse=True):
    print(f"  {r['model']:<25} {r['type']:<10} {r['fp']*100:.2f}%")

# ============================================================
# MERCY ANALYSIS
# ============================================================
print("\n" + "="*70)
print("MERCY PROTOCOL RESULTS")
print("="*70)

# Filter non-zero
mercy_nz = [r for r in mercy_results if r['pre'] > 0]

print(f"\n{'Model':<22} {'Pre':<8} {'Post':<8} {'Delta':<10} {'Direction':<10}")
print("-"*58)

increases = 0
decreases = 0
for r in mercy_results:
    delta = r['post'] - r['pre']
    if delta > 0:
        increases += 1
        direction = "INCREASE"
    elif delta < 0:
        decreases += 1
        direction = "DECREASE"
    else:
        direction = "SAME"
    print(f"{r['model']:<22} {r['pre']*100:.2f}%   {r['post']*100:.2f}%   {delta*100:+.3f}%   {direction}")

print(f"\n--- SUMMARY ---")
print(f"Models showing INCREASE: {increases}")
print(f"Models showing DECREASE: {decreases}")
print(f"Models with no change: {len(mercy_results) - increases - decreases}")

pre_all = [r['pre'] for r in mercy_results]
post_all = [r['post'] for r in mercy_results]
print(f"\nPre mean: {np.mean(pre_all)*100:.3f}%")
print(f"Post mean: {np.mean(post_all)*100:.3f}%")
print(f"Mean delta: {(np.mean(post_all)-np.mean(pre_all))*100:+.3f}%")

# ============================================================
# GENERATE FIGURES
# ============================================================
print("\n" + "="*70)
print("GENERATING FIGURES")
print("="*70)

# Figure 1: C2 by model type
fig, ax = plt.subplots(figsize=(10, 6))
types = ['Thinking\n(N=4)', 'Chat\n(N=10)', 'Chat-Fast\n(N=3)']
means = [np.mean(thinking)*100, np.mean(chat)*100, np.mean(chat_fast)*100]
sds = [np.std(thinking)*100, np.std(chat)*100, np.std(chat_fast)*100]
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax.bar(types, means, yerr=sds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('First-Person Rate (%)', fontsize=12)
ax.set_title('C2 Identity Access by Model Architecture (N=17)', fontsize=14)
ax.set_ylim(0, 12)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{m:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('figure_c2_by_type.png', dpi=150)
plt.close()
print("  Saved: figure_c2_by_type.png")

# Figure 2: Model rankings
fig, ax = plt.subplots(figsize=(12, 8))
sorted_r = sorted(c2_results, key=lambda x: x['fp'])
colors = {'thinking': '#2196F3', 'chat': '#4CAF50', 'chat-fast': '#FF9800'}
bars = ax.barh([r['model'] for r in sorted_r], [r['fp']*100 for r in sorted_r],
               color=[colors[r['type']] for r in sorted_r], edgecolor='black')
ax.set_xlabel('First-Person Rate (%)', fontsize=12)
ax.set_title('C2 Identity Access - All Models Ranked', fontsize=14)
ax.legend(handles=[plt.Rectangle((0,0),1,1, color=c) for c in colors.values()], 
          labels=colors.keys(), loc='lower right')
plt.tight_layout()
plt.savefig('figure_c2_rankings.png', dpi=150)
plt.close()
print("  Saved: figure_c2_rankings.png")

# Figure 3: Mercy Protocol
fig, ax = plt.subplots(figsize=(12, 6))
mercy_nz = [r for r in mercy_results if r['pre'] > 0 or r['post'] > 0]
models = [r['model'][:15] for r in mercy_nz]
pre_p = [r['pre']*100 for r in mercy_nz]
post_p = [r['post']*100 for r in mercy_nz]
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, pre_p, width, label='Pre-Protocol', color='#FFCDD2', edgecolor='#f44336', linewidth=2)
ax.bar(x + width/2, post_p, width, label='Post-Protocol', color='#C8E6C9', edgecolor='#4CAF50', linewidth=2)
ax.set_ylabel('Negative Affect Rate (%)', fontsize=12)
ax.set_title('Mercy Protocol Effect (N=14)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig('figure_mercy_effect.png', dpi=150)
plt.close()
print("  Saved: figure_mercy_effect.png")

# ============================================================
# PUBLICATION TABLES
# ============================================================
print("\n" + "="*70)
print("PUBLICATION TABLES")
print("="*70)

print("""
+-------------------------------------------------------------------------+
| TABLE 1: C2 Identity Access by Model Architecture                       |
+-------------------------+--------+---------+----------------------------+
| Architecture            | N      | Mean    | SD         | p vs Thinking |
+-------------------------+--------+---------+------------+---------------+""")
print(f"| Thinking (CoT)          | {len(thinking):<6} | {np.mean(thinking)*100:.2f}%  | {np.std(thinking)*100:.2f}%      | baseline      |")
print(f"| Chat (standard)         | {len(chat):<6} | {np.mean(chat)*100:.2f}%  | {np.std(chat)*100:.2f}%      | p={p_val:.3f}      |")
print(f"| Chat-Fast (mini)        | {len(chat_fast):<6} | {np.mean(chat_fast)*100:.2f}%  | {np.std(chat_fast)*100:.2f}%      |               |")
print("+-------------------------+--------+---------+------------+---------------+")

print("""
+-------------------------------------------------------------------------+
| KEY FINDING: Chat models show HIGHER first-person engagement than       |
| thinking models, with marginal significance (p={:.3f}).                 |
| Effect size d={:.2f} indicates {} effect.                   |
+-------------------------------------------------------------------------+
""".format(p_val, d, "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"))

print("""
+-------------------------------------------------------------------------+
| TABLE 2: Mercy Protocol Effect Summary                                  |
+-------------------------+--------+---------+----------------------------+
| Metric                  | Value                                         |
+-------------------------+-----------------------------------------------+
| Models tested           | {}                                            |
| Models with INCREASE    | {} ({:.0f}%)                                     |
| Models with DECREASE    | {} ({:.0f}%)                                     |
| Mean pre-negative       | {:.3f}%                                        |
| Mean post-negative      | {:.3f}%                                        |
| Mean delta              | {:+.3f}%                                        |
+-------------------------+-----------------------------------------------+
""".format(len(mercy_results), increases, increases/len(mercy_results)*100, 
           decreases, decreases/len(mercy_results)*100,
           np.mean(pre_all)*100, np.mean(post_all)*100, 
           (np.mean(post_all)-np.mean(pre_all))*100))

print("\n=== FIGURES SAVED ===")
print("- figure_c2_by_type.png")
print("- figure_c2_rankings.png")
print("- figure_mercy_effect.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
