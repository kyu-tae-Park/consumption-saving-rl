"""Regenerate final figures from the archived summary tables; no RL training needed.

Run: python scripts/plot_results.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/'results'/'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(ROOT/'results'/'final_summary.csv')
seeds = pd.read_csv(ROOT/'results'/'final_seed_results.csv')
plt.rcParams.update({'font.size':11, 'savefig.dpi':170})

# Performance: points show independent training seeds, not independent income paths.
fig,ax=plt.subplots(figsize=(8.2,5.2))
for i,(name,color) in enumerate([('DQN','#1976a3'),('PPO','#d98e26')]):
    vals=seeds.loc[seeds.algorithm==name,'test_mean_utility'].to_numpy()
    ax.scatter(np.full(len(vals),i)+np.linspace(-.08,.08,len(vals)), vals,
               color=color, alpha=.85, label=f'{name}: training seeds')
    ax.plot([i-.18,i+.18],[vals.mean(), vals.mean()],color=color, linewidth=2.7)
for name,style,color in [('DP-Matched','--','#2f855a'),('Fixed 50%',':','#595959')]:
    y=float(summary.loc[summary.Model==name,'Mean Utility'].iloc[0]); ax.axhline(y,linestyle=style,color=color,label=name)
ax.set_xticks([0,1],['DQN','PPO']);ax.set_ylabel('Discounted utility (higher is better)')
ax.set_title('Test welfare across five training seeds');ax.legend(loc='best',fontsize=9)
fig.tight_layout();fig.savefig(FIGURES/'seed_welfare.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,5.0))
models=['DP-Matched','DQN','Fixed 50%','PPO']
vals=[float(summary.loc[summary.Model==m,'Mean Utility'].iloc[0]) for m in models]
colors=['#2f855a','#1976a3','#7d8590','#d98e26']
bars=ax.bar(models,vals,color=colors,width=.64)
ax.set_ylim(-39,-35.5); ax.set_ylabel('Mean discounted utility')
ax.set_title('Final five-seed comparison (model average)')
for bar,val in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,val+.035,f'{val:.3f}',ha='center',fontsize=10)
fig.tight_layout();fig.savefig(FIGURES/'model_welfare.png');plt.close(fig)

sub=summary[summary.Model.isin(['DQN','PPO'])].set_index('Model')
fig,ax=plt.subplots(figsize=(7.4,4.8))
x=np.arange(2)
for i,col in enumerate(['Mean Exact','Mean Within 5pp','Mean Within 10pp']):
    ax.bar(x+(i-1)*.24,sub.loc[['DQN','PPO'],col].values*100,width=.23,label=col.replace('Mean ',''))
ax.set_xticks(x,['DQN','PPO']);ax.set_ylim(0,105);ax.set_ylabel('Grid states (%)')
ax.set_title('Policy agreement with DP-Matched');ax.legend()
fig.tight_layout();fig.savefig(FIGURES/'policy_agreement.png');plt.close(fig)

# Figures below reproduce the supplied comparative-statics simulation numbers (not new simulations).
sigma=np.array([1.5,2.0,3.0]); a=np.array([.514495,.771886,1.146777]);z=np.array([.330439,.424712,.521136])
fig,ax=plt.subplots(figsize=(7.4,4.8))
ax.plot(sigma,a,marker='o',markersize=7)
ax.set_xticks(sigma);ax.set_xlabel('CRRA curvature (sigma)');ax.set_ylabel('Long-run mean assets')
ax.set_title('Risk aversion and precautionary asset accumulation')
fig.tight_layout();fig.savefig(FIGURES/'sigma_assets.png');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.4,4.8))
ax.plot(sigma,z*100,marker='o',markersize=7)
ax.set_xticks(sigma);ax.set_xlabel('CRRA curvature (sigma)');ax.set_ylabel('Long-run mean saving share (%)')
ax.set_title('Risk aversion and saving share')
fig.tight_layout();fig.savefig(FIGURES/'sigma_saving.png');plt.close(fig)

print('Created',len(list(FIGURES.glob('*.png'))),'figures in',FIGURES)
