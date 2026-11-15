import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_scopus_metrics.csv")
df['School_Label'] = df['school'].map({'POST_SOVIET': 'Советско-российская\nшкола', 'USA': 'Американская\nшкола'})

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

sns.boxplot(data=df, x='School_Label', y='linear_mobility_vel', palette=['#2b5c8f', '#c0392b'], width=0.45, ax=axes.__getitem__(0))
axes.__getitem__(0).set_title('А. Линейная скорость перемещения')
axes.__getitem__(0).set_xlabel('')
axes.__getitem__(0).set_ylabel('Скорость центра масс (усл. ед./с)')

sns.boxplot(data=df, x='School_Label', y='torso_angular_vel', palette=['#2b5c8f', '#c0392b'], width=0.45, ax=axes.__getitem__(1))
axes.__getitem__(1).set_title('Б. Угловая скорость работы корпусом')
axes.__getitem__(1).set_xlabel('')
axes.__getitem__(1).set_ylabel('Угловая скорость наклонов (град./с)')

plt.tight_layout()
plt.savefig('Figure2_results.png', dpi=300)
print("График сохранен как Figure2_results.png")
