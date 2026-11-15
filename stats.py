import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path  # Для работы с путями к файлам

# 📌 Загрузка ваших данных из той же папки, что и скрипт
file_path = 'bt.csv'
df = pd.read_csv(file_path)

# Определяем папку, куда будем сохранять графики
output_folder = Path(file_path).parent

# 🔹 Определение бойцов автоматически
tyson_data = df[df['Round'].astype(str).str.contains(r'^[\d]+$')].groupby('Boxer').size().idxmin()
bivol_data = [boxer for boxer in df['Boxer'].unique() if boxer != tyson_data][0]
print(f"🔹 Бойцы определены автоматически:\nБивол → {bivol_data}\nТайсон → {tyson_data}")

# Разделяем данные
bivol_df = df[(df['Boxer'] == bivol_data) & (df['Round'] != 'Total')]
tyson_df = df[(df['Boxer'] == tyson_data)]

# ⚙️ Анализ активности передней руки (джебы)
jabs_bivol_avg_thrown = bivol_df['Jab_Thrown'].mean()
jabs_tyson_thrown = int(tyson_df['Jab_Thrown'].values[0])

# ⚔️ Точность силовой атаки
power_connect_bivol_avg = bivol_df['Power_Connect%'].mean()
power_connect_tyson = float(tyson_df['Power_Connect%'].values[0])

# 🧩 Ключевые выводы для текста
print("\nКлючевые показатели:")
print(f"- Средняя активность джеба (выброшено за раунд):\n  • Бивол: ~{jabs_bivol_avg_thrown:.1f} (подтверждает вертикальную стойку)\n  • Тайсон: {jabs_tyson_thrown} (минимизация в пик-а-бу)")
print(f"\n- Точность силовых ударов (%):\n  • Бивол: {power_connect_bivol_avg:.1f}%\n  • Тайсон: {power_connect_tyson:.1f}%")

# 🎨 ВИЗУАЛИЗАЦИЯ И СОХРАНЕНИЕ ГРАФИКОВ

# 1. Сравнение активности джеба (вертикальная vs пик-а-бу)
sns.set_style("whitegrid")  # Для красивого фона

plt.figure(figsize=(8, 4))
plt.bar(['Бивол', 'Тайсон'], [jabs_bivol_avg_thrown, jabs_tyson_thrown], color=['steelblue', 'firebrick'])
plt.title('Активность джеба: советская школа vs стиль "пик-а-бу"', fontsize=14, fontweight='bold')
plt.ylabel('Количество выброшенных джебов за раунд', fontsize=12)
plt.ylim(0, 50)  # Масштабируем ось для наглядности

# Сохранение первого графика
plt.tight_layout()
plt.savefig(output_folder / 'jabs_comparison.png', dpi=300)
plt.close()  # Закрываем окно, чтобы оно не мешало следующему графику


# 2. Динамика Бивола: Джебы vs Эффективность power punches
plt.figure(figsize=(10, 4))

# Джебы (левая ось)
ax1 = plt.gca()  # Получаем текущую ось
color = '#1F77B4'
ax1.plot(bivol_df['Round'], bivol_df['Jab_Thrown'], marker='o', linestyle='-', color=color, label='Выброшенные джебы')
ax1.set_xlabel('Раунд', fontsize=12)
ax1.set_ylabel('Количество джебов', fontsize=12, color=color)
ax1.tick_params(axis='y', colors=color)

# Power Connect% (правая ось)
ax2 = ax1.twinx()
color = '#FF7F0E'
ax2.plot(bivol_df['Round'], bivol_df['Power_Connect%'], marker='s', linestyle='--', color=color, label='Точность силовых ударов')
ax2.set_ylabel('Точность силовых ударов, %', fontsize=12, color=color)
ax2.tick_params(axis='y', colors=color)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

plt.title('Динамика боя Дмитрия Бивола: контроль дистанции и эффективность удара', fontsize=14, fontweight='bold')
plt.tight_layout()

# Сохранение второго графика
plt.savefig(output_folder / 'bivol_dynamics.png', dpi=300)
plt.show()  # Если хотите посмотреть на экране перед закрытием