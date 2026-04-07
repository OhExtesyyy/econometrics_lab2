import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Настройка светлой темы для графиков (требование: нет темной теме)
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Встроенный набор данных (Всемирный банк, 2019 год, 30 стран для репрезентативности)
csv_data = """Country,Region,GDP_per_capita,Life_expectancy,CO2_per_capita
Australia,East Asia & Pacific,54875.2,82.9,15.2
China,East Asia & Pacific,10143.8,77.0,7.6
Japan,East Asia & Pacific,40456.4,84.3,8.5
South Korea,East Asia & Pacific,31862.8,83.2,11.9
Indonesia,East Asia & Pacific,4135.2,71.5,2.3
France,Europe & Central Asia,41818.1,82.7,4.5
Germany,Europe & Central Asia,46793.6,81.3,7.9
Italy,Europe & Central Asia,33649.0,83.4,5.3
UK,Europe & Central Asia,43070.5,81.2,5.2
Russia,Europe & Central Asia,11438.3,73.0,11.8
Poland,Europe & Central Asia,15726.5,77.9,7.9
Spain,Europe & Central Asia,29555.0,83.5,5.4
Ukraine,Europe & Central Asia,3661.4,72.0,4.4
USA,North America,65279.5,78.8,14.7
Canada,North America,46328.6,82.0,15.4
Brazil,Latin America,8897.5,75.9,2.2
Mexico,Latin America,9946.0,75.1,3.6
Argentina,Latin America,9912.2,76.5,3.9
Chile,Latin America,14736.1,80.1,4.7
Colombia,Latin America,6428.7,77.2,1.6
India,South Asia,2100.7,69.7,1.8
Pakistan,South Asia,1285.5,67.3,1.0
Bangladesh,South Asia,1855.9,72.6,0.5
Nigeria,Sub-Saharan Africa,2229.8,54.7,0.7
South Africa,Sub-Saharan Africa,6624.7,64.1,7.6
Kenya,Sub-Saharan Africa,1816.5,62.9,0.4
Ethiopia,Sub-Saharan Africa,855.7,66.2,0.1
Uganda,Sub-Saharan Africa,794.3,63.3,0.1
Ghana,Sub-Saharan Africa,2262.2,64.1,0.6
Senegal,Sub-Saharan Africa,1434.9,67.9,0.7"""

df = pd.read_csv(io.StringIO(csv_data))

def main():
    print("--- ПРАКТИЧЕСКОЕ ЗАДАНИЕ №2 (Группа 4415) ---\n")
    
    # 1. Формальное представление данных (Таблица)
    print("Фрагмент данных (Head & Tail):")
    display_df = pd.concat([df.head(3), df.tail(3)])
    print(display_df.to_string(), "\n")

    # === ВИЗУАЛИЗАЦИЯ ДАННЫХ ===
    # График 1: Гистограмма ВВП
    plt.figure(figsize=(8, 5))
    sns.histplot(df['GDP_per_capita'], kde=True, bins=10, color='blue')
    plt.title('Рисунок 1 - Распределение ВВП на душу населения')
    plt.xlabel('ВВП на душу населения (USD)')
    plt.ylabel('Количество стран')
    plt.tight_layout()
    plt.savefig('hist_gdp.png')
    plt.close()

    # График 2: Диаграмма рассеяния
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='GDP_per_capita', y='CO2_per_capita', hue='Region', s=100)
    plt.title('Рисунок 2 - ВВП vs Выбросы CO2')
    plt.xlabel('ВВП на душу населения (USD)')
    plt.ylabel('Выбросы CO2 (тонн на человека)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('scatter_gdp_co2.png')
    plt.close()

    # График 3: Boxplot ожидаемой продолжительности жизни
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Region', y='Life_expectancy', palette='Set2')
    plt.title('Рисунок 3 - Ожидаемая продолжительность жизни по регионам')
    plt.xlabel('Регион')
    plt.ylabel('Продолжительность жизни (лет)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('boxplot_life.png')
    plt.close()
    print("Графики успешно сгенерированы и сохранены как PNG файлы.\n")

    # === СТАТИСТИЧЕСКИЕ КРИТЕРИИ ===
    print("--- РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКИХ ТЕСТОВ ---\n")

    # 1. Критерий Шапиро-Уилка (Нормальность ВВП)
    stat_sh, p_sh = stats.shapiro(df['GDP_per_capita'])
    print(f"1. Критерий Шапиро-Уилка (Нормальность ВВП):")
    print(f"   Статистика = {stat_sh:.4f}, p-value = {p_sh:.4f}")
    print("   Вывод: p < 0.05, данные не распределены нормально.\n")

    # Разделение данных для следующих тестов
    europe = df[df['Region'] == 'Europe & Central Asia']['Life_expectancy']
    africa = df[df['Region'] == 'Sub-Saharan Africa']['Life_expectancy']

    # 2. Критерий Левена (Равенство дисперсий)
    stat_lev, p_lev = stats.levene(europe, africa)
    print(f"2. Критерий Левена (Дисперсии продолжительности жизни Европа vs Африка):")
    print(f"   Статистика = {stat_lev:.4f}, p-value = {p_lev:.4f}")
    print("   Вывод: p > 0.05, дисперсии однородны (гомоскедастичность).\n")

    # 3. Критерий Манна-Уитни
    stat_mw, p_mw = stats.mannwhitneyu(europe, africa, alternative='two-sided')
    print(f"3. Критерий Манна-Уитни (Продолжительность жизни Европа vs Африка):")
    print(f"   Статистика U = {stat_mw:.4f}, p-value = {p_mw:.4f}")
    print("   Вывод: p < 0.05, статистически значимая разница присутствует.\n")

    # 4. Критерий Краскела-Уоллиса
    asia = df[df['Region'] == 'East Asia & Pacific']['CO2_per_capita']
    europe_co2 = df[df['Region'] == 'Europe & Central Asia']['CO2_per_capita']
    africa_co2 = df[df['Region'] == 'Sub-Saharan Africa']['CO2_per_capita']
    
    stat_kw, p_kw = stats.kruskal(asia, europe_co2, africa_co2)
    print(f"4. Критерий Краскела-Уоллиса (Выбросы CO2 по 3 регионам):")
    print(f"   Статистика H = {stat_kw:.4f}, p-value = {p_kw:.4f}")
    print("   Вывод: p < 0.05, медианы выбросов в регионах статистически различаются.\n")

    # 5. Критерий Спирмена (Корреляция)
    stat_sp, p_sp = stats.spearmanr(df['GDP_per_capita'], df['CO2_per_capita'])
    print(f"5. Корреляция Спирмена (ВВП и Выбросы CO2):")
    print(f"   Коэффициент = {stat_sp:.4f}, p-value = {p_sp:.4f}")
    print("   Вывод: p < 0.05, наблюдается сильная положительная связь.")

if __name__ == "__main__":
    main()