import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


# 1. Открыть датасет из ЛР 2 через pandas
df = pd.read_csv('sales_data.csv')

# 2. Вывести все продажи в штате Montana
montana_sales = df[df['State'] == 'Montana']

print(f"Продажи в штате Montana:\n{montana_sales}")

# 3. Вывести среднюю выручку с продаж клиентам какой – либо возрастной группы
age_filter = df[(df['Customer_Age'] >= 18) & (df['Customer_Age'] <= 44)]
age_revenue = age_filter['Revenue'].mean()

print(f"Средняя выручка с продаж клиентам в возрасте от 18 до 44 лет: {age_revenue:.2f}")

# 4. Вывести графики по продажам в год и месяц
sales_by_year = df.groupby('Year')['Revenue'].sum()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sales_by_year.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Продажи по годам')
plt.xlabel('Год')
plt.ylabel('Выручка')
plt.xticks(rotation=0)


month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

df_temp_month = df.copy()
df_temp_month['Month_num'] = df_temp_month['Month'].map(month_map)
sales_by_month = df_temp_month.groupby('Month_num')['Revenue'].sum()
num_to_month = {num: month for month, num in month_map.items()}
sales_by_month.index = sales_by_month.index.map(num_to_month)

plt.subplot(1, 2, 2)
sales_by_month.plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Продажи по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 5. Вывести график соотношений Unit_Cost и Unit_Price
plt.figure(figsize=(10, 5))
plt.scatter(df['Unit_Cost'], df['Unit_Price'], alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.3)
min_val = min(df['Unit_Cost'].min(), df['Unit_Price'].min())
max_val = max(df['Unit_Cost'].max(), df['Unit_Price'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Цена = Себестоимость')
plt.title('Соотношение себестоимости и цены')
plt.xlabel('Себестоимость')
plt.ylabel('Цена')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 6. Построить линейную регрессию методом наименьших квадратов
df_clean = df[['Unit_Cost', 'Unit_Price']].dropna()
x = df_clean['Unit_Cost']
y = df_clean['Unit_Price']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line + intercept

plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.3)
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Регрессия: y = {slope:.2f}x + {intercept:.2f}')

plt.title('Линейная регрессия: Цены от себестоимости')
plt.xlabel('Себестоимость')
plt.ylabel('Цена')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"Уравнение регрессии: Цена = {slope:.4f} * себестоимость + {intercept:.4f}\n"
      f"Коэффициент детерминации (R²): {r_squared:.4f}\n"
      f"Корреляция (r): {r_value:.4f}")

# 7. Сравнить с линейной регрессией из пакета scikit-learn
df_cleans = df[['Unit_Cost', 'Unit_Price']].dropna()
x = df_cleans['Unit_Cost'].values
y = df_cleans['Unit_Price'].values

# scipy
slope_scipy, intercept_scipy, r_scipy, p_val, std_err = stats.linregress(x, y)
r2_scipy = r_scipy ** 2

# scikit-learn
X = x.reshape(-1, 1)
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)
slope_sklearn = model_sklearn.coef_[0]
intercept_sklearn = model_sklearn.intercept_
r2_sklearn = model_sklearn.score(X, y)

# Вывод результатов
print("Сравнение линейных регрессий")
print(f"scipy:")
print(f"  Наклон (slope):     {slope_scipy:.6f}")
print(f"  Свободный член:     {intercept_scipy:.6f}")
print(f"  R²:                 {r2_scipy:.6f}\n")

print(f"scikit-learn:")
print(f"  Наклон (coef_):     {slope_sklearn:.6f}")
print(f"  Свободный член:     {intercept_sklearn:.6f}")
print(f"  R² (score):         {r2_sklearn:.6f}\n")

print(f"Разница в наклоне:     {abs(slope_scipy - slope_sklearn):.2e}")
print(f"Разница в intercept:   {abs(intercept_scipy - intercept_sklearn):.2e}")
print(f"Разница в R²:          {abs(r2_scipy - r2_sklearn):.2e}")

# 8. Построить график продаж по времени
df_temp_date = df.copy()
df_temp_date['Date'] = pd.to_datetime(df_temp_date['Date'])
df_temp_date = df_temp_date.sort_values('Date').reset_index(drop=True)
daily_sales = df_temp_date.groupby('Date')['Revenue'].sum()

plt.figure(figsize=(10, 5))
plt.plot(daily_sales.index, daily_sales.values, color='steelblue', linewidth=1)
plt.title('Продажи по времени')
plt.xlabel('Дата')
plt.ylabel('Выручка')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

