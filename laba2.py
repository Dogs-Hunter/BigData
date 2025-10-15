import pandas as pd
import matplotlib.pyplot as plt

# 1. Открыть датасет через пандас
df = pd.read_csv('sales_data.csv')

# 2. Показать первые и последние 5 элементов дата сета
print(f"Первые 5 строк: \n{df.head()}"
      f"\nПоследние 5 строк: \n{df.tail()}")

# 3. Вывести информацию по дата сету и типам данных
print(f"\nИнформация по датасету: \n{df.info()}")

# 4. Указать какая метрика уже есть в дата сете
print(f"\nМетрики: \n{df.columns.tolist()}")

# 5.	Для поля Order Quantity найти количество элементов,
#          среднее значение, минимум, максимум, стандартное
#          отклонение и медиану
count = df['Order_Quantity'].count()
mean = df['Order_Quantity'].mean()
min_val = df['Order_Quantity'].min()
max_val = df['Order_Quantity'].max()
std = df['Order_Quantity'].std()
median = df['Order_Quantity'].median()

print(f"\nПоле: Order Quantity"
      f"\nКоличество элементов: {count}"
      f"\nСреднее значение: {mean:.2f}"
      f"\nМинимум: {min_val}"
      f"\nМаксимум: {max_val}"
      f"\nСтандартное отклонение: {std:.2f}"
      f"\nМедиана: {median}")

# 6. Для поля Order Quantity построить гистограмму
plt.hist(df['Order_Quantity'], bins=30, color='skyblue', edgecolor='black')
plt.title('Гистограмма Order Quantity')
plt.xlabel('Количество заказов')
plt.ylabel('Частота')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 7. Посчитать количество продуктов каждого типа и построить круговую диаграмму
product_counts = df['Product_Category'].value_counts()

plt.figure(figsize=(10, 10))
plt.pie(
    product_counts,
    labels=product_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.Paired.colors
)
plt.title('Распределение продуктов по категориям')
plt.axis('equal')
plt.show()

# 8. Вывести все локации, где происходила продажа (страна и штаты)
locations = df[['Country', 'State']].drop_duplicates().reset_index(drop=True)
locations = locations.sort_values(['Country', 'State']).reset_index(drop=True)

print(f"\nВсе локации (страна и штат), где происходили продажи: \n{print(locations.to_string(index=False))}")

# 9. Вывести все продукты компании
unique_products = df['Product'].unique()

print("\nСписок уникальных продуктов:")
for product in unique_products:
    print(product)

# 10. Вывести столбчатую диаграмму 10 самых продаваемых продуктов
top_products = (df.groupby('Product')['Order_Quantity'].sum().nlargest(10))

plt.figure(figsize=(10, 6))
top_products.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('10 самых продаваемых продуктов')
plt.xlabel('Количество проданных единиц')
plt.ylabel('Продукт')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



