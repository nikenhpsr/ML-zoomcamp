```python
import pandas as pd
```

### Questions 1 : Pandas Version

```python
# pandas version
pd.__version__
```

### Getting the Data

```python
data = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv')
data.head()
```

### Questions 2

```python
# How many columns in dataset?
data.columns
```

### Questions 3

```python
# Which columns in the dataset have missing values?
missing_values = data.isnull().sum()

column_with_missing_values = missing_values[missing_values > 0]
print(column_with_missing_values)
```

### Questions 4

```python
# Number of unique values in the 'ocean_proximity' column
unique_values = data['ocean_proximity'].nunique()
print("Number of unique non-numeric values in 'ocean_proximity' column:", unique_values)
```

### Questions 5

```python
# Average value of the 'median_house_value' for the houses near the bay
near_bay = data[data['ocean_proximity'] == 'NEAR BAY']

mean_median_house_value = near_bay['median_house_value'].mean()

print(mean_median_house_value)
```

### Questions 6

1.  Calculate the average of total_bedrooms column in the dataset.
2.  Use the fillna method to fill the missing values in total_bedrooms
    with the mean value from the previous step.
3.  Now, calculate the average of total_bedrooms again.
4.  Has it changed?

> Hint: take into account only 3 digits after the decimal point.

```python
# calculate initialize total_bedrooms mean

initial_mean = data['total_bedrooms'].mean()
print('initial mean is', initial_mean)

# fill missing values with initial mean value
fill_missing_value = data['total_bedrooms'].fillna(initial_mean)

# final mean of total_bedrooms
final_mean = fill_missing_value.mean()
print('mean after filling missing value wth mean is', final_mean)
```

### Questions 7

1.  Select all the options located on islands.
2.  Select only columns housing_median_age, total_rooms, total_bedrooms.
3.  Get the underlying NumPy array. Let\'s call it X.
4.  Compute matrix-matrix multiplication between the transpose of X
    and X. To get the transpose, use X.T. Let\'s call the result XTX.
5.  Compute the inverse of XTX.
6.  Create an array y with values \[950, 1300, 800, 1000, 1300\].
7.  Multiply the inverse of XTX with the transpose of X, and then
    multiply the result by y. Call the result w.
8.  What\'s the value of the last element of w?

> Note: You just implemented linear regression. We\'ll talk about it in the next lesson.

```python
# step 1: Select all the options located on islands.

island_data = data[data['ocean_proximity'] == 'ISLAND']

# step 2: Select only columns housing_median_age, total_rooms, total_bedrooms.

df = island_data[['housing_median_age', 'total_rooms', 'total_bedrooms']]

# step 3: Get the underlying NumPy array. Let's call it X.
import numpy as np

X = df.to_numpy()

# step 4: Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
XTX = np.dot(X.T, X)

# Step 5: Compute the inverse of XTX.
XTX_inverse = np.linalg.inv(XTX)

# Step 6: Create an array y with values [950, 1300, 800, 1000, 1300].
y = np.array([950, 1300, 800, 1000, 1300])

# Step 7: Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = np.dot(np.dot(XTX_inverse, X.T), y)

# Step 8: Get the value of the last element of w.
last_element_of_w = w[-1]

# Print the result
print("The value of the last element of w is:", last_element_of_w)
```
