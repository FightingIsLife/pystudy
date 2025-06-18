import pandas as pd

ass = pd.Series([1, 7, 5, 3.0], name="A")

print(ass)
print(ass.values)
print(ass.head(1))
print(ass.tail(3).head(2))
print(ass.dtype)
print(ass.shape)
print(ass.cumprod())
print(ass.cumsum())
print(ass.describe())
print(ass.describe()['count'])
print(ass.describe()['mean'])
print(ass.describe()['max'])
print(ass.describe()['std'])
print(ass.describe()['min'])
print(ass.mean())
print(ass.rank())

