import pandas as pd

iris_df = pd.read_csv("../iris.csv")


label_map = {

        "Setosa": 0,
        "Versicolor": 1,
        "Virginica": 2,
}


iris_df['variety'] = iris_df['variety'].apply(lambda x: label_map[x])

x, y =  iris_df[['sepal.width', 'petal.width']].values,  iris_df['variety'].values
print(y)
print(x)

