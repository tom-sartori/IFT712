import pandas as pd
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self, test_size: float = 0.25):
        df: pd.DataFrame = pd.read_csv('resources/data.csv')

        self.x_tab = df.drop(['id', 'species'], axis=1).values
        self.y_tab = df['species'].values

        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(self.x_tab, self.y_tab, test_size=test_size, random_state=2))
