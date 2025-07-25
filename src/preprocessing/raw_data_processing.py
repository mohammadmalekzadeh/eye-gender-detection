from src.utlis import BASE_DIR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataProcessing():
    def __init__(self, raw_data = BASE_DIR+'/data/raw/raw_data.csv', n_components=13, test_size=0.3, random_state=42):
        self.df = pd.read_csv(raw_data)
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()


    def handling_missing_values(self, df):
        missing_threshold = 0.10
        missing_proportions = self.df.isnull().mean()
        cols_to_drop = missing_proportions[missing_proportions > missing_threshold].index
        
        self.df.drop(columns=cols_to_drop, inplace=True)
        
        for column in self.df.columns:
            if column != 'gender' and self.df[column].isnull().any():
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    self.df[column] = self.df.groupby('gender')[column].transform(lambda x: x.fillna(x.mean()))

        print('[!] Handling missing values was Done')
        return df


    def train_test_split(self, df):
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        test_df.drop(columns='gender', inplace=True)

        print('[!] Splitting of test and train data was done')
        return train_df, test_df


    def standardization(self, train_df, test_df):
        ss = StandardScaler()
        pca = PCA(self.n_components, whiten=False, svd_solver='auto', tol=0.0)
        gender = train_df['gender']
        train_df = train_df.drop(columns='gender')

        for col in train_df.columns:
            train_df[col] = ss.fit_transform(train_df[[col]])
            test_df[col] = ss.transform(test_df[[col]])

        train_df_v = pca.fit_transform(train_df)
        test_df_v = pca.transform(test_df)

        train_df_pca = pd.DataFrame(
        train_df_v,
        index=train_df.index,
        columns=[f'pixel_{i+1}' for i in range(train_df_v.shape[1])]
        )
        test_df_pca = pd.DataFrame(
        test_df_v,
        index=test_df.index,
        columns=[f'pixel_{i+1}' for i in range(test_df_v.shape[1])]
        )

        train_df = pd.concat([train_df_pca, gender], axis=1)

        print('[!] Standardization was Done')
        return train_df, test_df_pca


    def save_dataset(self, train_df, test_df):
        train_df.to_csv(BASE_DIR+'/data/processed/train_df.csv', index=False)
        test_df.to_csv(BASE_DIR+'/data/processed/test_df.csv', index=False)

        print('[!] train_df and test_df Saved')

    def transform(self):
        df = self.handling_missing_values(self.df)
        train_df, test_df = self.train_test_split(df)
        train_df, test_df = self.standardization(train_df, test_df)
        self.save_dataset(train_df, test_df)
