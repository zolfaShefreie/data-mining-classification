import pandas as pd

if __name__ == '__main__':
    df = pd.read_excel('./dataset.xls')
    df = df.dropna()
    df.drop('Unnamed: 0', axis='columns', inplace=True)
    df = df.reset_index()
    df.to_excel('./test.xls')