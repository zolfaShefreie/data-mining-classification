import pandas as pd

from preprocessing_data import Preprocessing
from model_process import ModelProcess


TRAIN_PATH = './dataset.xls'
TEST_PATH = './test-no-stage.xls'

if __name__ == '__main__':
    model_process = ModelProcess()
    if model_process.needed_train():
        df = pd.read_excel(TRAIN_PATH, parse_dates=True)
        df.columns = ['Index', 'Customer', 'Agent', 'SalesAgentEmailID',
                      'ContactEmailID', 'Stage', 'Product', 'Close_Value', 'Created Date',
                      'Close Date']
        train_x_data, val_x_data, train_y_data, val_y_data = Preprocessing().preprocess_train_val_data(df)
        model_process.train(train_x_data, train_y_data)
        print(model_process.get_result_validation(val_x_data, val_y_data))
    else:
        model_process.load_model()
    test_df = pd.read_excel(TEST_PATH)
    test_df = Preprocessing().preprocess_test_data(test_df)
    test_y_df = pd.read_excel('./test-just-stage.xls')
    p = Preprocessing()
    p.load_categories()
    test_y_df = p.label_encode(test_y_df, 'Stage')
    model_process.get_result_test(test_df)
    model_process.print_result_validation(test_df, test_y_df)

