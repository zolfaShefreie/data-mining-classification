import pandas as pd

from preprocessing_data import Preprocessing
from model_process import ModelProcess


# BEST_MODEL_PATH = "resources/best_model.sav" #change this line as you wish
#
# model = ModelProcess(file_path=BEST_MODEL_PATH).load_model()
TEST_PATH = './test.xls'


def inference(path: str) -> list:
    """

    :param path: path of test dataset
    :return:
    result is the output of function which should be
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    """
    model_process = ModelProcess()
    model_process.load_model()
    test_df = pd.read_excel(path, index_col=0)
    test_df = Preprocessing().preprocess_test_data(test_df)
    return model_process.get_result_test(test_df)


if __name__ == '__main__':
    print(inference(TEST_PATH))


