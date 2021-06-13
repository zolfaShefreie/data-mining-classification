from model_process import ModelProcess


from pandas import DataFrame
BEST_MODEL_PATH = "resources/best_model.sav" #change this line as you wish

model = ModelProcess(file_path=BEST_MODEL_PATH).load_model()


def inference(path: str) -> list:
    """

    :param path: path of test dataset
    :return:
    result is the output of function which should be
    somethe like: [0,1,1,1,0]
    0 -> Lost
    1 -> Won
    """

    result = []
    # your code starts here
    # model.predict()
    
    ## your code ends here
    return result
    

