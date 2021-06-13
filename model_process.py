from sklearn.ensemble import RandomForestClassifier
import joblib


class ModelProcess:

    def __init__(self, file_path='./model.sav'):
        self.model = RandomForestClassifier(random_state=30, n_estimators=100, criterion='entropy',
                                            max_depth=17, bootstrap=False)
        self.file_path = file_path

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def save_model(self):
        joblib.dump(self.model, open(self.file_path, 'wb'))

    def load_model(self):
        joblib.load(open(self.file_path, 'rb'))

    def get_result_test(self, x_test):
        return self.model.predict(x_test)
