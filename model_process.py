from sklearn.ensemble import RandomForestClassifier
import joblib


class ModelProcess:

    def __init__(self, file_path='./model.sav'):
        self.model = RandomForestClassifier(random_state=30, n_estimators=100, criterion='entropy',
                                            max_depth=17, bootstrap=False)
        self.file_path = file_path

    def needed_train(self):
        try:
            open(self.file_path, 'r')
        except FileNotFoundError:
            return True
        return False

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.save_model()

    def save_model(self):
        joblib.dump(self.model, open(self.file_path, 'wb'))

    def load_model(self):
        self.model = joblib.load(open(self.file_path, 'rb'))

    def get_result_test(self, x_test):
        return self.model.predict(x_test)

    def print_result_validation(self):
        pass
