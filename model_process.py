from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
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

    def get_result_validation(self, x_test, y_test):
        return self.model.score(x_test, y_test)

    def print_result_validation(self, x_test, y_test):
        predict = self.get_result_test(x_test)
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict))
        print('f1 score : ', f1_score(y_test, predict))
        print('score: ', self.model.score(x_test, y_test))
