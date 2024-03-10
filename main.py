from utils import get_data
from model import SVM_class
from typing import Tuple


def get_hyperparameters() -> Tuple[float, int, float]:
    #implement this
    # get the hyperparameters
    learning_rate=float(0.001)
    #num_iters=100000
    #C=0.1
    #learning_rate=float(0.01)
    num_iters=50000
    C=1
    #learning_rate=float(0.1)
    #num_iters=10000
    #C=10
    return learning_rate,num_iters,C
    #raise NotImplementedError


def main() -> None:
    # hyperparameters
    learning_rate, num_iters,C=get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()   
       
    # create a model
    svm = SVM_class()

    # fit the model
    svm.fit(
            X_train, y_train, C=C,
            learning_rate=learning_rate,
            num_iters=num_iters,
        )

    # evaluate the model
    accuracy = svm.accuracy_score(X_test, y_test)
    precision = svm.precision_score(X_test,y_test)
    recall = svm.recall_score(X_test, y_test)
    f1_score = svm.f1_score(X_test, y_test)     

    print(f'accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')


if __name__ == '__main__':
    main()
