import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


if os.path.exists('X_train_total.pkl'):
    X_train_total = pd.read_pickle('X_train_total.pkl')
    X_test_total = pd.read_pickle('X_test_total.pkl')
    #X_validate_total = pd.read_pickle('X_validate_total.pkl')
    y_test_total = pd.read_pickle('y_test_total.pkl')
    y_train_total = pd.read_pickle('y_train_total.pkl')
    #y_validate_total = pd.read_pickle('y_validate_total.pkl')
else:
    head = "../processed_data"
    data_folders = os.listdir(head)
    datasets = {}
    X_train_total, X_test_total, y_train_total, y_test_total = [None for _ in range(4)]
    for data_folder in data_folders:
        datasets[data_folder] = {}
        for split in ["Train_X", "Train_Y"]:
            print(data_folder, split)
            datasets[data_folder][split] = pd.read_excel(os.path.join(head, data_folder, split + '.xlsx')).dropna()
            X, y = datasets[data_folder][split].iloc[:, 1:-1], datasets[data_folder][split].iloc[:,
                                                               -1].to_numpy().reshape(-1, 1)
            #X['t']=X.index
            X_train, X_test, y_train, y_test = map(lambda d: normalize(d, axis=0), train_test_split(X, y))
            # print(type(X_train))
            if not isinstance(X_train_total, pd.DataFrame):
                X_train_total = pd.DataFrame(X_train)
                X_test_total = pd.DataFrame(X_test)
                y_train_total = pd.DataFrame(y_train)
                y_test_total = pd.DataFrame(y_test)
            else:
                X_train_total = pd.concat([X_train_total, pd.DataFrame(X_train)])
                X_test_total = pd.concat([X_test_total, pd.DataFrame(X_test)])
                y_train_total = pd.concat([y_train_total, pd.DataFrame(y_train)])
                y_test_total = pd.concat([y_test_total, pd.DataFrame(y_test)])
    print(X_train_total)
    #X_train_total,X_validate_total,y_train_total,y_validate_total = train_test_split(X_train_total, y_train_total)
    X_train_total.to_pickle('X_train_total.pkl')
    #X_validate_total.to_pickle('X_validate_total.pkl')
    X_test_total.to_pickle('X_test_total.pkl')
    y_train_total.to_pickle('y_train_total.pkl')
    y_test_total.to_pickle('y_test_total.pkl')
    #y_validate_total.to_pickle('y_validate_total.pkl')

print(X_train_total.shape)
