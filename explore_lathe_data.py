import pandas as pd
import os
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import normalize
from sklearn.linear_model import Lars,Ridge,LassoLars,ElasticNet,BayesianRidge,SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
head = './data/Lathe'

model_funcs = [Lars,Ridge,LassoLars,ElasticNet,BayesianRidge,SGDRegressor,MLPRegressor,IsotonicRegression]

components = [
    "Xaxial",
    "Yradial"
]

datasets = {
    component: pd.read_excel(os.path.join(head, "{}-{}-train.xlsx".format(component, "lathe"))).dropna()
    for component in components
}
splitsets = {}
normsets = {}
norms = {}
for dataset in datasets:
    X_train, X_test, Y_train, Y_test = tts(datasets[dataset].iloc[:, 1:-1], datasets[dataset].iloc[:, -1])
    #print(Y_train)
    splitsets[dataset] = {
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train.to_numpy().reshape(-1,1),
        'Y_test': Y_test.to_numpy().reshape(-1,1)
    }
    #print(splitsets[dataset]['Y_train'])
    normsets[dataset] = {k:None for k in splitsets[dataset]}
    #print(normsets)
    for set_ in splitsets[dataset]:
        normsets[dataset][set_] = normalize(splitsets[dataset][set_],axis=0)
    #print(normsets[dataset]['Y_train'])
models = {}
for model_func in model_funcs:
    models[model_func] = {}
    for component in normsets:
        X_train,Y_train = normsets[component]['X_train'],normsets[component]['Y_train'].ravel()
        X_test, Y_test = normsets[component]['X_test'], normsets[component]['Y_test'].ravel()

        models[model_func][component] = model_func()
        models[model_func][component].fit(X_train,Y_train)
        print(component+"{}:".format(repr(model_func))+str(models[model_func][component].score(X_test,Y_test)))
        #print(models[model_func][component].__dict__)


# Xaxial<class 'sklearn.linear_model._least_angle.Lars'>:0.8991904708387619
# Yradial<class 'sklearn.linear_model._least_angle.Lars'>:0.9203349381959296
# Xaxial<class 'sklearn.linear_model._ridge.Ridge'>:0.6988821351575203
# Yradial<class 'sklearn.linear_model._ridge.Ridge'>:0.7071722946078256
# Xaxial<class 'sklearn.linear_model._least_angle.LassoLars'>:-0.031205300046360348
# Yradial<class 'sklearn.linear_model._least_angle.LassoLars'>:-0.03685225068366371
# Xaxial<class 'sklearn.linear_model._coordinate_descent.ElasticNet'>:-0.031205300046360348
# Yradial<class 'sklearn.linear_model._coordinate_descent.ElasticNet'>:-0.03685225068366371
# Xaxial<class 'sklearn.linear_model._bayes.BayesianRidge'>:0.906965834371071
# Yradial<class 'sklearn.linear_model._bayes.BayesianRidge'>:0.9277037713610815
# Xaxial<class 'sklearn.linear_model._stochastic_gradient.SGDRegressor'>:0.0032853002823525213
# Yradial<class 'sklearn.linear_model._stochastic_gradient.SGDRegressor'>:-0.008508740600268405
# Xaxial<class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>:0.7402199954024179
# Yradial<class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>:0.7769819767465164
#
# Process finished with exit code 1


        

