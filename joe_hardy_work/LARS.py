from tqdm import tqdm
from joe_hardy_work.processed_data_load import X_train_total,X_test_total,y_train_total,y_test_total

from sklearn.linear_model import BayesianRidge, Lars

funcs = [BayesianRidge, Lars]

highest_score = 0.0
highest_scoring_model = None
for _ in tqdm(range(10000)):
    for func in funcs:
        model = func()
        model.fit(X_train_total, y_train_total.to_numpy().ravel())
        score = model.score(X_test_total, y_test_total.to_numpy().ravel())
        # print("{}-{}: {}".format("Total",repr(func),str(score)))
        # print(model.__dict__)
        if score > highest_score:
            highest_score = score
            highest_scoring_model = model

print(score, model, model.__dict__)

"""
Bridgeport_1 Train_X
Bridgeport_1 Train_Y
Bridgeport_2 Train_X
Bridgeport_2 Train_Y
Bridgeport_3 Train_X
Bridgeport_3 Train_Y
Drill Train_X
Drill Train_Y
Lathe Train_X
Lathe Train_Y
(13958, 6)
100%|██████████| 10000/10000 [00:48<00:00, 204.34it/s]
0.8095017880734434 Lars(copy_X=True, eps=2.220446049250313e-16, fit_intercept=True, fit_path=True,
     n_nonzero_coefs=500, normalize=True, precompute='auto', verbose=False) {'fit_intercept': True, 'verbose': False, 'normalize': True, 'precompute': 'auto', 'n_nonzero_coefs': 500, 'eps': 2.220446049250313e-16, 'copy_X': True, 'fit_path': True, 'alphas_': array([1.62273252e-04, 1.46908251e-04, 3.26646555e-05, 1.22720962e-05,
       3.78981810e-06, 1.54996363e-06, 0.00000000e+00]), 'n_iter_': 6, 'coef_': array([-0.15277962,  0.53135132, -0.08261724,  0.68411249,  0.61238313,
       -0.74144254]), 'active_': [1, 3, 4, 5, 0, 2], 'coef_path_': array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        -0.19668968, -0.34484148],
       [ 0.        ,  0.21446468,  1.0815169 ,  1.3137668 ,  1.43707338,
         1.43928139,  1.46199944],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        , -0.17794468],
       [ 0.        ,  0.        ,  0.86705222,  0.92733012,  0.93008235,
         1.12678102,  1.41090116],
       [ 0.        ,  0.        ,  0.        ,  0.27669958,  0.54168121,
         0.5586894 ,  0.55883217],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.26844989,
        -0.32001634, -0.35603326]]), 'intercept_': 0.001335458821423205}

Process finished with exit code 0
"""
