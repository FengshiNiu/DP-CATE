# In the following, we have

# 1. **An abstract data generator** that takes in the covariates distribution, the propensity model, and the outcome model to generate simulated conditional RCT data
# 2. **An implementation of all econml CATE estimators**
# 3. **An implementation of EBM econml CATE estimators**
# 4. **Several specifications of DGP**
# 5. **A table of estimators and their mse and other error distribution summary statistics ordered by mse**


import numpy as np
import pandas as pd
import scipy
import time
from doubleml.datasets import fetch_bonus
from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from econml.orf import DROrthoForest
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML, NonParamDML
from econml.sklearn_extensions.model_selection import GridSearchCVList
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper, WeightedLasso, WeightedLassoCV
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, LogisticRegression, LogisticRegressionCV
import lightgbm as lgb
from econml.sklearn_extensions.linear_model import WeightedLassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

# Define DGP 
def data_generator(n, 
                   d, 
                   covariates_model, 
                   propensity_model,
                   control_outcome_model,
                   treatment_effect_model,
                   special_test_point=None,
                   special_test_value=None,
                   seed = 1234
                   ):
    """Generates iid population data for given covariates_model, propensity_model, control_outcome_model, and treatment_effect_model.
    
    Parameters
    ----------
        n (int): total sample size. An additional 0.5*n sample is used for test
        d (int): number of covariates
        covariates_model (func): d-dimension covariate DGP, take in an input d
        propensity_model (func): probability of treatment conditional on covariates
        control_outcome_model (func): untreated outcome conditional on covariates
        treatment_effect_model (func): treatment effect conditional on covariates
    """
    np.random.seed(seed)
    idx = np.arange(int(n * 1.5)).reshape([-1, 1])
    # Generate covariates
    X = np.apply_along_axis(lambda i: covariates_model(d), 1, idx)
    # Generate treatment
    P = np.apply_along_axis(lambda x: propensity_model(x), 1, X).reshape([-1, 1])
    T = np.apply_along_axis(lambda p: np.random.binomial(1, p, 1), 1, P).reshape([-1, 1])
    # Generate outcome
    Y0 =  np.apply_along_axis(lambda x: control_outcome_model(x), 1, X).reshape([-1, 1])
    treatment_effect = np.apply_along_axis(lambda x: treatment_effect_model(x), 1, X).reshape([-1, 1])
    Y1 = Y0 + treatment_effect
    Y = Y0 * (1-T) + Y1 * T
    P, T, Y0, Y1, Y = P.ravel(), T.ravel(), Y0.ravel(), Y1.ravel(), Y.ravel()

    X, X_test = X[:n], X[n:]
    P, P_test = P[:n], P[n:]
    T, T_test = T[:n], T[n:]
    Y0, Y0_test = Y0[:n], Y0[n:]
    Y1, Y1_test = Y1[:n], Y1[n:]
    Y, Y_test = Y[:n], Y[n:]
    treatment_effect, treatment_effect_test = treatment_effect[:n], treatment_effect[n:]

    data = {'X':X, 
            'P': P,
            'T': T, 
            'Y': Y,
            'Y0': Y0, 
            'Y1': Y1, 
            'treatment_effect': treatment_effect, 
            'X_test': X_test, 
            'treatment_effect_test': treatment_effect_test,
            'Y_test': Y_test,
            'special_test': [special_test_point, special_test_value]
            }

    return data

def regressor(seed=123):
    return GridSearchCVList([LassoCV(),
                            RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-2),
                            lgb.LGBMRegressor(random_state=seed)],
                            param_grid_list=[{},
                                            {'max_depth': [5,10,20],'min_samples_leaf': [5, 10]},
                                            {'learning_rate': [0.02,0.05,0.08], 'max_depth': [3, 5]}],
                            cv=3,
                            scoring='neg_mean_squared_error',
                            n_jobs=-2)
    # return GridSearchCVList([Lasso(),
    #                         RandomForestRegressor(n_estimators=400, random_state=seed),
    #                         GradientBoostingRegressor(random_state=seed),
    #                         ExplainableBoostingRegressor(random_state=seed, outer_bags=100)],
    #                         param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
    #                                         {'max_depth': [3, 6, None],
    #                                         'min_samples_leaf': [10, 30]},
    #                                         {'n_estimators': [50, 100, 150],
    #                                         'max_depth': [3, 6, None],
    #                                         'min_samples_leaf': [10, 30]},],
    #                         cv=5,
    #                         scoring='neg_mean_squared_error',
    #                         n_jobs=-2)

def regressor_final(seed=123):
    return GridSearchCVList([LassoCV(),
                            RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-2),
                            GradientBoostingRegressor(random_state=seed),
                            ExplainableBoostingRegressor(random_state=seed, outer_bags=20)],
                            param_grid_list=[{},
                                            {'max_depth': [5,10,20],'min_samples_leaf': [5, 10]},
                                            {'n_estimators': [50, 100, 150],
                                             'max_depth': [3, 6, None],
                                             'min_samples_leaf': [10, 30]},],
                            cv=3,
                            scoring='neg_mean_squared_error',
                            n_jobs=-2)
    # return GridSearchCVList([Lasso(),
    #                         RandomForestRegressor(n_estimators=400, random_state=seed),
    #                         GradientBoostingRegressor(random_state=seed),
    #                         ExplainableBoostingRegressor(random_state=seed, outer_bags=100)],
    #                         param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
    #                                         {'max_depth': [3, 6, None],
    #                                         'min_samples_leaf': [10, 30]},
    #                                         {'n_estimators': [50, 100, 150],
    #                                         'max_depth': [3, 6, None],
    #                                         'min_samples_leaf': [10, 30]},],
    #                         cv=5,
    #                         scoring='neg_mean_squared_error',
    #                         n_jobs=-2)

def clf(seed=123):
    return GridSearchCVList([LogisticRegressionCV(max_iter=1000),
                                  RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-2),
                                  lgb.LGBMClassifier()],
                                 param_grid_list=[{},
                                                  {'max_depth': [5,10,20],
                                                   'min_samples_leaf': [5, 10]},
                                                  {'learning_rate':[0.01,0.05,0.1],
                                                   'max_depth': [3,5]}],
                                 cv=3,
                                 scoring='neg_log_loss',
                                 n_jobs=-2)
    # return GridSearchCVList([LogisticRegression(),
    #                         RandomForestClassifier(n_estimators=400, random_state=seed),
    #                         GradientBoostingClassifier(random_state=seed),
    #                         ExplainableBoostingClassifier(random_state=seed, outer_bags=100)],
    #                         param_grid_list=[{'C': [0.01, .1, 1, 10, 100]},
    #                                         {'max_depth': [3, 6, None],
    #                                         'min_samples_leaf': [10, 30]},
    #                                         {'n_estimators': [50, 100, 150],
    #                                         'max_depth': [3, 6, None],
    #                                         'min_samples_leaf': [10, 30]},],
    #                         cv=5,
    #                         scoring='neg_log_loss',
    #                         n_jobs=-2)


def generate_CATE_learners(model_y, model_t, X, regressor=regressor, seed=123):
    #### Double Machine Learning estimators ####
    est_LinearDML = LinearDML(model_y=regressor(),
                            model_t=model_t,
                            discrete_treatment=True,
                            linear_first_stages=False,
                            random_state=seed)

    est_SparseLinearDML = SparseLinearDML(model_y=regressor(),
                                        model_t=model_t,
                                        featurizer=PolynomialFeatures(degree=3),
                                        discrete_treatment=True,
                                        linear_first_stages=False,
                                        random_state=seed)

    est_CausalForestDML = CausalForestDML(model_y=regressor(),
                                        model_t=model_t,
                                        criterion='mse', n_estimators=1000,
                                        min_impurity_decrease=0.001,
                                        discrete_treatment=True,
                                        random_state=seed)

    est_NonParamDML = NonParamDML(model_y=regressor(),
                                model_t=model_t,
                                model_final=regressor_final(),
                                discrete_treatment=True,
                                random_state=seed)


    #### Doubly robust estimators ####
    est_LinearDR = LinearDRLearner(model_regression=model_y,
                                model_propensity=model_t)

    est_LinearDRPolyFeature = LinearDRLearner(model_regression=model_y,
                                            model_propensity=model_t,
                                            featurizer=PolynomialFeatures(degree=2, 
                                                            interaction_only=True, include_bias=False))
                        
    est_SparseLinearDR = SparseLinearDRLearner(model_regression=model_y,
                                            model_propensity=model_t,
                                            featurizer=PolynomialFeatures(degree=3, interaction_only=True, include_bias=False))

    est_ForestDR = ForestDRLearner(model_regression=model_y,
                                model_propensity=model_t,
                                cv=5,
                                n_estimators=1000,
                                min_samples_leaf=10,
                                verbose=0, 
                                min_weight_fraction_leaf=.01)

    est_NPDR = DRLearner(model_regression=model_y,
                        model_propensity=model_t,
                        model_final=regressor_final())

    #### Meta Learners ####
    # T learner
    est_Tlearner = TLearner(models=regressor())

    # S learner
    est_Slearner = SLearner(overall_model=regressor())

    # X learner
    est_Xlearner = XLearner(models=regressor(), propensity_model=model_t)

    # Domain Adaptation learner
    est_DAlearner = DomainAdaptationLearner(models=model_y,
                                        final_models=regressor(),
                                        propensity_model=model_t)

    #### Forest Learners ####
    # DROrthoForest
    subsample_ratio = 0.4
    lambda_reg = np.sqrt(np.log(X.shape[1]) / (10 * subsample_ratio * X.shape[0]))

    est_DROrthoForest = DROrthoForest(
        n_trees=200, min_leaf_size=10,
        max_depth=30, subsample_ratio=subsample_ratio,
        propensity_model = model_t,
        model_Y = model_y,
        propensity_model_final=LogisticRegression(C=1/(X.shape[0]*lambda_reg), penalty='l1', solver='saga'), 
        model_Y_final=WeightedLasso(alpha=lambda_reg)
    )
    learner_dic = {'LinearDML': est_LinearDML, 
                    'SparseLinearDML': est_SparseLinearDML,
                    # 'DML': est_DML,
                    'CausalForestDML': est_CausalForestDML,
                    'NonParamDML': est_NonParamDML,
                    'LinearDR': est_LinearDR, 
                    'LinearDRPolyFeature': est_LinearDRPolyFeature, 
                    'SparseLinearDR': est_SparseLinearDR, 
                    'ForestDR': est_ForestDR, 
                    'NPDR': est_NPDR, 
                    'Tlearner': est_Tlearner, 
                    'Slearner': est_Slearner, 
                    'Xlearner': est_Xlearner, 
                    'DAlearner': est_DAlearner, 
                    'DROrthoForest': est_DROrthoForest,
                    'CausalForestDML': est_CausalForestDML
                    }
    return learner_dic

def generate_EBM_CATE_learners(model_y, model_t, regressor=regressor, outer_bags=10, seed=123):
    #### Double Machine Learning estimators ####
    est_EBM_NonParamDML = NonParamDML(model_y=regressor(),
                                model_t=model_t,
                                model_final=ExplainableBoostingRegressor(random_state=seed, outer_bags=outer_bags),
                                discrete_treatment=True,
                                random_state=seed)

    #### Doubly robust estimators ####
    est_EBM_DR = DRLearner(model_regression=regressor(),
                        model_propensity=model_t,
                        model_final=ExplainableBoostingRegressor(random_state=seed, outer_bags=outer_bags))

    #### Meta Learners ####
    # T learner
    est_EBM_Tlearner = TLearner(models=ExplainableBoostingRegressor(random_state=seed, outer_bags=outer_bags))

    # S learner
    est_EBM_Slearner = SLearner(overall_model=ExplainableBoostingRegressor(random_state=seed, outer_bags=outer_bags))

    # X learner
    est_EBM_Xlearner = XLearner(models=ExplainableBoostingRegressor(random_state=seed, outer_bags=outer_bags), propensity_model=model_t)

    # Domain Adaptation learner
    est_EBM_DAlearner = DomainAdaptationLearner(models=model_y,
                                        final_models=ExplainableBoostingRegressor(random_state=seed, outer_bags=outer_bags),
                                        propensity_model=model_t)
    
    ebm_learner_dic = {'EBM_NonParamDML': est_EBM_NonParamDML,
                       'EBM_DR': est_EBM_DR,
                       'EBM_Tlearner': est_EBM_Tlearner, 
                       'EBM_Slearner': est_EBM_Slearner, 
                       'EBM_Xlearner': est_EBM_Xlearner, 
                       'EBM_DAlearner': est_EBM_DAlearner}
    
    return ebm_learner_dic

def run_and_report_mse(model_list, learner_dic, Y, T, X, X_test, treatment_effect_test):
    # model_list should be a subset of learner_dic.keys()
    records = []
    for learner_name in model_list:
        est = learner_dic[learner_name]
        record = {}
        print(learner_name)
        startTime = time.time()
        est.fit(Y, T, X=X)
        record['fit_time'] =  (time.time() - startTime)
        error_test = abs(est.effect(X_test) - treatment_effect_test)
        mse_test = (error_test**2).mean()

        mse = (error_test**2).mean()
        error_50, error_75, error_95 = scipy.stats.mstats.mquantiles(error_test, prob=[0.5, 0.75, 0.95])
        record['model_name'] = learner_name
        record['mse'] = mse 
        record['error_50'] = error_50
        record['error_75'] = error_75
        record['error_95'] = error_95
        records.append(record)
    
    return records