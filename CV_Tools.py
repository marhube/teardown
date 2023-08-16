#************** Start importing python modules
import pandas as pd
import random
import numpy as np
from itertools import chain, combinations
#from sklearn.metrics import auc, RocCurveDisplay
from plotnine import *
#******* Start importere python moduler for utvikling og evaluering av xgboost (Extreme Gradient Boosting)- prediksjonsmodell
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
#******* Slutt importere python moduler for utvikling og evaluering av xgboost (Extreme Gradient Boosting)- prediksjonsmodell
#********* End importing model modules
#************** End importing python modules
#
def set_params(model_variables,increasing=[],decreasing=[],params={}):
    # Setter først defaultverdier
    if len(params) == 0:
        params = {
            'objective': "binary:logistic",
            'eval_metric': 'logloss',
            'lambda': 10000,
            # 'n_estimators': 1000
        }
    # Bare legg til monotonicity constraints hvis det faktisk er noen variabler som skal ha slike constraints
    if(len(set(increasing).intersection(model_variables)) > 0 or
           len(set(decreasing).intersection(model_variables)) > 0):
        #
        constraints_dict = {}
        for feature in model_variables:
            constr_value = 0  # Tilsvarere ingen monotonitetskrav
            if feature in decreasing:
                constr_value = -1  # Monotonitetskrav synkende
            elif feature in increasing:  # Monotonitetskrav stigende
                constr_value = 1
            #
            constraints_dict[feature] = constr_value
        #
        constraints_list = [constraints_dict[feature] for feature in model_variables]
        params['monotone_constraints'] = str(tuple(constraints_list))
    #
    return(params)
# Mer info om "powerset" https://stackoverflow.com/questions/18035595/powersets-in-python-using-itertools
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
#
def xgb_cv_predict(X,y,params,cv=5):
    # Forste steg er å dele X og y inn i "k-folds".
    kfold = [random.randint(0, cv-1) for iter in range(X.shape[0])]
    predictions = [None]*X.shape[0]
    #
    for fold in range(cv):
        fit_rows = [ind for ind in range(X.shape[0]) if kfold[ind] != fold]
        pred_rows = [ind for ind in range(X.shape[0]) if kfold[ind] == fold]
        X_fit_fold = X.iloc[fit_rows,:]
        y_fit_fold = y.iloc[fit_rows]
        #
        X_pred_fold = X.iloc[pred_rows,:]
        #
        xgb_fit_data = xgb.DMatrix(X_fit_fold, y_fit_fold)
        xgb_score_data = xgb.DMatrix(X_pred_fold)
        #
        fold_model = xgb.train(params=params,
                                 dtrain=xgb_fit_data,
                                 num_boost_round=1000,
                                 early_stopping_rounds=10,
                                 evals=[(xgb_fit_data, 'train')],
                                 verbose_eval=False)
        #
        fold_predictions = fold_model.predict(data=xgb_score_data)
        # Lagrer prediksjonene
        for ind, row in enumerate(pred_rows):
            predictions[row] = fold_predictions[ind]
        #
    return predictions
#
def binary_LL(target,predict):
    # Hjelpefunksjon for å regne ut log likelihood ved binære utfall (to mulige utfall)
    ## Assumet that the input is two lists
    ll_frame = pd.DataFrame({'target': target, 'predict': predict})
    ll_frame['likelihood'] = ll_frame.apply(
        lambda x: x.predict if x.target == 1 else (1-x.predict if x.target == 0 else None),
        axis=1)
    #
    ll_frame['ll'] = ll_frame['likelihood'].map(lambda x: np.log(x) if x is not None else x)
    ll = sum(ll_frame['ll'])
    return ll
#
def scoregroup_count(target, predict, ngroups=5):
    sorted_target_predict = pd.DataFrame({"target":target,'predict':predict}).sort_values(
        by="predict",ascending = False)
    #
    sorted_target_predict['Scoregroup'] = np.ceil(
        [ngroups*i/sorted_target_predict.shape[0] for i in list(range(1,sorted_target_predict.shape[0]+1))])
    #
    '''
    Since the scoregroups are going to be used as groups for a groupby it is more convenient 
    if they are  integers
    '''
    sorted_target_predict['Scoregroup'] = sorted_target_predict['Scoregroup'].astype({"Scoregroup": int})
    scoregroup_stats = pd.DataFrame({'Scoregroup': list(range(1,ngroups+1))})
    gb = sorted_target_predict.groupby(['Scoregroup'])
    counts = gb.size().to_frame(name='counts')
    #
    target_sum_name = 'number_of_incidents'
    #
    scoregroup_stats = (counts.join(gb.agg({"target": 'sum'}).rename(columns={"target": target_sum_name}))
                       .join(gb.agg({"target": 'mean'}).rename(columns={"target": 'proportion_incidents'}))
                       .join(gb.agg({"predict": 'min'}).rename(columns={"predict": 'min_predict'}))
                       .join(gb.agg({"predict": 'max'}).rename(columns={"predict": 'max_predict'}))
                        .join(gb.agg({"predict": 'mean'}).rename(columns={"predict": 'mean_predict'}))
                       .reset_index())
    #
    scoregroup_stats = scoregroup_stats.assign(cum_share_of_test_group=scoregroup_stats['counts'].cumsum() /
                                               scoregroup_stats['counts'].sum(),
                                               cum_share_of_incidents=scoregroup_stats[target_sum_name].cumsum() /
                                               scoregroup_stats[target_sum_name].sum())
    #
    return(scoregroup_stats)
#
# Exhaustive search
def exhaustive_search(df,response_variable,candidate_variables,default_selection=[],ngroups=5):
    uncertain_variables = list(set(candidate_variables).difference(set(default_selection)))
    variable_choices = list(powerset(uncertain_variables))
    # Remove any entry with no variables (if present)
    variable_choices.remove(())
    list_of_choices = []
    auc_scores = []
    ll_scores = []
    AIC_scores = []
    BIC_scores = []
    #
    sgc_dict = {}
    #
    for iter_nr, choice in enumerate(variable_choices):
        current_selection = list(set(default_selection).union(set(list(choice))))
        X = df[current_selection]
        y = df[response_variable]
        list_of_choices.append(current_selection)
        params = set_params(current_selection)
        y_pred = xgb_cv_predict(X, y, params, cv=5)
        #
        ll_score = binary_LL(y.tolist(), y_pred)
        ll_scores.append(ll_score)
        AIC_score = 2*len(current_selection)-ll_score
        AIC_scores.append(AIC_score)
        BIC_score = len(current_selection)*np.log(len(y.tolist()))-2*ll_score
        BIC_scores.append(BIC_score)
        sgc_dict[str(current_selection)] =  scoregroup_count(df[response_variable], y_pred, ngroups=ngroups)
        #
        auc_score = roc_auc_score(y.tolist(), y_pred)
        auc_scores.append(auc_score)
    #
    new_eval_frame = pd.DataFrame({'var_choice': list_of_choices, 'auc': auc_scores,'ll': ll_scores, 'AIC': AIC_scores, 'BIC': BIC_scores})
    new_eval_frame['number_of_features'] = new_eval_frame['var_choice'].map(lambda x: len(x))
    #sorted_new_eval_frame = new_eval_frame.sort_values(by='auc', ascending=False)
    # OBSSSSS BIC skal minimeres, ikke maksimeres
    sorted_new_eval_frame = new_eval_frame.sort_values(by='BIC', ascending=True)
    return sorted_new_eval_frame,sgc_dict
