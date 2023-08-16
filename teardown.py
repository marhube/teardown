#************** Start importing python modules
import sys # For sys.exit()
import os # For
import time # For time.time()
import pandas as pd
import numpy as np
import PySimpleGUI as sg
#***** Start ANOVA-relaterte moduler
import statsmodels.api as sm
from statsmodels.formula.api import ols
#***** SluttANOVA-relaterte moduler
#******* Start importere python moduler for utvikling og evaluering av xgboost (Extreme Gradient Boosting)- prediksjonsmodell
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score
#******* Slutt importere python moduler for utvikling og evaluering av xgboost (Extreme Gradient Boosting)- prediksjonsmodell
#****** Start moduler for å lage figurer
import matplotlib.pyplot as plt
from plotnine import *
from mizani.formatters import percent_format
#****** Slutt moduler for å lage figurer

#************** End importing python modules
#
#****** Start Import some auxiliary code
import ReadDataTools
import CV_Tools
#****** End Import some auxiliary code
#
sg.theme("DarkTeal2")
layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],[sg.Button("Submit")]]
###Building Window
window = sg.Window('Last opp zip-fil', layout, size=(600,150))
#
event, values = window.read()
#
# Memo to self: The first json file (sensordata) contains metadata
if event == "Submit":    
    df,meta_dict = ReadDataTools.convertJSON2Frame(values["-IN-"])
    print(f'meta_dict er {meta_dict}')
    print(df.info())   
    # Se på konsistens og riktighet til dataene******** 
    print(f"Tidligste obsersvasjonstidspunkt er {df['timestamp'].min()} og seneste er  {df['timestamp'].max()} ")
    example_variables = ['timestamp','train_id','location_type','distance_to_pole']
    print(df.query("train_id == 0 and timestamp == '2023-06-11T19:21:23.043666'")[example_variables])
    # Kommentar:  Enten må det være snakk om ulike tidspunkter eller ulike tog
    # Litt mer om konsistens
    print(df[['train_id','wind_direction']].drop_duplicates().sort_values(by='train_id'))
    # Kommentar: Vindretningen er unikt bestemt av 'train_id'
    # Sjekker om det er sammenheng mellom windhastighet og "train_id" ("tog-id") eller "location_type" (type sted bru,tunnel,åpent landskap o.s.v.)
    #  Two-way ANOVA wind_speed ~  train_id + location_type
    wind_model = ols('wind_speed ~ C(train_id) + C(location_type)', data=df).fit() 
    print(sm.stats.anova_lm(wind_model, typ=2))
    #Kommentar: 99,8% av variansen ($R^2$) til 'wind_speed' er forklart av 'train_id'
#   #********* Start   Utforskning for valg av modell
    firstTrain = df.query('train_id==0').sort_values(by='timestamp')

    #*********** Start exhaustive searching
    # "default_selection er variabler som jeg på forhånd har bestemt at skal være med i modellen
    #
    before_exhaustive = time.time()
    response_variable = 'incident'
    sorted_new_eval_frame,sgc_dict = CV_Tools.exhaustive_search(
        df=firstTrain,
        response_variable = response_variable,
        candidate_variables = ['kontaktkraft','distance_to_pole','hastighet'],
        default_selection=['kontaktkraft'],
        ngroups=5
        )
    after_exhaustive = time.time()
    #
    print(f'Exhaustive search tok {str(after_exhaustive-before_exhaustive)} sekunder')
    print(sorted_new_eval_frame)
    #*********** End exhaustive searching
    
    #***** Start testing av valgt modell
    response_variable = 'incident'
    model_features = ['kontaktkraft','hastighet','distance_to_pole']
    # Splitt opp i treningsdata og testdata
    train_data = df.query('train_id >= 1 and train_id  <=5')
    test_data = df.query('train_id > 5')
    # Gjør så selve modelltilpasningen
    params = CV_Tools.set_params(model_features)
    xgb_train_data = xgb.DMatrix(train_data[model_features], train_data[response_variable])
    xgb_test_data = xgb.DMatrix(test_data[model_features])
    #
    validation_model = xgb.train(
        params=params,
        dtrain=xgb_train_data,
                       num_boost_round=1000,
                       early_stopping_rounds=10,
                       evals=[(xgb_train_data, 'train')],
                       verbose_eval=20)
    # Start evaluere prediksjonsmodell
    validation_importances = validation_model.get_score(importance_type='gain')
    print(validation_importances)
    # Lager prediksjoner
    test_predictions = validation_model.predict(data=xgb.DMatrix(test_data[model_features]))
    sgc_validation = CV_Tools.scoregroup_count(test_data[response_variable], test_predictions, ngroups=10)
    class_predictions = [round(x) for x in test_predictions]
    accuracy = accuracy_score([str(x) for x in test_data[response_variable].to_list()], [str(x) for x in class_predictions])
    false_negatives = [test_predictions[ind] for ind in range(len(test_predictions)) if test_data[response_variable].to_list()[ind] == 1 and test_predictions[ind] <0.5] 
    false_positives = [test_predictions[ind] for ind in range(len(test_predictions)) if test_data[response_variable].to_list()[ind] == 0 and test_predictions[ind] >=0.5]
    #
    eval_auc_score = roc_auc_score(test_data[response_variable].tolist(), test_predictions)
    print(f'accuracy er {accuracy}, antall falske negative er {len(false_negatives)} og antall falske positive er {len(false_positives)} ')
    # Slut evaluere prediksjonsmodell
    #**** Slutt testing av valgt modell
    ## Start kode for å illustrere hvordan hendelsene fordeler seg på ulike 'train_id'
    df_group_train = df.groupby(['train_id'])
    gb = df[['incident','train_id']].sort_values(by="train_id").groupby(['train_id'])
    counts = df_group_train.size().to_frame(name='counts')  
    response_variable = 'incident'
    target_sum_name = 'number_of_incidents'
    #
    train_id_stats = (counts.join(gb.agg({response_variable: 'sum'}).rename(columns={response_variable: target_sum_name}))
                       .join(gb.agg({response_variable: 'mean'}).rename(columns={response_variable: 'proportion_incidents'}))
                       .reset_index())   
    train_id_stats['proportion_of_all_events'] =   train_id_stats['counts'].map(lambda x: x/df.shape[0])  
    train_id_stats['proportion_of_all_incidents'] =   train_id_stats[target_sum_name].map(lambda x: x/df[response_variable].sum()) 
    train_id_stats['cumulative_share_events'] =   train_id_stats['counts'].cumsum()/train_id_stats['counts'].sum()
    train_id_stats.sort_values('proportion_incidents',inplace = True,ascending = False)
    train_id_stats['cumulative_share_incidents'] =   train_id_stats['proportion_of_all_incidents'].cumsum()
    #
    train_id_stats['train_id_factor'] =  pd.Categorical(train_id_stats['train_id'],categories = train_id_stats['train_id'].to_list(),ordered=True)
    #
    var_name = "Counting_variable"
    value_name = "Proportion"
    #"Rydder" litt i dataene (omstrukturer til "long" format) som forberedelse til å bruke ggplot til å lage barchart
    plot_data = train_id_stats.melt(
        id_vars=['train_id_factor'],
        value_vars=["proportion_of_all_incidents","cumulative_share_incidents"],
        var_name = var_name,
        value_name = value_name,
        ignore_index = True
        )
    #
    train_id_plot =  (ggplot(plot_data, aes(x="train_id_factor",y = value_name, fill = var_name))
                    + geom_bar(stat="identity",position = "dodge")
                    + scale_y_continuous(labels=percent_format())
                    + geom_text(
                        aes(label = value_name),
                          va = 'top',
                        format_string="{:.1%}",
                        size=20
                    )
                  + theme(
                      figure_size=(16, 8),
                      legend_title = element_text(size=30),
                      legend_text = element_text(size=20)
                  )
                )
    #
    #train_id_plot.save("train_id_plot.png", dpi=600)
    train_id_plot.draw(show=True)    
#
               

