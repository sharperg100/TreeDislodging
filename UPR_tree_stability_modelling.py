"""
Script used to fit logistic regression to observed tree stability and 
to estimate tree stability for a range of design floods.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
import numpy as np
import os


def main():

    only_fit_the_model = False  # False means also run the model

    split_training_data = {'Do': False, 'Split': 0.3}

    ### Disused:
    parameter_combos = {'moment_only': ['volume_term'],
                        'moment_only_cm': ['volume_term_cm'],
                        'tri_factor': ['volume_term', 'delta_a', 'd'],
                        'dual_factor': ['delta_term', 'd'],
                        'bulk_factor': ['bulk_term'],
                        'diameter_sqr': ['diameter_sqr'],
                        'diameter_cubed': ['diameter_cubed']}

    # model_parameters = parameter_combos['diameter_sqr']
    ###

    model_parameters = ['Xb3']

    depth_cutoff = {'do': False, 'value': 3.0}

    filename_template = 'tree_db/UPR_Trees_event.csv'
    fit_events = ['Jan2011_02', 'Q00002Y']

    ignore_trees = {'do': True, 'file': 'tree_db/Ignore_trees_02.csv', 'list': []}
    if ignore_trees['do']:
        ignore_df = pd.read_csv(ignore_trees['file'], header=0, index_col=0)
        ignore_trees['list'] = ignore_df.index.to_list()

    stability_model = fit_model(filename_template, fit_events, depth_cutoff, model_parameters,
                                split_training_data, ignore_trees)

    if not only_fit_the_model:
        run_model(filename_template, depth_cutoff, model_parameters, stability_model, ignore_trees)


def run_model(filename_template, depth_cutoff, model_parameters, stability_model, ignore_trees):
    print()
    print('Model fitted... now moving on to run the model on a range of events!')

    events = ['Q00002Y', 'Q00005Y', 'Q00010Y', 'Q00020Y', 'Q00050Y', 'Q00100Y', 'Q00200Y', 'Q00500Y',
              'Q01000Y', 'Q02000Y', 'Q05000Y', 'Q10000Y', 'Jan2011_02']

    # set up the results dataframe
    tree_db_filename = filename_template.replace('event', events[0])

    all_results = pd.read_csv(tree_db_filename, header=0, index_col=0, usecols=['BasinID', 'X', 'Y'])
    all_strain = pd.read_csv(tree_db_filename, header=0, index_col=0, usecols=['BasinID', 'X', 'Y'])

    if ignore_trees['do']:
        print('Dropping trees excluded from the analysis')
        print(ignore_trees['list'])
        all_results.drop(ignore_trees['list'], axis=0, inplace=True)
        all_strain.drop(ignore_trees['list'], axis=0, inplace=True)

    # Get and prepare the data
    for event in events:
        print('Running tree stability model on event: {}'.format(event))
        tree_db_filename = filename_template.replace('event', event)
        data = pd.read_csv(tree_db_filename, header=0, index_col=0)
        data.dropna(subset=['competence', 'M_D'], inplace=True)  # Drop dry trees
        if ignore_trees['do']:
            data.drop(ignore_trees['list'], axis=0, inplace=True)
        # apply the depth cut-off
        if depth_cutoff.get('do'):
            if 'root_depth' in data.columns:
                depth_array = data['root_depth'].values
                data['root_depth'] = np.where(depth_array > depth_cutoff.get('value'),
                                           depth_cutoff.get('value'),
                                           depth_array)

        data['V1'] = np.power(data['Dbh'], 3.0)
        data['V2'] = np.square(data['Dbh']) * data['H']
        data['V3'] = np.pi * np.square(data['area']) / (4 * data['H'])

        data['Xa1'] = data['M_D'] / data['V1']
        data['Xa2'] = data['M_D'] / data['V2']
        data['Xa3'] = data['M_D'] / data['V3']

        data['Xb1'] = data['M_D'] * data['competence'] / (data['root_depth'] * data['V1'])
        data['Xb2'] = data['M_D'] * data['competence'] / (data['root_depth'] * data['V2'])
        data['Xb3'] = data['M_D'] * data['competence'] / (data['root_depth'] * data['V3'])

        # Disused parameters:
        # data['volume_term'] = data['M_D'] / data['volume']
        # data['delta_term'] = data['M_D'] / (data['delta_a'] * data['volume'])
        # data['d_term'] = data['M_D'] * data['d'] / data['volume']
        # data['bulk_term'] = data['M_D'] * data['d'] / (data['delta_a'] * data['volume'])

        # assign the variables for fitting and do prediction
        X = data[model_parameters]
        data['Predicted'] = stability_model.predict(X)
        data.to_csv('stability_model_results/predicted_output_{}.csv'.format(event))
        print('Done writing model results')
        all_results[event] = data['Predicted']
        all_strain[event] = data['strain']

    all_results.to_csv('stability_model_results/predicted_output_summary.csv')
    all_strain.to_csv('stability_model_results/strain_output_summary.csv')
    print('Stability model simulations complete!')
    

def fit_model(filename_template, fit_events, depth_cutoff, model_parameters, split_training_data, ignore_trees):
    # open the dataframe with columns: Depth, bss, u, sp, M
    tree_db_filename = filename_template.replace('event', fit_events[0])
    data = pd.read_csv(tree_db_filename, header=0, index_col=0)
    data['event'] = fit_events[0]
    ignore_list = []
    if ignore_trees['do']:
        print('Dropping trees excluded fro the analysis')
        print(ignore_trees['list'])
        data.drop(ignore_trees['list'], axis=0, inplace=True)
    # Append the other results
    if len(fit_events) > 1:
        for index, item in enumerate(fit_events):
            if index > 0:
                tree_db_filename = filename_template.replace('event', fit_events[index])
                add_trees = pd.read_csv(tree_db_filename, header=0, index_col=0)
                add_trees['event'] = fit_events[index]
                if ignore_trees['do']:
                    add_trees.drop(ignore_trees['list'], axis=0, inplace=True)
                data = data.append(add_trees)

    # Drop dry trees
    data.dropna(subset=['competence', 'M_D'], inplace=True)
    data = data[data['M_D'] > 0.0001]
    print(data)

    # apply the depth cut-off
    if depth_cutoff.get('do'):
        if 'root_depth' in data.columns:
            depth_array = data['root_depth'].values
            data['root_depth'] = np.where(depth_array > depth_cutoff.get('value'),
                                       depth_cutoff.get('value'),
                                       depth_array)

    data['V_jap'] = np.square(data['Dbh']*100)
    data['V1'] = np.power(data['Dbh'], 3)
    data['V2'] = np.square(data['Dbh']) * data['H']
    data['V3'] = np.pi * np.square(data['area']) / (4 * data['H'])

    data['X_jap'] = data['M_D']*1000 / data['V_jap']
    data['Xa1'] = data['M_D'] / data['V1']
    data['Xa2'] = data['M_D'] / data['V2']
    data['Xa3'] = data['M_D'] / data['V3']

    data['Xb1'] = data['M_D'] * data['competence'] / (data['root_depth'] * data['V1'])
    data['Xb2'] = data['M_D'] * data['competence'] / (data['root_depth'] * data['V2'])
    data['Xb3'] = data['M_D'] * data['competence'] / (data['root_depth'] * data['V3'])

    # Disused terms:
    # data['volume_term'] = data['M_D'] / data['volume']
    # data['delta_term'] = data['M_D'] / (data['delta_a'] * data['volume'])
    # data['d_term'] = data['M_D'] * data['d'] / data['volume']
    # data['bulk_term'] = data['M_D'] * data['d'] / (data['delta_a'] * data['volume'])
    # data['diameter'] = np.sqrt(data['volume'] / data['H'])*100  # units: cm
    # data['diameter_cubed'] = data['M_D'] / np.power(data['diameter'], 3)
    # data['volume_term_cm'] = data['M_D'] / (np.square(data['diameter']) * data['H'])
    # data['diameter_sqr'] = data['M_D'] / np.square(data['diameter'])

    # assign the variables for fitting
    all_data = {}
    all_parameters = ['Xa1', 'Xa2', 'Xa3', 'Xb1', 'Xb2', 'Xb3', 'X_jap']
    outlogreg = ''
    for use_parameter in all_parameters:
        X = data[[use_parameter]]
        # X = data[model_parameters]
        y = data['stable']

        # upscale the disturbed trees to create a more balanced data set
        ovs = SMOTE(random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_training_data['Split'], random_state=0)
        columns = X_train.columns
        if not split_training_data['Do']:
            X_train, X_test, y_train, y_test = X, X, y, y
        os_data_X, os_data_y = ovs.fit_resample(X_train, y_train)
        os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
        os_data_y = pd.DataFrame(data=os_data_y, columns=['stable'])

        # we can Check the numbers of our data
        print("length of oversampled data is ", len(os_data_X))
        print("Number of undisturbed trees in oversampled data", len(os_data_y[os_data_y['stable'] == 0]))
        print("Number of disturbed trees", len(os_data_y[os_data_y['stable'] == 1]))
        print("Proportion of undisturbed data in oversampled data is ",
              len(os_data_y[os_data_y['stable'] == 0])/len(os_data_X))
        print("Proportion of subscription data in oversampled data is ",
              len(os_data_y[os_data_y['stable'] == 1])/len(os_data_X))

        # implement the model
        X = os_data_X
        y = os_data_y
        logit_model = sm.Logit(y, X)
        result = logit_model.fit()
        print(result.summary2())

        # fit the logit model
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        logreg = LogisticRegression()
        # logreg.fit(X_train, y_train)
        logreg.fit(X, y)
        y_pred = logreg.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

        # accuracy on full dataset
        X = data[[use_parameter]]
        y = data['stable']
        data['Predicted'] = logreg.predict(X)
        summary_data = {}

        # Output general info
        data.to_csv(os.path.join('fitting_model_results', 'predicted_output.csv'))
        outputs = logreg.coef_[0].tolist()
        outputs.append(logreg.intercept_.tolist()[0])
        print(outputs)
        coefficient = -1 * outputs[1] / outputs[0]
        print(f'The model coefficient is: {coefficient}')
        summary_data['Coefficient'] = coefficient
        print('Accuracy of logistic regression classifier on full set: {:.2f}'.format(logreg.score(X, y)))

        # Analyse the events
        for event in fit_events:
            print(f'\nAnalysing event: {event}')
            event_data = data.loc[data['event'] == event]
            event_data.drop('event', axis=1, inplace=True)
            event_data.to_csv(os.path.join('fitting_model_results', f'predicted_output_{event}_{use_parameter}.csv'))
            dfg = event_data.groupby(by=['stable', 'Predicted']).size().reset_index(name='Count')
            dfg.to_csv(os.path.join('fitting_model_results', f'predicted_output_count_{event}_{use_parameter}.csv'))

            print(dfg)
            if dfg.shape[0] > 3:
                stable_correct = dfg.loc[0, 'Count'] / (dfg.loc[0, 'Count'] + dfg.loc[1, 'Count'])
                unstable_correct = dfg.loc[3, 'Count'] / (dfg.loc[2, 'Count'] + dfg.loc[3, 'Count'])
                overall_correct = (dfg.loc[0, 'Count'] + dfg.loc[3, 'Count']) / event_data.shape[0]
                forest_stable = (dfg.loc[0, 'Count'] + dfg.loc[2, 'Count']) / event_data.shape[0]
                summary_data[f'{event}_stable'] = stable_correct
                summary_data[f'{event}_unstable'] = unstable_correct
                summary_data[f'{event}_overall'] = overall_correct
                summary_data[f'{event}_forest_stability'] = forest_stable
            elif dfg.shape[0] > 2:
                stable_correct = dfg.loc[0, 'Count'] / (dfg.loc[0, 'Count'] + dfg.loc[1, 'Count'])
                unstable_correct = 1.0
                overall_correct = (dfg.loc[0, 'Count'] + dfg.loc[2, 'Count']) / event_data.shape[0]
                forest_stable = (dfg.loc[0, 'Count']) / event_data.shape[0]
                summary_data[f'{event}_stable'] = stable_correct
                summary_data[f'{event}_unstable'] = unstable_correct
                summary_data[f'{event}_overall'] = overall_correct
                summary_data[f'{event}_forest_stability'] = forest_stable
            elif dfg.shape[0] > 1:
                stable_correct = dfg.loc[0, 'Count'] / (dfg.loc[0, 'Count'] + dfg.loc[1, 'Count'])
                summary_data[f'{event}_stable'] = stable_correct
            else:
                summary_data[f'{event}_stable'] = 0

        all_data[use_parameter] = summary_data
        if model_parameters[0] == use_parameter:
            outlogreg = logreg

    outdf = pd.DataFrame(all_data).transpose()
    col = outdf.pop('Q00002Y_stable')
    outdf.insert(outdf.shape[1], col.name, col)
    col = outdf.pop('{}_forest_stability'.format(fit_events[0]))
    outdf.insert(outdf.shape[1], col.name, col)
    print(outdf)
    outdf.to_csv(os.path.join('fitting_model_results', 'model_fit_info.csv'))

    return outlogreg


if __name__ == "__main__":
    main()
