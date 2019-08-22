import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from utils import ml_classifier, common_lists

THEORETICAL_DATA = pd.read_csv('files/data_teo_corrected_simple.csv')

FEATURES, LABELS_SETS, SEQ_LIST, ML_CLFS = common_lists()

# iterate over the classification models
for ml_clf in ML_CLFS:
    # define an empty dataframe to save final results for each classification model
    df_final = pd.DataFrame()

    # define the parameterized classifiers names and the corresponding scikit-learn models
    classifiers_names, classifiers = ml_classifier(ml_clf)

    # create numpy arrays filled with ones to temporarily save performance measure results
    results_accuracy = -np.ones((len(classifiers), len(LABELS_SETS)))
    results_weighted_accuracy = -np.ones((len(classifiers), len(LABELS_SETS)))
    results_precision = -np.ones((len(classifiers), len(LABELS_SETS)))
    results_recall = -np.ones((len(classifiers), len(LABELS_SETS)))
    results_f1_score = -np.ones((len(classifiers), len(LABELS_SETS)))

    # iterate over labels sets (i.e. rotamers or rotamer families)
    for ls_cnt, ls in enumerate(LABELS_SETS):

        # define ROSUM matrix for the corresponding labels set
        b_matrix = pd.read_csv('files/b_matrix_{}.csv'.format(ls)) 

        # define lists to save true and predicted labels 
        always_true = [] 
        true_pos = []
        true_list = []
        pred_list = []
        
        # define dictionaries to save true and predicted labels
        true_dict = {cln: [] for cln in classifiers_names}
        pred_dict = {cln: [] for cln in classifiers_names}
        always_true_dict = {cln: [] for cln in classifiers_names}
        true_pos_dict = {cln: [] for cln in classifiers_names}

        # define the training and test sets with a leave-one-out approach
        for rmv in range(len(THEORETICAL_DATA)):
            
            # define the test-set
            test_set = THEORETICAL_DATA.loc[rmv]

            # define the training set
            train_set = THEORETICAL_DATA.drop(THEORETICAL_DATA.index[[rmv]])

            # define a training set with the same dinucleotide sequence as the test-set
            train_set_seq = train_set[train_set.SEQ == test_set.SEQ]

            X_train = train_set_seq[FEATURES]
            X_test = pd.DataFrame(test_set[FEATURES]).T

            y_train = train_set_seq[ls]
            y_test = test_set[ls]

            # iterate over the parameterized classifiers
            for idx, clf in enumerate(classifiers):
                
                # define classifier name
                clf_name = classifiers_names[idx]

                # fit classifier model with train set feautures an labels
                clf.fit(X_train, y_train)

                # define predicted label (rotamer or rotamer family)
                y_pred = (clf.predict(X_test))[0]
           
                true_dict[clf_name].append(y_test)
                pred_dict[clf_name].append(y_pred)

                w_test = []

                # define label assignment weight from ROSUM matrices
                weight = float(b_matrix[(b_matrix['ROT_i'] == y_test) & (b_matrix['ROT_j'] == y_pred)]['a_ij'])
                w_test.append(weight)

                # define always true weighted accuracy: hypothetical case where all the labels are 
                # correctly classified
                weight_always_true = float(b_matrix[(b_matrix['ROT_i'] == y_test) & 
                                                    (b_matrix['ROT_j'] == y_test)]['a_ij'])

                always_true_dict[clf_name].append(weight_always_true)

                # label assignemnt weights for the true positive classified labels
                if y_test == y_pred:
                    true_pos_dict[clf_name].append(weight)
        
        # iterate over the classifiers names to compute and temporarily save performance measures
        for idx,clf_name in enumerate(classifiers_names):
             # define list with true labels for the corresponding classifier
            true_list = true_dict[clf_name]
            
            #define list with predicted labels for the corresponding classifier
            pred_list = pred_dict[clf_name]

            # define true positive predicted labels list for the corresponding classifier
            true_pos = true_pos_dict[clf_name]

            # define 'always true' predicted labels list for the corresponding classifier
            always_true = always_true_dict[clf_name]

            # compute and define all the performance measures
            precision = precision_recall_fscore_support(true_list,pred_list,average='macro')[0]
            recall = precision_recall_fscore_support(true_list,pred_list,average='macro')[1]
            f1_score = precision_recall_fscore_support(true_list,pred_list,average='macro')[2]
            accuracy = accuracy_score(true_list, pred_list)
            w_accuracy = np.sum(true_pos)/np.sum(always_true)

            # save all the performance measures
            results_accuracy[idx,ls_cnt] = accuracy
            results_weighted_accuracy[idx,ls_cnt] = w_accuracy
            results_precision[idx,ls_cnt] = precision
            results_recall[idx,ls_cnt] = recall
            results_f1_score[idx,ls_cnt] = f1_score

    # save the temporary results in a DataFrame with partial results
    for n in range(len(results_accuracy)):
        
        # define a DataFrame with partial results
        df_partial = pd.DataFrame({'02_ClassifierName': classifiers_names[n],
                         '03_Groups': LABELS_SETS,
                         '04_Accuracy': results_accuracy[n],
                         '05_W_Accuracy': results_weighted_accuracy[n],
                         '06_precision': results_precision[n],
                         '07_recall': results_recall[n],
                         '08_f1_score': results_f1_score[n]})

        # append the partial results to the final DataFrame
        df_final = df_final.append(df_partial)

    # save the final results for the corresponding classifier
    df_final.to_csv('results_{}_theotheo.csv'.format(ml_clf), index=False)
