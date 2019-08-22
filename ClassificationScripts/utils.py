import numpy as np


class RandomGuess:
    def __init__(self):
        self.unique_labels = None

    def fit(self, X, y):
        """Fit the RandomGuess model according to the given training data.
        Sets the unique labels for this dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)
        Returns
        -------
        self : object

        """
        self.unique_labels = np.unique(y)
        return self

    def predict(self, X):
        """Predict class for X.
        Returns a random label for each value of X from the unique labels
        found in the training dataset.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted random classes.
        """
        return np.random.choice(self.unique_labels, size=X.shape[0])


def common_lists(common_lists):
    """
    Returns a group of lists common to the three classification scripts
        features : list with the features names, i.e. 13C' CS
        labels_sets : list with the labels sets names, i.e. rotamer o families of rotamers
        seq_list : list with the 16 combinations of RNA dinucleotide sequences
        ml_clfs : list with the 6 classifier model abreviated names
    
    Parameters
    ---------- 
    common_lists : str
    """    
    features = ["C1'(i-1)_CS","C2'(i-1)_CS","C3'(i-1)_CS","C4'(i-1)_CS","C5'(i-1)_CS",
            "C1'(i)_CS","C2'(i)_CS","C3'(i)_CS","C4'(i)_CS","C5'(i)_CS"]
    labels_sets =  ['46_rotamers', 'δδ_families', 'δδγ_families', 'δδα_families', 'δδαγ_families', 'αγ_families', 
                'A_noA_families', 'A*_noA*_families']
    seq_list = ['AA','AC','AG','AU','CA','CC','CG','CU','GA','GC','GG','GU','UA','UC','UG','UU']

    ml_clfs = ['NN', 'DT', 'RF', 'MLP', 'SVM', 'RAND']
    
    return(features, labels_sets, seq_list, ml_clfs)


def ml_classifier(ml_clf):
    """
    Generates a list of classifier names and a list with the corresponding 
    paremeterized scikit-learn classifiers
    
    Parameters
    ---------- 
    ml_clf : str
        classifier model ('NN', 'DT', 'RF', 'MLP', 'SVM' or 'RAND') where
        'NN' means Nearest neighbor
        'DT' means Decision Tree
        'RF' means Random Forest
        'MLP' means Multi-Layer Perceptron, a type of neural network
        'SVM' means Support Vector Machine
        'RAND' means Random Guess
    """    
    if ml_clf == 'NN':
        from sklearn.neighbors import KNeighborsClassifier

        classifiers_names = ["1-NN", "2-NN", "3-NN", "4-NN", "5-NN"]

        classifiers = [
            KNeighborsClassifier(1),
            KNeighborsClassifier(2),
            KNeighborsClassifier(3),
            KNeighborsClassifier(4),
            KNeighborsClassifier(5)]
    
    if ml_clf == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        
        classifiers_names = ["DT gini 20", "DT gini all", "DT entropy 20", "DT entropy all"]

        classifiers = [
            DecisionTreeClassifier(criterion='gini', max_depth=20),
            DecisionTreeClassifier(criterion='gini', max_depth=None),
            DecisionTreeClassifier(criterion='entropy', max_depth=20),
            DecisionTreeClassifier(criterion='entropy', max_depth=None)]

    if ml_clf == 'RF':
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        
        classifiers_names = ["RF gini 20 1", "RF gini 20 auto", "RF entropy 20 1", "RF entropy 20 auto"]

        classifiers = [
            RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=10, max_features=1),
            RandomForestClassifier(criterion='gini', max_depth=20, n_estimators=10, max_features='auto'),
            RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=10, max_features=1),
            RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=10, max_features='auto')]
    
    if ml_clf == 'MLP':
        from sklearn.neural_network import MLPClassifier
        
        classifiers_names = ["MLP 0.0001 500 lbfgs", "MLP 0.0001 500 sgd", "MLP 0.0001 500 adam", 
                             "MLP 0.0001 500 sgd", "MLP 0.0001 750 sgd", "MLP 0.0001 1000 sgd"]

        classifiers = [
            MLPClassifier(alpha=0.0001, max_iter=500, solver='lbfgs'),
            MLPClassifier(alpha=0.0001, max_iter=500, solver='sgd'),
            MLPClassifier(alpha=0.0001, max_iter=500, solver='adam'),
            MLPClassifier(alpha=0.0001, max_iter=500, solver='sgd'),
            MLPClassifier(alpha=0.0001, max_iter=750, solver='sgd'),
            MLPClassifier(alpha=0.0001, max_iter=1000, solver='sgd')]
    
    if ml_clf == 'SVM':
        from sklearn.svm import SVC
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        classifiers_names = ["L-SVM 0.025", "L-SVM 0.1", "L-SVM 0.5", "L-SVM 1.0", "L-SVM 2.0",
                             "RBF-SVM 0.025", "RBF-SVM 0.1", "RBF-SVM 0.5", "RBF-SVM 1.0", "RBF-SVM 2.0"]

        classifiers = [
            SVC(kernel="linear", C=0.025),
            SVC(kernel="linear", C=0.1),
            SVC(kernel="linear", C=0.5),
            SVC(kernel="linear", C=1.0),
            SVC(kernel="linear", C=2.0),
            SVC(kernel="rbf", C=0.025),
            SVC(kernel="rbf", C=0.1),
            SVC(kernel="rbf", C=0.5),
            SVC(kernel="rbf", C=1.0),
            SVC(kernel="rbf", C=2.0)]
    
    if ml_clf == 'RAND':
        classifiers_names, classifiers = (['Random Guess'],[RandomGuess()])
                
    return(classifiers_names, classifiers)
