import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

from networkx.readwrite import json_graph

def cyclomatic_complexity(G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    P = nx.components.number_strongly_connected_components(G)
    return E - N + P

def number_of_simple_cycles(G):
    if G.number_of_nodes() > 150:
        return -1
    sc = nx.algorithms.cycles.simple_cycles(G)
    return len(list(sc))

def preprocess_dataset(dataset, with_semantic_column=True):
    if with_semantic_column:
        columns = ['id', 'nodes', 'cyclomatic_complexity', 'number_of_simple_cycles', \
        'A', 'B', 'C', 'D', 'E', 'F', 'semantic']
    else:
        columns = ['id', 'nodes', 'cyclomatic_complexity', 'number_of_simple_cycles', \
        'A', 'B', 'C', 'D', 'E', 'F']
   
    data = [None] * len(dataset)
    for i in range(len(dataset)):
        G = json_graph.adjacency_graph(dataset.cfg[i])
        print(i, G.number_of_nodes(), cyclomatic_complexity(G))
        instruction_classes = dict()
        instruction_classes['A'] = dataset.lista_asm[i].count('mov') + dataset.lista_asm[i].count('lea')
        instruction_classes['B'] = dataset.lista_asm[i].count('add') + dataset.lista_asm[i].count('mul')
        instruction_classes['F'] = dataset.lista_asm[i].count('jge') + dataset.lista_asm[i].count('jmp') + \
                                    dataset.lista_asm[i].count('jne') + dataset.lista_asm[i].count('jl') + \
                                        dataset.lista_asm[i].count('je')
        instruction_classes['C'] = dataset.lista_asm[i].count('xmm')
        instruction_classes['D'] = dataset.lista_asm[i].count('and') + dataset.lista_asm[i].count('or') + \
                                    dataset.lista_asm[i].count('not') + dataset.lista_asm[i].count('xor')                        
        instruction_classes['E'] = dataset.lista_asm[i].count('call')
        entry = [dataset.id[i],\
            G.number_of_nodes(),\
            cyclomatic_complexity(G),\
            number_of_simple_cycles(G),\
            instruction_classes['A'], instruction_classes['B'], instruction_classes['C'],\
            instruction_classes['D'], instruction_classes['E'], instruction_classes['F']]
        if with_semantic_column:
            entry.append(dataset.semantic[i])
        data[i] = entry
    #print(len(data))
    #print(data[0:2])
    return pd.DataFrame.from_records(data, columns=columns)

def plot_confusion_matrix(y_true, prediction, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, prediction)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, prediction)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def svm_classifier(dataset, kernel):
    from sklearn import svm
    from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
    from sklearn import metrics
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    train, test = train_test_split(dataset, test_size=0.2)
    model = svm.SVC(kernel=kernel, C=1)
    columns = ['id', 'nodes', 'cyclomatic_complexity', 'number_of_simple_cycles', \
        'A', 'B', 'C', 'D', 'E', 'F']
    X_train = train.loc[:, columns]
    X_test = test.loc[:, columns]
    model.fit(X_train, train.semantic)
    acc_train = model.score(X_train, train.semantic)
    acc_test = model.score(X_test, test.semantic)
    prediction = model.predict(X_test)
    print(acc_train, acc_test)
    print("Accuracy: ", metrics.accuracy_score(test.semantic, prediction))
    class_names = unique_labels(dataset.semantic)
    print(classification_report(test.semantic, prediction, labels=None, target_names=class_names, digits=3))
    cm = confusion_matrix(test.semantic, prediction, labels=None, sample_weight=None)
    print(cm)
    plot_confusion_matrix(test.semantic, prediction, classes=class_names, normalize=False)
    #cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=15)
    #scores = cross_val_score(model, dataset.loc[:,columns], dataset.semantic, cv=cv)
    #print(scores)  
    #print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model

def decision_tree_classifier(dataset, criterion, splitter):
    from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
    from sklearn import metrics
    from sklearn import tree
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    train, test = train_test_split(dataset, test_size=0.2)
    model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter)
    columns = ['id', 'nodes', 'cyclomatic_complexity', 'number_of_simple_cycles', \
        'A', 'B', 'C', 'D', 'E', 'F']
    X_train = train.loc[:, columns]
    X_test = test.loc[:, columns]
    model.fit(X_train, train.semantic)
    acc_train = model.score(X_train, train.semantic)
    acc_test = model.score(X_test, test.semantic)
    prediction = model.predict(X_test)
    print(acc_train, acc_test)
    print("Accuracy: ", metrics.accuracy_score(test.semantic, prediction))
    class_names = unique_labels(dataset.semantic)
    print(classification_report(test.semantic, prediction, labels=None, target_names=class_names, digits=3))
    cm = confusion_matrix(test.semantic, prediction, labels=None, sample_weight=None)
    print(cm)
    plot_confusion_matrix(test.semantic, prediction, classes=class_names, normalize=False)
    #cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=15)
    #scores = cross_val_score(model, dataset.loc[:,columns], dataset.semantic, cv=cv)
    #print(scores)  
    #print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return model


def calculate_blind_test(model, blind_test_file):
    dataset = pd.read_json(blind_test_file, lines=True)
    new_dataset = preprocess_dataset(dataset, False)
    prediction = model.predict(new_dataset)
    return prediction

if __name__ == "__main__":
    PREPROCESS = False
    dataset = None
    new_dataset = None

    if PREPROCESS:
        dataset = pd.read_json('noduplicatedataset.json', lines=True)
        new_dataset = preprocess_dataset(dataset)
        new_dataset.to_json('preprocessed_data.json')
    else:
        new_dataset = pd.read_json('preprocessed_data.json', lines=False)
        #start_time = time.process_time()
        #svm_model1 = svm_classifier(new_dataset, 'linear')
        #print("SVM ran in  %s seconds." % (time.process_time() - start_time))
        #svm_model2 = svm_classifier(new_dataset, 'rbf')
        #svm_model3 = svm_classifier(new_dataset, 'poly')
        #nbc_model1 = decision_tree_classifier(new_dataset, 'entropy', 'best')
        #nbc_model2 = decision_tree_classifier(new_dataset, 'gini', 'random')
        start_time1 = time.process_time()
        nbc_model3 = decision_tree_classifier(new_dataset, 'entropy', 'random')
        print("DT ran in  %s seconds." % (time.process_time() - start_time1))
        #nbc_model4 = decision_tree_classifier(new_dataset, 'gini', 'best')
        prediction = calculate_blind_test(nbc_model3, 'nodupblindtest.json')
        np.savetxt('1938032.txt', prediction, fmt="%s")