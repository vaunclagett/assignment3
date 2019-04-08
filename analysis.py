#based on joshua Philipose's code, with some modfication
#also based on sklearn nueral network visulaizetion of MLP

from time import time as time
from math import inf as inf
from numpy.random import RandomState

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.feature_selection import SelectKBest as SKB
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import scale
from sklearn import preprocessing

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import completeness_score
from sklearn.metrics import accuracy_score as accuracy

from sklearn.datasets import load_digits
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_breast_cancer

from scipy.stats import kurtosis

import matplotlib.pyplot as plt

import mlrose
import random
import numpy as np
import pandas as pd
import scipy

masterTime = time()

######################################################
#LOAD DATA
np.random.seed(2019)

digits = load_digits()
digits_data = scale(digits.data)
# n_samples, n_features = data.shape
# n_digits = len(np.unique(digits.target))
digits_labels = digits.target
n_samples, n_features = digits_data.shape
print('digits samples ' + str(n_samples) + ' features ' + str(n_features))
print('digits categories:' + str(len(digits_labels)))


rng = RandomState(0)
faces = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces_data =  scale(faces.data)
n_samples, n_features = faces_data.shape
print('faces samples ' + str(n_samples) + ' features ' + str(n_features))
# global centering
faces_data = faces_data - faces_data.mean(axis=0)
# local centering
faces_data -= faces_data.mean(axis=1).reshape(n_samples, -1)
faces_labels = faces.target
print('faces categories:' + str(len(faces_labels)))
print('\n\n\n')
# digits_data = digits_data
# digits_labels = digits_labels
# faces_data = faces_data
# faces_labels = faces_labels
X_train_digits, X_test_digits, Y_train_digits, Y_test_digits = train_test_split(digits_data, digits_labels, random_state=2019, test_size=.3)
X_train_faces, X_test_faces, Y_train_faces, Y_test_faces = train_test_split(faces_data, faces_labels, random_state=2019, test_size=.3)

############################################################################
#DEFINITIONS

#MAKE PLOTS
def plot_xys(xs, ys, lbls, title, xlab, ylab):
    plt.grid()
    plt.title(title)
    
    colors = ['r', 'm', 'g', 'b', 'k', 'c', 'y']

    for i in range(len(xs)):
        plt.plot(xs[i], ys[i], colors[i % len(colors)] + ',-', label=lbls[0])
    #
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.savefig(title + '.png')
    plt.show()

#
def plot_xys2(xs, ys, lbls, title, xlab, ylab):
    plt.grid()
    plt.title(title)
    
    colors = ['r', 'm', 'g', 'b', 'k', 'c', 'y']

    for i in range(len(xs)):
        # print(xs[i])
        # print(ys[i])
        # print(lbls[i])
        plt.plot(xs[i], ys[i], colors[i % len(colors)] + ',-', label=lbls[i])
    #
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.savefig(title + '.png')
    plt.show()

#
def run_kmeans(data,labels,clusters,inits,set_name):
    restarts = 5
    hScores = []
    cScores = []
    vScores = []
    times = []
    for c in clusters:
        km = KMeans(init='k-means++', n_clusters=c, n_init=restarts)

        t0 = time()
        km.fit(data)
        times.append(time()-t0)
        predictions = km.predict(data)
        hScores.append(homogeneity_score(labels, predictions))
        cScores.append(completeness_score(labels, predictions))
        vScores.append(v_measure_score(labels,predictions))

    #
    labels = ['Homoegeneity','Completeness','V']
    scores = [hScores,cScores,vScores]
    a = np.arange(1,4,1)
    # print(a)
    plot_xys2([clusters for i in a], scores, labels,
        set_name + ' K-Means Performance',
        'Clusters', 'Score') 
#
def run_expectation_maximization(data,labels,clusters,set_name):
    restarts = 5
    hScores = []
    cScores = []
    vScores = []
    times = []
    for c in clusters:
        print(c)
        em = GMM(n_components=c)

        t0 = time()
        em.fit(data)
        times.append(time()-t0)
        predictions = em.predict(data)
        hScores.append(homogeneity_score(labels, predictions))
        cScores.append(completeness_score(labels, predictions))
        vScores.append(v_measure_score(labels,predictions))

    #
    labels = ['Homoegeneity','Completeness','V']
    scores = [hScores,cScores,vScores]
    a = np.arange(1,4,1)
    plot_xys2([clusters for i in a], scores, labels,
        set_name + ' EM Performance',
        'Clusters', 'Score') 

#
def run_PCA(data,max_comp,set_name):
    for i in max_comp:
        pca = PCA(n_components=i)
        PCAreducedData = pca.fit_transform(data)

        plot_xys([range(1,i+1)], [pca.explained_variance_], 
            ['Score'], set_name + ' PCA Component Variance',
            'Components', 'Variance') 
        plt.show()

#
def measure_kurtosis(data):
    #sum of kurtosis of r.v. of data
    return np.mean(kurtosis(data))


#
def run_ICA(data,components,set_name):
    kurtoses_gain = []
    m = max(components)
    for i in components:
        kurtosis_a = np.mean(np.abs(kurtosis(data)))

        ica = FastICA(n_components=i,max_iter=1000)
        ICAreducedData = ica.fit_transform(data)

        kurtosis_b = i*np.mean(np.abs(kurtosis(ICAreducedData)))/m
        # *i/m because need to normalize the kurtosis against
        # the change in num components
        kurtoses_gain.append([kurtosis_a/kurtosis_b])
    #
    # print(kurtoses_gain)
    plot_xys([components], [kurtoses_gain], ['Kurtosis'],
        set_name + ': ICA Sum of Kurtosis as a Function of Components',
        'Components', 'Kurtosis Gain')

#
def run_GRP(data,labels,components,set_name):
    vScores = []
    for i in components:
        grp = GRP(n_components=i)
        GRPreducedData = grp.fit_transform(data)

        km = KMeans(n_clusters=10)
        km.fit(GRPreducedData)        
        predictions = km.predict(GRPreducedData)
        vScores.append(v_measure_score(labels, predictions))
        
    #
    plot_xys([components], [vScores], ['V'],
        set_name + ': GRP V of 10 KM clusters trained on \nRandomized Projections',
        'Components', 'V Score')

#
def run_SKB(data,labels,k,set_name):
    skb = SKB(f_classif, k=k)
    # skb = SKB(chi2, k=k)
    SKBreducedData = skb.fit_transform(data, labels)
    # print(skb.scores_)

    scores = skb.scores_


    plot_xys([list(range(1,len(scores)+1,1))], 
        [scores], ['Score'],
        set_name + ': SelectKBest Scores',
        'Components', 'Score')

    #FFT to figure out graph oscillation
    #theory - osc from 64 pixel edge size

    # fft = np.abs(np.real(np.fft.fft(scores)))
    # print(len(fft))
    # # fft = fft[1:10]
    # # fft = fft[round(len(fft/2)):len(fft)]
    # n = 10
    # print(np.argsort(fft)[-n:]/64)
    # plt.plot(fft)
    # plt.show()


    # print(set_name)
    # indeces = np.arange(1,len(scores)+1,1)
    # print(indeces)
    # print(indeces[scores > 35])

#
def run_kmeans_dimredux(data,labels,clusters,set_name,
    pca_max_comp,ica_components,grp_components,skb_k):

    #dimmensionality reduction models
    pca = PCA(n_components=pca_max_comp)
    PCAreducedData = pca.fit_transform(data)

    ica = FastICA(n_components=ica_components)
    ICAreducedData = ica.fit_transform(data)

    grp = GRP(n_components=grp_components)
    GRPreducedData = grp.fit_transform(data)

    skb = SKB(f_classif, k=skb_k)
    SKBreducedData = skb.fit_transform(data, labels)

    models = [pca,ica,grp,skb]
    reduced_data = [PCAreducedData,ICAreducedData,
    GRPreducedData,SKBreducedData]

    super_hScores = []
    super_cScores = []
    super_vscores = []
    super_times = []

    restarts = 5

    for i,d in enumerate(reduced_data):
        hScores = []
        cScores = []
        vscores = []
        times = []
        for c in clusters:
            # print(c)
            km = KMeans(init='k-means++', n_clusters=c, n_init=restarts)

            t0 = time()
            km.fit(d)
            times.append(time()-t0)
            predictions = km.predict(d)
            hScores.append(homogeneity_score(labels, predictions))
            cScores.append(completeness_score(labels, predictions))
            vscores.append(v_measure_score(labels,predictions))

        #
        super_hScores.append(hScores)
        super_cScores.append(cScores)
        super_vscores.append(vscores)
        super_times.append(times)
    #

    labels = ['PrincipalCA','IndependentCA','RandomProj','SelectKBest']

    plot_xys2([clusters for m in models], super_hScores, labels,
        set_name + ' K-Means - Homogeneity Scores by Cluster',
        'Number of Clusters', 'Homoegeneity Score') 

    plot_xys2([clusters for m in models], super_cScores, labels,
        set_name + ' K-Means - Completness Scores by Cluster',
        'Number of Clusters', 'Completeness Score') 

    plot_xys2([clusters for m in models], super_times, labels,
        set_name + ' K-Means - Times by Cluster',
        'Number of Clusters', 'Time (s)')

    plot_xys2([clusters for m in models], super_vscores, labels,
        set_name + ' K-Means - V Scores by Cluster',
        'Number of Clusters', 'V Measure Score')
#\

#
def run_em_dimredux(data,labels,clusters,set_name,
    pca_components,ica_components,grp_components,skb_k):

     #############################################

    #dimmensionality reduction models
    pca = PCA(n_components=pca_components)
    PCAreducedData = pca.fit_transform(data)

    ica = FastICA(n_components=ica_components)
    ICAreducedData = ica.fit_transform(data)

    grp = GRP(n_components=grp_components)
    GRPreducedData = grp.fit_transform(data)

    skb = SKB(f_classif, k=skb_k)
    SKBreducedData = skb.fit_transform(data, labels)

    models = [pca,ica,grp,skb]
    reduced_data = [PCAreducedData,ICAreducedData,
    GRPreducedData,SKBreducedData]

    super_hScores = []
    super_cScores = []
    super_vscores = []
    super_times = []

    restarts = 5

    for i,d in enumerate(reduced_data):
        hScores = []
        cScores = []
        vscores = []
        times = []
        for c in clusters:
            # print(c)
            em = GMM(n_components=c)

            t0 = time()
            em.fit(d)
            times.append(time()-t0)
            predictions = em.predict(d)
            hScores.append(homogeneity_score(labels, predictions))
            cScores.append(completeness_score(labels, predictions))
            vscores.append(v_measure_score(labels,predictions))

        #
        super_hScores.append(hScores)
        super_cScores.append(cScores)
        super_vscores.append(vscores)
        super_times.append(times)
    #

    labels = ['PrincipalCA','IndependentCA','RandomProj','SelectKBest']

    plot_xys2([clusters for m in models], super_hScores, labels,
        set_name + ' EM - Homogeneity Scores by Cluster',
        'Number of Clusters', 'Homoegeneity Score') 

    plot_xys2([clusters for m in models], super_cScores, labels,
        set_name + ' EM - Completness Scores by Cluster',
        'Number of Clusters', 'Completeness Score') 

    plot_xys2([clusters for m in models], super_times, labels,
        set_name + ' EM - Times by Cluster',
        'Number of Clusters', 'Time (s)')

    plot_xys2([clusters for m in models], super_vscores, labels,
        set_name + ' EM - V Scores by Cluster',
        'Number of Clusters', 'V Measure Score')

#nice additional graphics:
    #PCA voronoi colors for all?
    #score map for select K best

#
def run_nn(data,labels,layers):

    mlp = MLPClassifier(hidden_layer_sizes=layers, 
        max_iter=50, alpha=1e-4,
        solver='sgd',tol=1e-2, random_state=1,
        learning_rate_init=.1)
    ntests = 4
    cross_train_accuracy = np.empty(ntests)
    cross_test_accuracy = np.empty(ntests)



    for n,m in enumerate(np.arange(0,ntests)):
        # print(m)
        # X_train, X_test, y_train, y_test = train_test_split(data,
        #     labels, test_size = 0.2, random_state=1, stratify=labels)
        # mlp.fit(X_train,y_train)
        X_train, X_test, y_train, y_test = train_test_split(data,
            labels, test_size = 0.4, random_state=1)
        mlp.fit(X_train,y_train)

        cross_train_accuracy[m] = mlp.score(X_train, y_train)
        cross_test_accuracy[m] = mlp.score(X_test, y_test)
        # print(cross_test_accuracy[m])

    #
    train_accuracy = sum(cross_train_accuracy)/ntests
    test_accuracy = sum(cross_test_accuracy)/ntests
    # print(train_accuracy)
    print(test_accuracy)

    return test_accuracy

#
def nn_dimredux(data,labels,layers,set_name,
    components):

    #
    super_test_scores = []
    test_scores = []
    for c in components:
        pca = PCA(n_components=c)
        PCAreducedData = pca.fit_transform(data)
        test_scores.append(run_nn(PCAreducedData,labels,layers))
    #
    print('    ' +str(max(test_scores)))
    super_test_scores.append(test_scores)
    test_scores = []
    for c in components:
        ica = FastICA(n_components=c)
        ICAreducedData = ica.fit_transform(data)
        test_scores.append(run_nn(ICAreducedData,labels,layers))
    #
    print('    ' +str(max(test_scores)))
    super_test_scores.append(test_scores)
    test_scores = []
    for c in components:
        grp = GRP(n_components=c)
        GRPreducedData = grp.fit_transform(data)
        test_scores.append(run_nn(GRPreducedData,labels,layers))
    #
    print('    ' +str(max(test_scores)))
    super_test_scores.append(test_scores)
    test_scores = []
    for c in components:
        skb = SKB(f_classif, k=c)
        SKBreducedData = skb.fit_transform(data,labels)
        test_scores.append(run_nn(SKBreducedData,labels,layers))
    #
    print('    ' +str(max(test_scores)))
    super_test_scores.append(test_scores)

    labels = ['PrincipalCA','IndependentCA','RandomProj','SelectKBest']

    a = np.arange(1,5,1)
    plot_xys2([components for i in a], super_test_scores, labels,
        set_name + ' NN from Dim Red. Best Accuracy Scores',
        'Clusters', 'Test Accruacy')


    # #dimmensionality reduction models
    # pca = PCA(n_components=pca_max_comp)
    # PCAreducedData = pca.fit_transform(data)

    # ica = FastICA(n_components=ica_components)
    # ICAreducedData = ica.fit_transform(data)

    # grp = GRP(n_components=grp_components)
    # GRPreducedData = grp.fit_transform(data)

    # skb = SKB(f_classif, k=skb_k)
    # SKBreducedData = skb.fit_transform(data, labels)

    # redux_models = [pca,ica,grp,skb]
    # reduced_data = [PCAreducedData,ICAreducedData,
    # GRPreducedData,SKBreducedData]

    # test_scores = []

    # for i,m in enumerate(redux_models):
    #     test_scores.append(run_nn(reduced_data[i],labels,layers))

#
def nn_cluster_dimredux(data,labels,layers,set_name,
    pca_max_comp,ica_components,grp_components,skb_k,
    km_clusters,em_clusters):

    #dimmensionality reduction models
    pca = PCA(n_components=pca_max_comp)
    PCAreducedData = pca.fit_transform(data)

    ica = FastICA(n_components=ica_components)
    ICAreducedData = ica.fit_transform(data)

    grp = GRP(n_components=grp_components)
    GRPreducedData = grp.fit_transform(data)

    skb = SKB(f_classif, k=skb_k)
    SKBreducedData = skb.fit_transform(data, labels)

    redux_models = [pca,ica,grp,skb]
    reduced_data = [PCAreducedData,ICAreducedData,
    GRPreducedData,SKBreducedData]

    #cluster dis boyz
    km = KMeans(init='k-means++', n_clusters=km_clusters, n_init=5)
    em = GMM(n_components=em_clusters)

    cluster_models = [km,em]

    test_scores = []
    # #

    km = KMeans(init='k-means++', n_clusters=km_clusters, n_init=30)
    clustered = km.fit_transform(PCAreducedData)
    test_scores.append(run_nn(clustered,labels,layers))

    em = GMM(n_components=em_clusters)
    clustered = em.fit_predict(PCAreducedData).reshape(-1,1)
    test_scores.append(run_nn(clustered,labels,layers))


    km = KMeans(init='k-means++', n_clusters=km_clusters, n_init=30)
    clustered = km.fit_transform(ICAreducedData)
    test_scores.append(run_nn(clustered,labels,layers))

    em = GMM(n_components=em_clusters)
    clustered = em.fit_predict(ICAreducedData).reshape(-1,1)
    test_scores.append(run_nn(clustered,labels,layers))


    km = KMeans(init='k-means++', n_clusters=km_clusters, n_init=30)
    clustered = km.fit_transform(GRPreducedData)
    test_scores.append(run_nn(clustered,labels,layers))

    em = GMM(n_components=em_clusters)
    clustered = em.fit_predict(GRPreducedData).reshape(-1,1)
    test_scores.append(run_nn(clustered,labels,layers))


    km = KMeans(init='k-means++', n_clusters=km_clusters, n_init=30)
    clustered = km.fit_transform(SKBreducedData)
    test_scores.append(run_nn(clustered,labels,layers))

    em = GMM(n_components=em_clusters)
    clustered = em.fit_predict(SKBreducedData).reshape(-1,1)
    test_scores.append(run_nn(clustered,labels,layers))

    return test_scores
    # print(test_scores)

# 
def nn_kmeans_dimredux(data,labels,layers,set_name,
    pca_max_comp,ica_components,grp_components,skb_k,
    km_clusters):

    #
    pca = PCA(n_components=pca_max_comp)
    PCAreducedData = pca.fit_transform(data)

    ica = FastICA(n_components=ica_components)
    ICAreducedData = ica.fit_transform(data)

    grp = GRP(n_components=grp_components)
    GRPreducedData = grp.fit_transform(data)

    skb = SKB(f_classif, k=skb_k)
    SKBreducedData = skb.fit_transform(data, labels)

    redux_models = [pca,ica,grp,skb]
    reduced_data = [PCAreducedData,ICAreducedData,
    GRPreducedData,SKBreducedData]

    #km
    PCA_scores = []
    ICA_scores = []
    GRP_scores = []
    SKB_scores = []
    for c in km_clusters:
        km = KMeans(init='k-means++', n_clusters=c, n_init=5)

        clustered = km.fit_transform(PCAreducedData)
        PCA_scores.append(run_nn(clustered,labels,layers))

        clustered = km.fit_transform(ICAreducedData)
        ICA_scores.append(run_nn(clustered,labels,layers))

        clustered = km.fit_transform(GRPreducedData)
        GRP_scores.append(run_nn(clustered,labels,layers))

        clustered = km.fit_transform(SKBreducedData)
        SKB_scores.append(run_nn(clustered,labels,layers))

    #
    test_scores = [PCA_scores,ICA_scores,GRP_scores,SKB_scores]
    the_labels = ['PCA','ICA','GRP','SKB']
    a = np.arange(1,5,1)
    plot_xys2([km_clusters for i in a], test_scores, the_labels,
        set_name + ' NN Trained on K-Means from \n' +
        'Dim Reduced Set',
        'Clusters', 'Testing Accuracy') 

# 
def nn_em_dimredux(data,labels,layers,set_name,
    pca_max_comp,ica_components,grp_components,skb_k,
    em_clusters):

    #
    pca = PCA(n_components=pca_max_comp)
    PCAreducedData = pca.fit_transform(data)

    ica = FastICA(n_components=ica_components)
    ICAreducedData = ica.fit_transform(data)

    grp = GRP(n_components=grp_components)
    GRPreducedData = grp.fit_transform(data)

    skb = SKB(f_classif, k=skb_k)
    SKBreducedData = skb.fit_transform(data, labels)

    redux_models = [pca,ica,grp,skb]
    reduced_data = [PCAreducedData,ICAreducedData,
    GRPreducedData,SKBreducedData]


    PCA_scores = []
    ICA_scores = []
    GRP_scores = []
    SKB_scores = []
    for c in em_clusters:
        em = GMM(n_components=c)

        clustered = em.fit_predict(PCAreducedData).reshape(-1,1)
        PCA_scores.append(run_nn(clustered,labels,layers))

        em = GMM(n_components=c)
        clustered = em.fit_predict(ICAreducedData).reshape(-1,1)
        ICA_scores.append(run_nn(clustered,labels,layers))

        em = GMM(n_components=c)
        clustered = em.fit_predict(GRPreducedData).reshape(-1,1)
        GRP_scores.append(run_nn(clustered,labels,layers))

        em = GMM(n_components=c)
        clustered = em.fit_predict(SKBreducedData).reshape(-1,1)
        SKB_scores.append(run_nn(clustered,labels,layers))

    #
    test_scores = [PCA_scores,ICA_scores,GRP_scores,SKB_scores]
    labels = ['PCA','ICA','GRP','SKB']
    a = np.arange(1,5,1)
    plot_xys2([em_clusters for i in a], test_scores, labels,
        set_name + ' NN Trained on EM from \n' +
        'Dim Reduced Set',
        'Clusters', 'Testing Accuracy') 


#######################################################
#RUN ANALYSIS

digits_name = 'Digits'
faces_name = ' Olivetti Faces'



# #kmeans, digits
# print('K-Means++ Digits\n')
# clusters = list(range(2, 50,1))
# # clusters = list(range(2, 12,2))
# inits = list([5])
# run_kmeans(digits_data,digits_labels,clusters,inits,digits_name)
# #23 or 30 clusters by v score

# #kmeans faces
# print('K-Means++ Faces\n')
# clusters = list(range(2, 120,1))
# inits = list([5])
# run_kmeans(faces_data,faces_labels,clusters,inits,faces_name)
# #35 clusters by v score

# #EM, Digits
# print('EM Digits\n')
# # clusters = list(range(2, 50,1))
# clusters = list(range(2, 50,2))
# run_expectation_maximization(digits_data,digits_labels,clusters,digits_name)
# # 26 or 32 clusters by v score

# print('EM Faces\n')
# #7.5 seconds/cluster runtime
# clusters = list(range(2, 62,20)) #(2,22,42)
# # favors maximizimg range
# clusters = list(range(50, 70,5)) #(2,22,42)
# #we gonna use 65,this is killing my computer at 450 seconds per iter
# run_expectation_maximization(faces_data,faces_labels,clusters,faces_name)
# #14,20,30 all above 14 reasonably high v score

# #PCA digits
# print('PCA digits\n')
# components = list([64])
# run_PCA(digits_data,components,digits_name)
# #comp(15,20,25,30,35,40) -> var(.15,.095,.60,.50,.45,.40)

# #PCA Faces
# print('PCA faces\n')
# components = list([400])
# run_PCA(faces_data,components,faces_name)
# #comp(25,50,100) -> var(.05,.025,.01)

# #ICA digits
# print('ICA digits\n')
# measure_kurtosis(digits_data.data)
# components = list(range(2,64,2))
# run_ICA(digits_data,components,digits_name)
# #try 30

# #ICA faces
# print('ICA faces\n')
# measure_kurtosis(faces_data.data)
# components = list(range(2,400,25))
# run_ICA(faces_data,components,faces_name)
# #try 200 componnets, stabliized kurtosis


# #GRP digits
# print('GRP digits\n')
# components = list(range(2,64,2))
# run_GRP(digits_data,digits_labels,components,digits_name)
# # KM(10): comp(40,52,62) -> Vscore(.6,.62,.64)

# #GRP faces
# print('GRP faces\n')
# components = list(range(2,400,15))
# #note that we are limited by number of samples
# #because 4096 dimm, might be better to have more
# run_GRP(faces_data,faces_labels,components,faces_name)
# # KM(10): comp(210,300) -> Vscore(.52,.55)

# #SKB digits
# print('SKB digits\n')
# k = 64;
# run_SKB(digits_data,digits_labels,k,digits_name)
# #note-discont grpah because NaN scores
# #comp(23,26,28,34,43) = score(225,295,230,320,265)


# #SKB faces
# print('SKB faces\n')
# k = 400;
# run_SKB(faces_data,faces_labels,k,faces_name)
# #try 909 seems pretty good?


# # kmeans-dimredux digits
# print('kmeans dimredux faces\n')
# clusters = list(range(2, 50,1))
# # clusters = list(range(2, 50,10)) #testing
# pca_max_comp = 25
# ica_components = 30
# grp_components = 62
# skb_k = 34
# run_kmeans_dimredux(digits_data,digits_labels,clusters,
#     digits_name,pca_max_comp,ica_components,grp_components,
#     skb_k)

# # kmeans-dimredux faces
# print('kmeans dimredux faces')
# clusters = list(range(2, 80,1))
# # clusters = list(range(2, 80,10)) #testing
# pca_max_comp = 100
# ica_components = 200
# grp_components = 300
# skb_k = 909
# run_kmeans_dimredux(faces_data,faces_labels,clusters,
#     faces_name + ' Improved',pca_max_comp,ica_components,
#     grp_components,skb_k)

# # EM-dimredux digits
# print('EM dimredux faces\n')
# clusters = list(range(2, 50,1))
# # clusters = list(range(2, 50,10)) #testing
# pca_max_comp = 25
# ica_components = 30
# grp_components = 62
# skb_k = 34
# run_em_dimredux(digits_data,digits_labels,clusters,
#     digits_name,pca_max_comp,ica_components,grp_components,
#     skb_k)

# # EM-dimredux faces
# print('EM dimredux faces\n')
# clusters = list(range(2, 50,1))
# # clusters = list(range(2, 50,10)) #testing
# pca_max_comp = 25
# ica_components = 30
# grp_components = 62
# skb_k = 34
# run_em_dimredux(faces_data,faces_labels,clusters,
#     faces_name + ' Improved',pca_max_comp,ica_components,
#     grp_components,skb_k)

# #baseline nn
# print('nn')
# #run_nn(digits_data,digits_labels,[10])
# run_nn(digits_data,digits_labels,[5])

#nn(dim_redux)
print('nn, dim redux')
components = list(range(2, 50,2))
layers = [5]
nn_dimredux(digits_data,digits_labels,layers,digits_name,
    components)


# print('nn ,kmeans,dimredux')
# # pca_max_comp = 20 #.09
# # ica_components = 10 # .17
# # grp_components = 10 # .20
# # skb_k = 10 # .32
# # pca_max_comp = 15 #.07
# # ica_components = 15 # ..16
# # grp_components = 15 # ..15
# # skb_k = 15 # .30
# # pca_max_comp = 30 #.07
# # ica_components = 5 # ..16
# # grp_components = 5 # ..15
# # skb_k = 5 # .30
# # pca_max_comp = 30 #.07
# # ica_components = 5 # .17
# # grp_components = 5 # ..08
# # skb_k = 5 # .36
# pca_max_comp = 20 #.07
# ica_components = 10 # .17
# grp_components = 10
# skb_k = 5 # .36
# layers = [5]
# km_clusters = list(range(2, 19,1))
# nn_kmeans_dimredux(digits_data,digits_labels,layers,digits_name,
#     pca_max_comp,ica_components,grp_components,skb_k,
#     km_clusters)

# print('nn ,EM,dimredux')
# # pca_max_comp = 20 #.46
# # ica_components = 10 # .17
# # grp_components = 10 # .41
# # skb_k = 5 #.26
# # pca_max_comp = 15 #.46
# # ica_components = 15 # .43
# # grp_components = 15 # .46
# # skb_k = 15 #.47
# # pca_max_comp = 17 #.47 @ 9
# # ica_components = 20 # .43 @ 13
# # grp_components = 15 # .42 @ 8
# # skb_k = 15 #.4 @ 7
# pca_max_comp = 20 #.47 @ 9
# ica_components = 20 # .43 @ 13
# grp_components = 15 # .42 @ 8
# skb_k = 15 #.4 @ 7

# layers = [5]
# emclusters = list(range(2, 19,1))
# nn_em_dimredux(digits_data,digits_labels,layers,digits_name,
#     pca_max_comp,ica_components,grp_components,skb_k,
#     emclusters)