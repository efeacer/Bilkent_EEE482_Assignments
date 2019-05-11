'''
Code for EEE482 homework 4, 2019 Spring. 
Author: Efe Acer
'''

import sys

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import h5py # to be able to use v 7.3 .mat files in the Python 
from sklearn.decomposition import PCA 
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from scipy.spatial.distance import cdist # Python equivalent of pdist2
from scipy.optimize import bisect

question = sys.argv[1]

def efe_acer_21602217_hw4(question):
    if question == '1':
        
        with h5py.File('hw4_data1.mat', 'r') as file:
            data_keys = list(file.keys())

        data = dict()
        with h5py.File('hw4_data1.mat', 'r') as file:
            for key in data_keys:
                data[key] = np.array(file[key])
                print('Shape of the data associated with %s:' % key,
                    np.shape(data[key]), '\n')

        # QUESTION 1
        print('QUESTION 1')

        faces = data['faces'].T

        # A sample stimuli
        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(faces[7].reshape(32, 32).T, cmap=plt.cm.bone)
        plt.title('Sample Stimulus')
        plt.show(block=False)

        # PART A
        print('PART A')

        pca = PCA(100, whiten=True) 
        pca.fit(faces) 

        plt.figure(figure_num)
        figure_num += 1
        plt.plot(pca.explained_variance_ratio_)
        plt.xlabel('PC')
        plt.ylabel('Proportion of Variance Explained')
        plt.title('Proportion of Variance Explained by Each Individual PC')
        plt.grid()
        plt.show(block=False)

        fig, axes = plt.subplots(5, 5, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(pca.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f3.png', bbox_inches='tight')

        # PART B
        print('PART B')

        # Obtain the reconstructions
        faces_PCA_10 = (faces - pca.mean_).dot(pca.components_[0:10].T).dot(pca.components_[0:10]) + pca.mean_
        faces_PCA_25 = (faces - pca.mean_).dot(pca.components_[0:25].T).dot(pca.components_[0:25]) + pca.mean_
        faces_PCA_50 = (faces - pca.mean_).dot(pca.components_[0:50].T).dot(pca.components_[0:50]) + pca.mean_

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f4.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_PCA_10[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f5.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_PCA_25[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f6.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_PCA_50[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f7.png', bbox_inches='tight')

        # Compute the mean and standard deviation of the reconstruction losses
        losses_PCA_10 = (faces - faces_PCA_10) ** 2
        MSE_PCA_10, std_PCA_10 = np.mean(losses_PCA_10), np.std(np.mean(losses_PCA_10, axis=1))
        losses_PCA_25 = (faces - faces_PCA_25) ** 2
        MSE_PCA_25, std_PCA_25 = np.mean(losses_PCA_25), np.std(np.mean(losses_PCA_25, axis=1))
        losses_PCA_50 = (faces - faces_PCA_50) ** 2
        MSE_PCA_50, std_PCA_50 = np.mean(losses_PCA_50), np.std(np.mean(losses_PCA_50, axis=1))

        print('Reconstruction loss (10 PCs): mean of MSEs = %f , std of MSEs = % f' % (MSE_PCA_10, std_PCA_10))
        print('Reconstruction loss (25 PCs): mean of MSEs = %f , std of MSEs = % f' % (MSE_PCA_25, std_PCA_25))
        print('Reconstruction loss (50 PCs): mean of MSEs = %f , std of MSEs = % f' % (MSE_PCA_50, std_PCA_50))

        # PART C
        print('PART C')

        ica_10 = FastICA(10, whiten=True, random_state=np.random.seed(7)) 
        ica_25 = FastICA(25, whiten=True, random_state=np.random.seed(7)) 
        ica_50 = FastICA(50, whiten=True, random_state=np.random.seed(7)) 
        ica_10.fit(faces) 
        ica_25.fit(faces) 
        ica_50.fit(faces) 

        fig, axes = plt.subplots(2, 5, figsize=(10, 4),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_10.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f8.png', bbox_inches='tight')

        fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_25.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f9.png', bbox_inches='tight')

        fig, axes = plt.subplots(5, 10, figsize=(12, 6),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(ica_50.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f10.png', bbox_inches='tight')

        # Obtain the reconstructions
        S_10 = ica_10.fit_transform(faces) 
        A_10 = ica_10.mixing_  
        faces_ICA_10 = S_10.dot(A_10.T) + ica_10.mean_
        S_25 = ica_25.fit_transform(faces) 
        A_25 = ica_25.mixing_  
        faces_ICA_25 = S_25.dot(A_25.T) + ica_25.mean_
        S_50 = ica_50.fit_transform(faces) 
        A_50 = ica_50.mixing_  
        faces_ICA_50 = S_50.dot(A_50.T) + ica_50.mean_

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_ICA_10[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f11.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_ICA_25[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f12.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_ICA_50[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f13.png', bbox_inches='tight')

        # Compute the mean and standard deviation of the reconstruction losses
        losses_ICA_10 = (faces - faces_ICA_10) ** 2
        MSE_ICA_10, std_ICA_10 = np.mean(losses_ICA_10), np.std(np.mean(losses_ICA_10, axis=1))
        losses_ICA_25 = (faces - faces_ICA_25) ** 2
        MSE_ICA_25, std_ICA_25 = np.mean(losses_ICA_25), np.std(np.mean(losses_ICA_25, axis=1))
        losses_ICA_50 = (faces - faces_ICA_50) ** 2
        MSE_ICA_50, std_ICA_50 = np.mean(losses_ICA_50), np.std(np.mean(losses_ICA_50, axis=1))
            
        print('Reconstruction loss (10 ICs): mean of MSEs = %f , std of MSEs = % f' % (MSE_ICA_10, std_ICA_10))
        print('Reconstruction loss (25 ICs): mean of MSEs = %f , std of MSEs = % f' % (MSE_ICA_25, std_ICA_25))
        print('Reconstruction loss (50 ICs): mean of MSEs = %f , std of MSEs = % f' % (MSE_ICA_50, std_ICA_50))

        # PART D
        print('PART D')

        nmf_faces = faces + np.abs(np.min(faces))
        nmf_10 = NMF(n_components=10, solver="mu", max_iter=500)
        W_10 = nmf_10.fit_transform(nmf_faces) 
        H_10 = nmf_10.components_
        nmf_25 = NMF(n_components=25, solver="mu", max_iter=500)
        W_25 = nmf_25.fit_transform(nmf_faces) 
        H_25 = nmf_25.components_
        nmf_50 = NMF(n_components=50, solver="mu", max_iter=500)
        W_50 = nmf_50.fit_transform(nmf_faces) 
        H_50 = nmf_50.components_

        fig, axes = plt.subplots(2, 5, figsize=(10, 4),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(nmf_10.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f14.png', bbox_inches='tight')

        fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(nmf_25.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f15.png', bbox_inches='tight')

        fig, axes = plt.subplots(5, 10, figsize=(12, 6),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(nmf_50.components_[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f16.png', bbox_inches='tight')

        # Obtain the reconstructions
        faces_NMF_10 = W_10.dot(H_10) - np.abs(np.min(faces))
        faces_NMF_25 = W_25.dot(H_25) - np.abs(np.min(faces))
        faces_NMF_50 = W_50.dot(H_50) - np.abs(np.min(faces))

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_NMF_10[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f17.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_NMF_25[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f18.png', bbox_inches='tight')

        fig, axes = plt.subplots(6, 6, figsize=(8,8),
                                subplot_kw={'xticks':[], 'yticks':[]},
                                gridspec_kw=dict(hspace=0.01, wspace=0.01))
        for i, ax in enumerate(axes.flat):
            ax.imshow(faces_NMF_50[i].reshape(32, 32).T, cmap=plt.cm.bone)
        #plt.savefig('q1f19.png', bbox_inches='tight')

        # Compute the mean and standard deviation of the reconstruction losses
        losses_NMF_10 = (faces - faces_NMF_10) ** 2
        MSE_NMF_10, std_NMF_10 = np.mean(losses_NMF_10), np.std(np.mean(losses_NMF_10, axis=1))
        losses_NMF_25 = (faces - faces_NMF_25) ** 2
        MSE_NMF_25, std_NMF_25 = np.mean(losses_NMF_25), np.std(np.mean(losses_NMF_25, axis=1))
        losses_NMF_50 = (faces - faces_NMF_50) ** 2
        MSE_NMF_50, std_NMF_50 = np.mean(losses_NMF_50), np.std(np.mean(losses_NMF_50, axis=1))

        print('Reconstruction loss (10 MFs): mean of MSEs = %f , std of MSEs = % f' % (MSE_NMF_10, std_NMF_10))
        print('Reconstruction loss (25 MFs): mean of MSEs = %f , std of MSEs = % f' % (MSE_NMF_25, std_NMF_25))
        print('Reconstruction loss (50 MFs): mean of MSEs = %f , std of MSEs = % f' % (MSE_NMF_50, std_NMF_50))
    
        plt.show()

    elif question == '2':

        with h5py.File('hw4_data2.mat', 'r') as file:
            data_keys = list(file.keys())

        data = dict()
        with h5py.File('hw4_data2.mat', 'r') as file:
            for key in data_keys:
                data[key] = np.array(file[key])
                print('Shape of the data associated with %s:' % key,
                    np.shape(data[key]), '\n')
        
        # QUESTION 2
        print('QUESTION 2')

        stype = data['stype']
        vresp = data['vresp'].T

        # PART A
        print('PART A')

        # Compute similarity matrices using different metrics
        sim_euclidean = cdist(vresp, vresp, metric='euclidean')
        sim_cosine = cdist(vresp, vresp, metric='cosine')
        sim_correlation = cdist(vresp, vresp, metric='correlation')

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(sim_euclidean)
        plt.colorbar()
        plt.title('Similarity of the Response Patterns (Euclidean)')
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(sim_cosine)
        plt.colorbar()
        plt.title('Similarity of the Response Patterns (Cosine)')
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(sim_correlation)
        plt.colorbar()
        plt.title('Similarity of the Response Patterns (Correlation)')
        plt.show(block=False)        

        # PART B
        print('PART B')

        def cmdscale(D):
            """
            Implementation of the classical multidimensional scaling (MDS) algorithm.
            Args:
                D: The symmetric matrix containing the distances between n 
                    objects in p dimensions
            Returns:
                X: Coordinates of n objects in the new space
            """
            N = D.shape[0]
            # Double centering procedure
            J = np.eye(N) - np.ones((N, N)) / N
            B = - J.dot(D ** 2).dot(J) / 2 # = X.X^T
            # Diagonalization
            evals, evecs = np.linalg.eigh(B)
            # Sort eigenpairs according to the descending order of eigenvalues                                               
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            # Extract the positive eigenvalues 
            pos_idx = np.where(evals > 0)[0]
            L = np.diag(np.sqrt(evals[pos_idx]))
            E = evecs[:, pos_idx]
            X = E.dot(L)
            return X

        # Run classical MDS
        MDS_euclidean = cmdscale(sim_euclidean)
        MDS_cosine = cmdscale(sim_cosine)
        MDS_correlation = cmdscale(sim_correlation)

        X_euclidean = MDS_euclidean[:, 0:2]
        X_cosine = MDS_cosine[:, 0:2]
        X_correlation = MDS_correlation[:, 0:2]

        SPEECH_INDICES = np.where(stype == 1)[0]
        NATURE_SOUND_INDICES = np.where(stype == 2)[0]

        # Separate data with different labels
        X_speech_euclidean = X_euclidean[SPEECH_INDICES]
        X_nature_euclidean = X_euclidean[NATURE_SOUND_INDICES]
        X_speech_cosine = X_cosine[SPEECH_INDICES]
        X_nature_cosine = X_cosine[NATURE_SOUND_INDICES]
        X_speech_correlation = X_correlation[SPEECH_INDICES]
        X_nature_correlation = X_correlation[NATURE_SOUND_INDICES]

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_speech_euclidean[:, 0], X_speech_euclidean[:, 1], s=20, marker='o', c='b')
        plt.scatter(X_nature_euclidean[:, 0], X_nature_euclidean[:, 1], s=20, marker='s', c='g')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Euclidean)')
        plt.legend(['speech', 'nature sound'])
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_speech_cosine[:, 0], X_speech_cosine[:, 1], s=20, marker='o', c='b')
        plt.scatter(X_nature_cosine[:, 0], X_nature_cosine[:, 1], s=20, marker='s', c='g')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Cosine)')
        plt.legend(['speech', 'nature sound'])
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_speech_correlation[:, 0], X_speech_correlation[:, 1], s=20, marker='o', c='b')
        plt.scatter(X_nature_correlation[:, 0], X_nature_correlation[:, 1], s=20, marker='s', c='g')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Correlation)')
        plt.legend(['speech', 'nature sound'])
        plt.show(block=False)

        # PART C
        print('PART C')

        def init_clusters(data, k, seed=7):
            """
            Initializes k cluster centers (means).
            Args:
                data: The input data 
                k: Preferred number of clusters
                seed: Random seed for reproducibility (default is 7)
            Returns:
                centers: Array of k cluster centers
            """
            N = data.shape[0] # number of samples
            np.random.seed(seed)
            cluster_idx = np.random.choice(N, k)
            centers = data[cluster_idx]
            return centers

        def build_dist_matrix(data, centers):
            """
            Builds a distance matrix containing the distance
            of each point to each cluster center.
            Args:
                data: The input data
                centers: Array of k cluster centers
            Returns:
                dist_matrix: The distance matrix, entry (i, j)
                    represents the distance of the ith data point
                    to the jth cluster center.
            """
            N = data.shape[0]
            K = centers.shape[0]
            dist_matrix = []
            for k in range(K):
                dist_matrix.append(np.sum((data - centers[k]) ** 2, axis=1))
            return np.array(dist_matrix).T

        def param_update_kmeans(data, old_centers):
            """
            Performs the parameter updates in a kmeans iteration.
            Args:
                data: The input data
                old_centers: Array of k cluster centers before the update
            Returns: 
                losses: Loss computed for each data point
                assignments: Assignment vector indicating which data 
                    point belongs to which cluster
                centers: Array of k cluster centers after the update
            """
            N, F = data.shape
            K = old_centers.shape[1]
            dist_matrix = build_dist_matrix(data, old_centers)
            assignments = np.argmin(dist_matrix, axis=1)
            losses = []
            for i in range(N):
                assignment = assignments[i]
                point_cluster = old_centers[assignment]
                losses.append(np.sum((data[i] - point_cluster) ** 2))
            centers = []
            for k in range(K):
                cluster_idx = np.where(assignments == k)[0]
                centers.append(np.mean(data[cluster_idx], axis=0))
            return np.array(losses), assignments, np.array(centers)    

        def kmeans(data, k, max_iters=100, threshold=1e-5):
            """
            Implementation of the kmeans clustering algorithm.
            Args:
                data: The input data
                k: Preferred number of clusters
                max_iters: Maximum number of iterations to run the algorithm
                    (default is 100)
                threshold: The threshold in the change in kmeans' loss metric, the 
                    algorithm converges when the threshold is reached (default is 1e-5)
            Returns:
                assignments: Assignment vector indicating which data 
                    point belongs to which cluster
                centers: Array of final k cluster centers 
            """
            centers_old = init_clusters(data, k)
            avg_losses = []
            for i in range(max_iters):
                # Perform iteration updates
                losses, assignments, centers = param_update_kmeans(data, centers_old)
                avg_loss = np.mean(losses)
                avg_losses.append(avg_loss)
                # Check for convergence
                if i > 0 and np.abs(avg_losses[-1] - avg_losses[-2]) <= threshold:
                    break
                # Prepare for the next iteration
                centers_old = centers
            return assignments, centers_old 

        # Run kmeans for different metrics
        assignments_euclidean, centers_euclidean = kmeans(X_euclidean, 2)
        assignments_cosine, centers_cosine = kmeans(X_cosine, 2)
        assignments_correlation, centers_correlation = kmeans(X_correlation, 2)

        # Separate the clusters' data
        X_0_euclidean = X_euclidean[np.where(assignments_euclidean == 0)[0]]
        X_1_euclidean = X_euclidean[np.where(assignments_euclidean == 1)[0]]
        X_0_cosine = X_cosine[np.where(assignments_cosine == 0)[0]]
        X_1_cosine = X_cosine[np.where(assignments_cosine == 1)[0]]
        X_0_correlation = X_correlation[np.where(assignments_correlation == 0)[0]]
        X_1_correlation = X_correlation[np.where(assignments_correlation == 1)[0]]

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_0_euclidean[:, 0], X_0_euclidean[:, 1], s=20, marker='^', c='darkorange')
        plt.scatter(X_1_euclidean[:, 0], X_1_euclidean[:, 1], s=20, marker='v', c='darkblue')
        plt.scatter(centers_euclidean[0, 0], centers_euclidean[0, 1], s=250, marker='$C1$', c='r')
        plt.scatter(centers_euclidean[1, 0], centers_euclidean[1, 1], s=250, marker='$C2$', c='r')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Euclidean)')
        plt.legend(['cluster 1', 'cluster 2'])
        plt.show(block=False)

        print('Cluster centers (Euclidean)')
        print('Center of cluster 1: (%f, %f)' % (centers_euclidean[0, 0], centers_euclidean[0, 1]))
        print('Center of cluster 2: (%f, %f)' % (centers_euclidean[1, 0], centers_euclidean[1, 1]))

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_0_cosine[:, 0], X_0_cosine[:, 1], s=20, marker='^', c='darkorange')
        plt.scatter(X_1_cosine[:, 0], X_1_cosine[:, 1], s=20, marker='v', c='darkblue')
        plt.scatter(centers_cosine[0, 0], centers_cosine[0, 1], s=250, marker='$C1$', c='r')
        plt.scatter(centers_cosine[1, 0], centers_cosine[1, 1], s=250, marker='$C2$', c='r')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Cosine)')
        plt.legend(['cluster 1', 'cluster 2'])
        plt.show(block=False)

        print('Cluster centers (Cosine)')
        print('Center of cluster 1: (%f, %f)' % (centers_cosine[0, 0], centers_cosine[0, 1]))
        print('Center of cluster 2: (%f, %f)' % (centers_cosine[1, 0], centers_cosine[1, 1]))

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_0_correlation[:, 0], X_0_correlation[:, 1], s=20, marker='^', c='darkorange')
        plt.scatter(X_1_correlation[:, 0], X_1_correlation[:, 1], s=20, marker='v', c='darkblue')
        plt.scatter(centers_correlation[0, 0], centers_correlation[0, 1], s=250, marker='$C1$', c='r')
        plt.scatter(centers_correlation[1, 0], centers_correlation[1, 1], s=250, marker='$C2$', c='r')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Correlation)')
        plt.legend(['cluster 1', 'cluster 2'])
        plt.show(block=False)

        print('Cluster centers (Correlation)')
        print('Center of cluster 1: (%f, %f)' % (centers_correlation[0, 0], centers_correlation[0, 1]))
        print('Center of cluster 2: (%f, %f)' % (centers_correlation[1, 0], centers_correlation[1, 1]))

        plt.show()

    elif question == '3':

        # QUESTION 3
        print('QUESTION 3')

        # PART A
        print('PART A')

        def tuning_curve(x, A, mu, sigma):
            """
            Gaussian shaped tuning function of a population of neurons.
            Args:
                x: The input stimulus
                A: Amplitude of the Gaussian-shaped tuning curve
                mu: Mean of the Gausssian-shaped tuning curve
                sigma: Standard deviation of the Gaussian-shaped tuning curve
            Returns:
                response: Resulting neural response
            """
            response = A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            return response

        A = 1 # amplitude
        SIGMA = 1 # standard deviation
        NUM_NEURONS = 21
        MU_VALS = np.arange(-10, 11)
        STIMULI = np.linspace(-15, 16, 500)

        activities = []
        for mu in MU_VALS:
            activities.append(tuning_curve(STIMULI, A, mu, SIGMA))

        figure_num = 0
        plt.figure(figure_num)
        figure_num += 1
        for i in range(NUM_NEURONS):
            plt.plot(STIMULI, activities[i])
        plt.xlabel('Stimulus')
        plt.ylabel('Activity')
        plt.title('Tuning Curves of a Population of Neurons')
        plt.show(block=False)          

        X = -1

        responses = tuning_curve(X, A, MU_VALS, SIGMA)

        plt.figure(figure_num)
        figure_num += 1
        plt.xlabel('Preferred Stimulus')
        plt.ylabel('Population Response')
        plt.title('Population Response to the Stimulus x = -1\nvs Preferred Stimuli of Neurons')
        plt.plot(MU_VALS, responses, marker='.', markerfacecolor='red')
        plt.show(block=False)

        # PART B
        print('PART B')

        NUM_TRIALS = 200
        STIMULI_RANGE = np.linspace(-5, 5, 500)

        def winner_take_all_decoder(preferred_stimuli, response):
            """
            Given a population response and preferred stimuli of the
            neurons, estimates the actual stimulus as the preferred
            stimulus of the neuron exhibiting the highest response.
            Args:
                preferred_stimuli: The preferred stimuli of the neurons
                response: The response the population exhibits
            Returns:
                stimulus: the estimated input stimulus
            """
            highest_idx = np.argmax(response)
            stimulus = preferred_stimuli[highest_idx]
            return stimulus

        np.random.seed(17) # for reproducibility
        responses = []
        stimuli = []
        estimated_stimuli_WTA = []
        errors_WTA = []
        for i in range(NUM_TRIALS):
            stimulus = np.random.choice(STIMULI_RANGE)
            response = tuning_curve(stimulus, A, MU_VALS, SIGMA)
            noise = np.random.normal(0, SIGMA / 20, NUM_NEURONS)
            response += noise
            estimated_stimulus_WTA = winner_take_all_decoder(MU_VALS, response)
            error_WTA = np.abs(stimulus - estimated_stimulus_WTA)
            responses.append(response)
            stimuli.append(stimulus)
            estimated_stimuli_WTA.append(estimated_stimulus_WTA)
            errors_WTA.append(error_WTA)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(np.arange(NUM_TRIALS), stimuli, color='r', s=10)
        plt.scatter(np.arange(NUM_TRIALS), estimated_stimuli_WTA, color='deepskyblue', s=10)
        plt.xlabel('Trial Number')
        plt.ylabel('Stimulus')
        plt.title('Actual and Estimated Stimuli Across Trials\n(Winner Take All Decoder)')
        plt.legend(['actual', 'estimated'], loc='upper right')
        plt.show(block=False)

        mean_error_WTA = np.mean(errors_WTA)
        std_error_WTA = np.std(errors_WTA)
        print('Error Statistics for Winner Take All Decoder')
        print('Mean of errors in stimuli estimation (absolute error is used):', mean_error_WTA)
        print('Standard deviation of errors in stimuli estimation (absolute error is used):', std_error_WTA)

        # PART C
        print('PART C')
            
        def nlogLL(x, response, A, mu_vals, sigma):
            """
            Given the input stimulus x, the response elicited by the neuron 
            population and the tuning curve parameters; computes the negative
            of the log-likelihood of seeing the given population response.
            Disregards some constant terms for simplicity.
            Args:
                x: The input stimulus
                response: The response elicited by the neuron population
                A: Amplitude of the tuning curve
                mu_vals: Preferred stimulus value of each neuron in the population
                sigma: Standard deviation of the tuning curve
            Returns:
                nlogLL: The negative log-likelihood to see the given response
            """
            nlogLL = 0
            for r_i, mu_i in zip(response, mu_vals):
                nlogLL += (r_i - tuning_curve(x, A, mu_i, sigma)) ** 2
            return nlogLL

        def MLE_decoder(response, A, mu_vals, sigma, stimuli_range):
            """
            Estimates the input stimulus to a neuron population by maximizing
            the likelihood to see the given population response.
            Args:
                response: The response elicited by the neuron population
                A: Amplitude of the tuning curve
                mu_vals: Preferred stimulus value of each neuron in the population
                sigma: Standard deviation of the tuning curve
                stimuli_range: Range of stimuli to consider
            Returns:
                est_stimulus: The estimated input stimulus
            """
            nlogLL_vals = []
            for stimulus in stimuli_range:
                nlogLL_vals.append(nlogLL(stimulus, response, A, mu_vals, sigma))
            min_idx = np.argmin(nlogLL_vals)
            est_stimulus = stimuli_range[min_idx]
            return est_stimulus            

        estimated_stimuli_MLE = []
        errors_MLE = []
        for response, stimulus in zip(responses, stimuli):
            estimated_stimulus_MLE = MLE_decoder(response, A, MU_VALS, SIGMA, STIMULI_RANGE)
            error_MLE = np.abs(stimulus - estimated_stimulus_MLE)
            estimated_stimuli_MLE.append(float(estimated_stimulus_MLE))
            errors_MLE.append(float(error_MLE))

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(np.arange(NUM_TRIALS), stimuli, color='r', s=10)
        plt.scatter(np.arange(NUM_TRIALS), estimated_stimuli_MLE, color='deepskyblue', s=10)
        plt.xlabel('Trial Number')
        plt.ylabel('Stimulus')
        plt.title('Actual and Estimated Stimuli Across Trials\n(MLE Decoder)')
        plt.legend(['actual', 'estimated'], loc='upper right')
        plt.show(block=False)

        mean_error_MLE = np.mean(errors_MLE)
        std_error_MLE = np.std(errors_MLE)
        print('Error Statistics for MLE Decoder')
        print('Mean of errors in stimuli estimation (absolute error is used):', mean_error_MLE)
        print('Standard deviation of errors in stimuli estimation (absolute error is used):', std_error_MLE)

        # PART D
        print('PART D')

        def nlogPosterior(x, response, A, mu_vals, sigma):
            """
            Given the input stimulus x, the response elicited by the neuron 
            population and the tuning curve parameters; computes the negative
            of the log-posterior probability of seeing the input stimulus 
            given population response. Disregards some constant terms for simplicity.
            Assumes that the prior follows a Gaussian with 0 mean and 2.5 standard
            deviation.
            Args:
                x: The input stimulus
                response: The response elicited by the neuron population
                A: Amplitude of the tuning curve
                mu_vals: Preferred stimulus value of each neuron in the population
                sigma: Standard deviation of the tuning curve
            Returns:
                nlogPosterior: The negative log-posterior to see the input stimulus
            """
            nlogPosterior = 0
            for r_i, mu_i in zip(response, mu_vals):
                nlogPosterior += ((r_i - tuning_curve(x, A, mu_i, sigma)) ** 2) 
            nlogPosterior /= (2 * (sigma / 20) ** 2)
            nlogPosterior += (x ** 2) / (2 * 2.5 ** 2)
            return nlogPosterior

        def MAP_decoder(response, A, mu_vals, sigma, stimuli_range):
            """
            Estimates the input stimulus to a neuron population by maximizing
            the posterior probability to see the input stimulus given the 
            population response.
            Args:
                response: The response elicited by the neuron population
                A: Amplitude of the tuning curve
                mu_vals: Preferred stimulus value of each neuron in the population
                sigma: Standard deviation of the tuning curve
                stimuli_range: Range of stimuli to consider
            Returns:
                est_stimulus: The estimated input stimulus
            """
            nlogPosterior_vals = []
            for stimulus in stimuli_range:
                nlogPosterior_vals.append(nlogPosterior(stimulus, response, A, mu_vals, sigma))
            min_idx = np.argmin(nlogPosterior_vals)
            est_stimulus = stimuli_range[min_idx]
            return est_stimulus

        estimated_stimuli_MAP = []
        errors_MAP = []
        for response, stimulus in zip(responses, stimuli):
            estimated_stimulus_MAP = MAP_decoder(response, A, MU_VALS, SIGMA, STIMULI_RANGE)
            error_MAP = np.abs(stimulus - estimated_stimulus_MAP)
            estimated_stimuli_MAP.append(float(estimated_stimulus_MAP))
            errors_MAP.append(float(error_MAP))

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(np.arange(NUM_TRIALS), stimuli, color='r', s=10)
        plt.scatter(np.arange(NUM_TRIALS), estimated_stimuli_MAP, color='deepskyblue', s=10)
        plt.xlabel('Trial Number')
        plt.ylabel('Stimulus')
        plt.title('Actual and Estimated Stimuli Across Trials\n(MAP Decoder)')
        plt.legend(['actual', 'estimated'], loc='upper right')
        plt.show(block=False)

        mean_error_MAP = np.mean(errors_MAP)
        std_error_MAP = np.std(errors_MAP)
        print('Error Statistics for MAP Decoder')
        print('Mean of errors in stimuli estimation (absolute error is used):', mean_error_MAP)
        print('Standard deviation of errors in stimuli estimation (absolute error is used):', std_error_MAP)        

        # PART E
        print('PART E')

        SIGMA_VALS = [0.1, 0.2, 0.5, 1, 2, 5]

        print('Takes a while...')
        np.random.seed(7) # for reproducibility
        errors_std_MLE = []
        for i in range(NUM_TRIALS):
            stimulus = np.random.choice(STIMULI_RANGE)
            error_std_MLE = []
            for sigma in SIGMA_VALS:
                response_std = tuning_curve(stimulus, A, MU_VALS, sigma)
                noise = np.random.normal(0, 1 / 20, NUM_NEURONS)
                response_std += noise
                estimated_stimulus_std_MLE = MLE_decoder(response_std, A, MU_VALS, sigma, STIMULI_RANGE)
                error_std_MLE.append(np.abs(stimulus - float(estimated_stimulus_std_MLE)))
            errors_std_MLE.append(np.array(error_std_MLE))
        errors_std_MLE = np.array(errors_std_MLE)

        mean_errors_std_MLE = []
        std_errors_std_MLE = []
        for i, sigma in enumerate(SIGMA_VALS):
            mean_error_std_MLE = np.mean(errors_std_MLE[:, i])
            std_error_std_MLE = np.std(errors_std_MLE[:, i])
            print('\nError Statistics for MLE Decoder (sigma = %.1f for the tuning curve)' % sigma)
            print('Mean of errors in stimuli estimation (absolute error is used):', mean_error_std_MLE)
            print('Standard deviation of errors in stimuli estimation (absolute error is used):', std_error_std_MLE)
            mean_errors_std_MLE.append(mean_error_std_MLE)
            std_errors_std_MLE.append(std_error_std_MLE)

        plt.figure(figure_num)
        figure_num += 1
        plt.errorbar(SIGMA_VALS, mean_errors_std_MLE, yerr=std_errors_std_MLE,
                    marker='.', markerfacecolor='r', ecolor='r', elinewidth=1, capsize=3)
        plt.xlabel('Standard Deviation of the Tuning Curve')
        plt.ylabel('Mean Absolute Error')
        plt.title('Mean Absolute Error of MLE Decoder vs Standard Deviation')
        plt.show(block=False)

        plt.show()

    elif question == '4':

        with h5py.File('hw4_data3.mat', 'r') as file:
            data_keys = list(file.keys())

        data_ = dict()
        with h5py.File('hw4_data3.mat', 'r') as file:
            for key in data_keys:
                data_[key] = np.array(file[key])
                print('Shape of the data associated with %s:' % key,
                    np.shape(data_[key]), '\n')

        # QUESTION 2
        print('QUESTION 2')           

        stype = np.array(data_['stype'])
        vresp = np.array(data_['vresp']).T 

        # PART A
        print('PART A')

        # Compute similarity matrix using euclidean distance
        sim_euclidean = cdist(vresp, vresp, metric='euclidean')

        def cmdscale(D):
            """
            Implementation of the classical multidimensional scaling (MDS) algorithm.
            Args:
                D: The symmetric matrix containing the distances between n 
                    objects in p dimensions
            Returns:
                X: Coordinates of n objects in 2 dimensions
            """
            N = D.shape[0]
            # Double centering procedure
            J = np.eye(N) - np.ones((N, N)) / N
            B = - J.dot(D ** 2).dot(J) / 2 # = X.X^T
            # Diagonalization
            evals, evecs = np.linalg.eigh(B)
            # Sort eigenpairs according to the descending order of eigenvalues                                               
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            # Extract the positive eigenvalues 
            pos_idx = np.where(evals > 0)[0]
            L = np.diag(np.sqrt(evals[pos_idx]))
            E = evecs[:, pos_idx]
            X = E.dot(L)
            return X

        # Run classical MDS
        MDS_euclidean = cmdscale(sim_euclidean)

        # Get the projections onto the first two MDS components
        X_euclidean = MDS_euclidean[:, 0:2]    

        FACE_INDICES = np.where(stype == 1)[0]
        BUILDING_INDICES = np.where(stype == 2)[0]

        # Separate data with different labels
        X_face_euclidean = X_euclidean[FACE_INDICES]
        X_building_euclidean = X_euclidean[BUILDING_INDICES]

        def compute_class_means(data, labels, num_classes):
            """
            Computes the mean of each class.
            Args:
                data: The input data
                labels: The given classes of each data point
                num_classes: Number of classes
            Returns:
                means: Means of the classes
            """
            means = []
            for i in range(num_classes):
                class_idx = np.where(labels == i)[0]
                means.append(np.mean(data[class_idx], axis=0))
            return np.array(means)

        def compute_cov(data, labels, class_means):
            """
            Computes an unbiased estimator for the common covariance matrix.
            Args
                data: The input data
                labels: The given classes of each data point
                class_means: Means of the classes
            Returns:
                cov: Unbiased estimator for the common covariance matrix
            """
            K = np.size(class_means) # number of classes
            N, F = data.shape
            cov = np.zeros((F, F))
            for k in range(K):
                for x_i in data[np.where(labels == k)[0]]:
                    cov += x_i.reshape(-1, 1).dot(x_i.reshape(-1, 1).T)
            cov /= (N - K)
            return cov

        def LDA_predictor(x, cov, mean_k, k, labels):
            """
            Computes the class discriminant.
            Args:
                x: The input data point 
                cov: Covariance matrix of the data
                mean_k: Mean of the input class
                k: The input class
                labels: The input data labels
            Returns:
                delta: Class discriminant
            """
            # Compute the prior probability of class k
            pi_k = np.size(labels[labels == k]) /np.size(labels)
            if np.size(x) == 1:
                cov_inv = np.array([[1 / cov]])
            else:
                cov_inv = np.linalg.inv(cov)
            delta = np.log(pi_k) - mean_k.T.dot(cov_inv).dot(mean_k) / 2 
            delta += x.T.dot(cov_inv).dot(mean_k)
            return delta            

        def LDA_binary_boundary(data, labels):
            """
            Computes a LDA-based binary classification boundary.
            Args:
                data: The input data
                labels: The class of each data point
            Returns: 
                x_vals: x coordinate values of the linear boundary
                y_vals: y coordinate values of the linear boundary
            """
            means = compute_class_means(data, labels, 2)
            cov = compute_cov(data, labels, means)
            min_x, max_x = np.min(data[:, 0]), np.max(data[:, 0])
            min_y, max_y = np.min(data[:, 1]), np.max(data[:, 1]) 
            x_vals = np.linspace(min_x, max_x, 500) 
            y_vals = []
            for x in x_vals:
                y_vals.append(bisect(
                    lambda y: LDA_predictor(np.array([x, y]), cov, means[0], 0, labels) - 
                    LDA_predictor(np.array([x, y]), cov, means[1], 1, labels),
                    min_y - np.sign(min_y) * 10 * min_y, max_y + np.sign(max_y) * 10 * max_y))    
            return x_vals, np.array(y_vals)

        def LDA_classifier(x, train, labels, num_classes=2):
            """
            Computes class prediction for an input using LDA-based binary classification.
            Args:
                x: The input data point
                train: The training data
                labels: The class of each training data point
                num_classes: Number of classes in the prediction process (default is 2)
            Returns: 
                class_: Class predicted for input based on binary LDA classifier
            """
            means = compute_class_means(train, labels, num_classes)
            cov = compute_cov(train, labels, means)
            deltas = []
            for k in range(num_classes):
                deltas.append(LDA_predictor(x, cov, means[k], k, labels))
            class_ = np.argmax(np.array(deltas))
            return class_

        x_vals_euclidean, y_vals_euclidean = LDA_binary_boundary(X_euclidean, stype - 1)

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.plot(x_vals_euclidean, y_vals_euclidean, color='r', linestyle='--')
        plt.scatter(X_face_euclidean[:, 0], X_face_euclidean[:, 1], s=20, marker='o', c='b')
        plt.scatter(X_building_euclidean[:, 0], X_building_euclidean[:, 1], s=20, marker='s', c='g')
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Euclidean)')
        plt.legend(['LDA decision\nboundary', 'face', 'building', ])
        plt.show(block=False)

        def CV_leave_one_out(data, labels):
            """
            Performs leave one out cross validation and returns the count
            of correct classifications.
            Args:
                data: The input data
                labels: The original labels of the data points
            Returns:
                correct_count: The number of correct classifications
            """
            correct_count = 0
            for i, test in enumerate(data):
                idx = np.arange(data.shape[0]) != i
                train = data[idx]
                train_labels = labels[idx]
                class_of_test = LDA_classifier(test, train, train_labels)
                correct_count += (class_of_test == labels[i])
            return correct_count

        N = X_euclidean.shape[0]

        correct_count_euclidean = CV_leave_one_out(X_euclidean, stype - 1)
        print('Leave one out cross validation results (euclidean):')
        print('%d class labels predicted correctly out of %d' % (correct_count_euclidean, N))
        print('Prediction accuracy: %.2f%%' % (100 * correct_count_euclidean / N))

        # PART B
        print('PART B')

        # Compute similarity matrix using correlation metric
        sim_correlation = cdist(vresp, vresp, metric='correlation')

        # Run classical MDS
        MDS_correlation = cmdscale(sim_correlation)

        # Get the projections onto the first two MDS components
        X_correlation = MDS_correlation[:, 0:2]
        # Normalize correlation data
        X_correlation[:, 0] = (X_correlation[:, 0] - np.mean(X_correlation[:, 0])) / np.std(X_correlation[:, 0])
        X_correlation[:, 1] = (X_correlation[:, 1] - np.mean(X_correlation[:, 1])) / np.std(X_correlation[:, 1])        

        # Separate data with different labels
        X_face_correlation = X_correlation[FACE_INDICES]
        X_building_correlation = X_correlation[BUILDING_INDICES]

        x_vals_correlation, y_vals_correlation = LDA_binary_boundary(X_correlation, stype - 1)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_face_correlation[:, 0], X_face_correlation[:, 1], s=20, marker='o', c='b')
        plt.scatter(X_building_correlation[:, 0], X_building_correlation[:, 1], s=20, marker='s', c='g')
        plt_axis = plt.axis()
        plt.plot(x_vals_correlation, y_vals_correlation, color='r', linestyle='--')
        plt.axis(plt_axis)
        plt.xlabel('Projections onto the First MDS Component')
        plt.ylabel('Projections onto the Second MDS Component')
        plt.title('MDS Analysis of the Response Patterns (Correlation)')
        plt.legend(['LDA decision\nboundary', 'face', 'building', ])
        plt.show(block=False)

        correct_count_correlation = CV_leave_one_out(X_correlation, stype - 1)
        print('Leave one out cross validation results (correlation):')
        print('%d class labels predicted correctly out of %d' % (correct_count_correlation, N))
        print('Prediction accuracy: %.2f%%' % (100 * correct_count_correlation / N))

        # PART C
        print('PART C')

        # Test LDA classification performance on MDS representations in different dimensions
        accuracies = []
        for i in range(1, 6):
            X_corr = MDS_correlation[:, 0:i]
            correct_count_correlation = CV_leave_one_out(X_corr, stype - 1)
            print('Leave one out cross validation results on %d dimensional MDS representation (correlation):' % i)
            print('%d class labels predicted correctly out of %d' % (correct_count_correlation, N))
            prediction_accuracy = 100 * correct_count_correlation / N
            print('Prediction accuracy: %.2f%%\n' % prediction_accuracy)
            accuracies.append(prediction_accuracy)
        accuracies = np.array(accuracies).flatten()

        plt.figure(figure_num)
        figure_num += 1
        plt.bar(np.arange(1, 6), accuracies, width=0.5, edgecolor='black', color=['r','orange', 'g', 'b', 'y'])
        plt.xlabel('Dimension of MDS representation')
        plt.ylabel('Percentage prediction accuracy')
        plt.title('Percentage Accuracy vs Dimension of MDS representation')
        plt.ylim(90, 100)
        plt.show(block=False)

        plt.show()
        
efe_acer_21602217_hw4(question)