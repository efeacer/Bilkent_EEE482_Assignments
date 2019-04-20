'''
Code for EEE482 homework 3, 2019 Spring. 
Author: Efe Acer
'''

import sys

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt # for plots
import h5py # to be able to use v 7.3 .mat file in the Python 
from scipy.optimize import curve_fit # Python equivalent of lsqcurvefit 
from scipy.stats import spearmanr #  Spearman correlation coefficient calculation
from scipy.stats import norm # for standard normal cdf
from scipy.optimize import fmin # fminsearch equivalent in Python

question = sys.argv[1]

def efe_acer_21602217_hw3(question):
    if question == '1':

        # QUESTION 1
        print('QUESTION 1\n')

        with h5py.File('hw3_data1.mat', 'r') as file:
            data_keys = list(file.keys())

        data = dict()
        with h5py.File('hw3_data1.mat', 'r') as file:
            for key in data_keys:
                data[key] = np.array(file[key]).flatten()
                print('Shape of the data associated with %s:\n' % 
                    key, np.shape(data[key]), '\n')

        # PART A 
        print('PART A\n')                
	    
        y = data['resp2'] # output
        N = np.size(y)
        X_raw = data['resp1'] # regressor
        # append the bias terms to the regressor
        X = X_raw.reshape((N, 1))
        X = np.concatenate((X, np.ones((N, 1))), axis=1)

        def OLS(y, X):
            """
            Given the matrix containing the data labels y, and the matrix
            containing the regressors X; computes the optimal weight vector
            such that the meas squared error (MSE) is minimized.
            Args:
                y: The data labels
                X: The regressors
            Returns 
                w_optimal: The optimal weight vector
            """
            w_optimal = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            return w_optimal

        w_optimal = OLS(y, X)
        print('The optimal weight vector that minimizes the mean squared error '
            'for the linear model is:\n w_optimal = [a, b] =', 
            w_optimal)

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_raw, y, s=15) 
        pred = X.dot(w_optimal)
        plt.plot(X_raw, pred, color='r')
        plt.title('The linear model')
        plt.ylabel('Label (resp2)')
        plt.xlabel('Regressor (resp1)')
        plt.legend(['y = %1.3f x + %1.3f' % (w_optimal[0], w_optimal[1])],
                loc='lower right')
        plt.show(block=False)

        def test_model(y, pred):
            """
            Test a given linearized model by computing the coefficient
            of determination (R^2). Returns the explained variance, 
            unexplained variance and R^2.
            Args:
                y: The data labels
                pred: The predicted valus
            Returns:
                e_var: The explained variance
                u_var: The unexplained variance
                R2: The coefficient of determination
            """
            mean_y = np.mean(y)
            N = np.size(y)
            total_var = np.sum((y - mean_y) ** 2) / (N - 1)
            u_var = np.sum((y - pred) ** 2) / (N - 1)
            temp = total_var - u_var
            R2 = 100 * (temp / total_var)
            return R2 / 100, u_var, R2

        e_var, u_var, R2 = test_model(y, pred)
        print('Explained variance of the linear model:', e_var)
        print('Unexplained variance of the linear model:', u_var)
        print('Coefficient of determination of the linear model:', R2)

        R = np.sqrt(R2)
        pearson = np.corrcoef(y, X_raw)[0, 1]
        print('The value of R:', R)
        print('The value of pearson correlation between '
            'the label and the regressor:', pearson)

        # PART B
        print('PART B\n')

        X_order2 = (X_raw ** 2).reshape((N, 1))
        X_order2 = np.concatenate((X_order2, X), axis=1)

        w_optimal_order2 = OLS(y, X_order2)
        print('The optimal weight vector that minimizes the mean squared error for '
            'the linearized second order model is:\nw_optimal_order2 = [a, b, c] =', 
            w_optimal_order2)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_raw, y, s=15) 
        order = np.argsort(X_raw)
        X_sorted = X_raw[order]
        pred = X_order2.dot(w_optimal_order2)
        plt.plot(X_sorted, pred[order], color='r')
        plt.title('The linearized second order model')
        plt.ylabel('Label (resp2)')
        plt.xlabel('Regressor (resp1)')
        plt.legend(['y = %1.3f x^2 + %1.3f x + %1.3f' % 
                    (w_optimal_order2[0], w_optimal_order2[1],
                    w_optimal_order2[2])], loc='lower right')
        plt.show(block=False)    

        e_var, u_var, R2 = test_model(y, pred)
        print('Explained variance of the linearized second order model:', e_var)
        print('Unexplained variance of the linearized second order model:', u_var)
        print('Coefficient of determination of the linearized second order:', R2)

        R = np.sqrt(R2)
        spearman = spearmanr(y, X_raw)
        print('The value of R:', R)
        print('The value of spearman correlation coefficient between '
            'the labels and the regressor:', spearman.correlation)

        # PART C
        print('PART C\n')

        def parametric_nonlinear_model(x, a, n, b):
            """
            An implementation of the parametric nonlinear model y = a * x^n + b 
            that will be used in the curve_fit 
            function.
            Args:
                x: The input value
                a, n, b: Parameters of the function that will be optimized
            Returns:
                result: The resulting y value
            """
            return a * (x ** n) + b

        w_optimal_parametric, _ = curve_fit(parametric_nonlinear_model,
                                     X_raw, y, p0=[1, 1, 0])
        a_optimal = w_optimal_parametric[0]
        n_optimal = w_optimal_parametric[1]
        b_optimal = w_optimal_parametric[2]
        print('The optimal values in the parametric nonlinear model (starting from (1, 1, 0)):'
            '\n a = %f, n = %f, b = %f' % (a_optimal, n_optimal, b_optimal)) 

        w_optimal_parametric_, _ = curve_fit(parametric_nonlinear_model,
                                     X_raw, y, p0=[10, 7, 100])
        a_optimal_ = w_optimal_parametric[0]
        n_optimal_ = w_optimal_parametric[1]
        b_optimal_ = w_optimal_parametric[2]
        print('The optimal values in the parametric nonlinear model (starting from (10, 7, 100):'
            '\n a = %f, n = %f, b = %f' % (a_optimal_, n_optimal_, b_optimal_)) 

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_raw, y, s=15) 
        pred = parametric_nonlinear_model(X_raw, a_optimal, n_optimal, b_optimal)
        plt.plot(X_sorted, pred[order], color='r')
        plt.title('The parametric nonlinear model')
        plt.ylabel('Label (resp2)')
        plt.xlabel('Regressor (resp1)')
        plt.legend(['y = %1.3f x^(%1.3f) + %1.3f' % 
                    (a_optimal, n_optimal, b_optimal)], loc='lower right')
        plt.show(block=False)

        e_var, u_var, R2 = test_model(y, pred)
        print('Explained variance of the parametric nonlinear model:', e_var)
        print('Unexplained variance of the parametric nonlinear model:', u_var)
        print('Coefficient of determination of the parametric nonlinear model:', R2)

        # PART D
        print('PART D')

        def nearest_neighbor_regression(y, X):
            """
            Performs nonlinear regression based on the nearest neighbor
            approach, meaning that the function predicts the label of the
            closest regressor for each input. 
            Args:
                y: The data labels
                X: The regressors
            Returns:
                pred: The predicted valus
            """
            pred = np.zeros(np.size(y))
            for index, x in enumerate(X):
                index_of_nearest = np.abs(X - x).argmin()
                pred[index] =  y[index_of_nearest]
            return pred

        pred = nearest_neighbor_regression(y, X_raw)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(X_raw, y, s=15) 
        plt.plot(X_sorted, pred[order], color='r')
        plt.title('The nearest neighbor regression based model')
        plt.ylabel('Label (resp2)')
        plt.xlabel('Regressor (resp1)')
        plt.show(block=False)    

        e_var, u_var, R2 = test_model(y, pred)
        print('Explained variance of the nearest neighbor regression based model:', e_var)
        print('Unexplained variance of the nearest neighbor regression based model:', u_var)
        print('Coefficient of determination nearest neighbor regression based model:', R2)

        plt.show()

    elif question == '2':

        # QUESTION 2
        print('QUESTION 2')

        # PART A
        print('PART A')

        def psychometric_function(I, mu, sigma):
            """
            Implementation of the psychometric function.
            Args:
                I: Intensity value
                mu: Mean of the normal distribution used
                sigma: Standard deviation of the normal distribution used.
            Returns:
                p: The resulting probability value
            """
            return 0.5 + norm.cdf(I, loc=mu, scale=sigma) / 2

        I = np.arange(1, 11)
        probs1 = psychometric_function(I, 6, 3)
        probs2 = psychometric_function(I, 3, 4)

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.plot(I, probs1)
        plt.plot(I, probs2)
        plt.title('Psychometric function')
        plt.xlabel('I')
        plt.ylabel('p_c(I)')
        plt.legend(['mu = 6, sigma = 3', 'mu = 3, sigma = 4'])
        plt.grid()
        plt.show(block=False)

        # PART B 
        print('PART B')

        def beurnoulli(p):
            """
            Performs a beurnoulli trial with success probability p.
            Args:
                p: The success probability
            Returns:
                outcome: The outcome of the experiment 
                    (1 means success, 0 means failure)
            """
            rand_val = np.random.rand()
            if rand_val <= p:
                return 1
            return 0

        def binomial(n, p):
            """
            Performs a binomial experiment that involves n independent
            trial where the success probability of each trial is p.
            Args:
                n: The number of trials
                p: The success probability of each trial
            Returns:
                success_count: Number of successes
                trials: An array containing the trial results
            """
            trials = [beurnoulli(p) for _ in range(n)]
            return np.count_nonzero(trials), np.array(trials)

        def simpsych(mu, sigma, I, T):
            """
            Simulates random draws using psychometric probabilities. Each draw itself
            is a binomial experiment.
            Args:
                mu: Normal distribution mean that will be given to psychometric function
                sigma: Normal distribution standard deviation that will be given to 
                    psychometric function
                I: An array containing intensity values
                T: An array containing values for number of trials
            Returns:
                C: An array containing the number of trials correct out of T at each
                    stimulus intensity I
                E: A matrix containing the trial result at each stimulus intensity and 
                    each of T trials at that intensity
            """
            size = np.size(T)
            C = np.zeros(size)
            E = np.zeros((size, int(np.max(T))))
            for i in range(size):
                C[i], E[i] = binomial(int(T[i]), psychometric_function(I[i], mu, sigma))
            return C, E

        mu = 5
        sigma = 1.5
        I = np.arange(1, 8)
        T = np.ones(7) * 100

        np.random.seed(981706)
        C, E = simpsych(mu, sigma, I, T)

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(I, C / T, c='r')
        probs = psychometric_function(I, mu, sigma)
        plt.plot(I, probs)
        plt.title('C/T vs I, and p_c(I)')
        plt.xlabel('I')
        plt.ylabel('C/T, p_c(I)')
        plt.legend(['p_c(I)', 'C/T'])
        plt.grid()
        plt.show(block=False)

        # PART C
        print('PART C')

        def nloglik(pp, I, T, C):
            """
            Computes and return the negative log-likelihood that we observe
            a certain data.
            Args:
                pp: A size 2 array that contains the mean and the standard
                    deviation of the normal distribution that will be given to
                    the psychometric function
                I: An array containing intensity values
                T: An array containing values for number of trials
                C: An array containing the number of trials correct out of T at each
                    stimulus intensity I
            Returns:
                nll: The negative log-likelihood that we observe the argument data
            """
            mu = pp[0]
            sigma = pp[1]
            size = np.size(T)
            ll = 0
            for i in range(size):
                p_i = psychometric_function(I[i], mu, sigma)
                ll += C[i] * np.log(p_i) + (T[i] - C[i]) * np.log(1 - p_i)
            return -ll

        mu_est = np.arange(2, 8.1, 0.1)
        sigma_est = np.arange(0.5, 4.6, 0.1)
        X, Y = np.meshgrid(mu_est, sigma_est)

        size_mu = np.size(mu_est)
        size_sigma = np.size(sigma_est)
        Z = np.zeros((size_sigma, size_mu))
        for i in range(size_sigma):
            for j in range(size_mu):
                    Z[i, j] = nloglik([mu_est[j], sigma_est[i]], I, T, C)


        plt.figure(figure_num)
        figure_num += 1
        plt.contour(X, Y, Z, 50)
        plt.title('Contour plot of negative log likelihood of the dataset')
        plt.xlabel('mu_est')
        plt.ylabel('sigma_est')
        plt.colorbar()
        plt.show(block=False)    

        # PART D
        print('PART D')

        mu_best, sigma_best = fmin(func=lambda pp: nloglik(pp, I, T, C), x0=[2, 2])
        print('Best mu_est:', mu_best)
        print('Best sigma_est:', sigma_best)

        # PART E
        print('PART E')

        def resample_matrix(matrix):
            """
            Resamples a matrix using bootstrapping on each row.
            Args:
                matrix: The matrix that will be resampled
            Returns:
                resample_matrix: The resampled matrix
            """
            num_rows, num_cols = np.shape(matrix)
            resampled_matrix = np.zeros((num_rows, num_cols))
            indices = np.arange(num_cols)
            for i in range(num_rows):
                bootstrap_indices = np.random.choice(indices, num_cols)
                resampled_matrix[i] = matrix[i][bootstrap_indices]
            return resampled_matrix

        NUM_RESAMPLES = 200

        mu_resamples = []
        sigma_resamples  = []
        np.random.seed(7) # to be able to reproduce results
        for i in range(NUM_RESAMPLES):
            E_resample = resample_matrix(E)
            C_resample = np.sum(E_resample, axis=1)
            mu_resample, sigma_resample = fmin(
                func=lambda pp: nloglik(pp, I, T, C_resample), x0=[2, 2], disp=False)
            mu_resamples.append(mu_resample)
            sigma_resamples.append(sigma_resample)


        def compute_confidence_interval(data, confidence):
            """
            Given the data and the confidence level, computes the confidence interval
            of the data samples.
            Args:
                data: The given data
                confidence: The confidence level, known as alpha (between 0 and 100)
            Returns:
                lower: The lowerbound of the confidence interval
                upper: The upperbound of the confidence interval
            """
            sorted_data = np.sort(data)
            lower = np.percentile(sorted_data, (100 - confidence) / 2)
            upper = np.percentile(sorted_data, confidence + (100 - confidence) / 2)
            return lower, upper

        plt.figure(figure_num)
        figure_num += 1
        plt.hist(mu_resamples, bins=20, color='c', edgecolor='k')
        plt.title('Histogram of mu_est resamples')
        plt.xlabel('mu_est')
        plt.ylabel('Count')
        plt.show(block=False)

        mu_lower, mu_upper = compute_confidence_interval(mu_resamples, 95)
        print('95%% Confidence interval for mu_est: (%1.5f, %1.5f)' % (mu_lower, mu_upper))

        plt.figure(figure_num)
        figure_num += 1
        plt.hist(sigma_resamples, bins=20, color='m', edgecolor='k')
        plt.title('Histogram of sigma_est resamples')
        plt.xlabel('sigma_est')
        plt.ylabel('Count')
        plt.show(block=False)

        sigma_lower, sigma_upper = compute_confidence_interval(sigma_resamples, 95)
        print('95%% Confidence interval for sigma_est: (%1.5f, %1.5f)' % (sigma_lower, sigma_upper))

        plt.show()

    elif question == '3':

        # QUESTION 3
        print('QUESTION 3\n')

        with h5py.File('hw3_data2.mat', 'r') as file:
            data_keys = list(file.keys())

        data = dict()
        with h5py.File('hw3_data2.mat', 'r') as file:
            for key in data_keys:
                data[key] = np.array(file[key])

        Yn = data['Yn'].flatten()
        Xn = data['Xn'].T
        print('Shape of Yn:', np.shape(Yn))
        print('Shape of Xn:', np.shape(Xn))

        # PART A
        print('PART A')

        def ridge(y, X, lambda_):
            """
            Given data labels and regressors, learns an optimal weight
            vector according to the ridge regression formulation.
            Args:
                y: The data labels
                X: The regressors
                lambda_: The regularization parameter
            Returns 
                w_optimal: The optimal weight vector
            """
            K = np.shape(X)[1]
            temp = np.linalg.inv(X.T.dot(X) + lambda_ * np.eye(K))
            w_optimal = temp.dot(X.T).dot(y)
            return w_optimal

        def compute_R2(Y, pred):
            """
            Tests a given linearized model by computing the coefficient
            of determination (R^2). R^2 is computed as the square of the 
            Pearson correlation between the labels and the predictions.
            Args:
                Y: The data labels
                pred: The predicted valus
            Returns:
                R2: The coefficient of determination
            """
            pearson = np.corrcoef(Y, pred)[0, 1]
            R2 = pearson ** 2
            return R2

        def cross_validation(y, X, k_fold, lambda_arr):
            """
            Performs k fold cross validation with three way split in each
            iteration. The aim is to tune the ridge regression's regularizer,
            lambda. Hence each value in an array of lambda values is integrated
            into the model and coefficient of determination (R^2) is calculated
            for each case. 
            Args:
                y: The data labels
                X: The regressors
                k_fold: Number of folds in cross validation
                lambda_arr: The regularization parameters to select from
            Returns:
                dict_valid: The R^2 values calculated in the validation
                    set for each fold and lambda value, stored as a dictionary.
                dict_test: The R^2 values calculated in the test set 
                    for each fold and lambda value, stored as a dictionary.
            """
            N = np.size(y)
            idx_unit = int(N / k_fold) 
            dict_valid = dict()
            dict_test = dict()
            for i in range(k_fold):
                valid_start = i * idx_unit
                test_start = (i + 1) * idx_unit
                train_start = (i + 2) * idx_unit
                valid_indices = np.arange(valid_start, test_start) % N
                test_indices = np.arange(test_start, train_start) % N
                train_indices = np.arange(train_start, N + valid_start) % N
                y_valid = y[valid_indices]
                X_valid = X[valid_indices]
                y_test = y[test_indices]
                X_test = X[test_indices]
                y_train = y[train_indices]
                X_train = X[train_indices]
                for lambda_ in lambda_arr:
                    w = ridge(y_train, X_train, lambda_)
                    dict_valid.setdefault(lambda_, []).append(compute_R2(y_valid, X_valid.dot(w)))
                    dict_test.setdefault(lambda_, []).append(compute_R2(y_test, X_test.dot(w)))
            dict_valid = dict((k, np.mean(v)) for k, v in dict_valid.items())
            dict_test = dict((k, np.mean(v)) for k, v in dict_test.items())
            return dict_valid, dict_test

        K_FOLD = 10

        lambda_arr = np.logspace(0, 12, num=500, base=10)
        # Takes about a minute to execute 
        dict_valid, dict_test = cross_validation(Yn, Xn, K_FOLD, lambda_arr)    

        lambda_optimal = max(dict_valid, key=lambda k: dict_valid[k])
        print('Optimal lambda:', lambda_optimal,
            '\nCorresponding R^2 in validation set:', dict_valid[lambda_optimal], 
            '\nCorresponding R^2 in test set:', dict_test[lambda_optimal])

        figure_num = 1
        lists1 = sorted(dict_valid.items()) # list of tuples sorted by key
        x1, y1 = zip(*lists1) # unpack a list of pairs into two tuples
        lists2 = sorted(dict_test.items()) 
        x2, y2 = zip(*lists2) 
        plt.figure(figure_num)
        figure_num += 1
        plt.plot(x2, y2, color='r')
        plt.plot(x1, y1, color='b')
        plt.legend(['test set', 'validation set',])
        plt.ylabel('R^2')
        plt.xlabel('lambda')
        plt.title('R^2 vs lambda')
        plt.xscale('log')
        plt.grid()
        plt.show(block=False)

        # PART B
        print('PART B')

        NUM_ITER = 500 # number of bootstrap iterations
        N = np.size(Yn) # number of samples 

        np.random.seed(7) # to be able to reproduce the results
        w_bootstrap_OLS = []
        for _ in range(NUM_ITER):
            # draw N samples with replacement 
            indices = np.arange(N)
            bootstrap_indices = np.random.choice(indices, N)
            # genereate the respective bootstrap labels and regressors
            y_bootstrap = Yn[bootstrap_indices]
            X_bootstrap = Xn[bootstrap_indices]
            w_OLS = ridge(y_bootstrap, X_bootstrap, 0)
            w_bootstrap_OLS.append(w_OLS)
        w_bootstrap_OLS = np.array(w_bootstrap_OLS).T # now rows indicate regressors

        x_vals = np.arange(1, 101)
        w_mean_OLS = np.mean(w_bootstrap_OLS, axis=1)
        w_std_OLS = np.std(w_bootstrap_OLS, axis=1)

        plt.figure(figure_num)
        figure_num += 1
        plt.errorbar(x_vals, w_mean_OLS, yerr=2 * w_std_OLS, ecolor='r',
                    elinewidth=0.5, capsize=2)
        plt.title('Model weights, w (OLS)')
        plt.xlabel('i')
        plt.ylabel('value of w_i')
        plt.show(block=False)

        z_vals = w_mean_OLS / w_std_OLS
        p_vals = 2 * (1 - norm.cdf(np.abs(z_vals)))
        significant_OLS = np.argwhere(p_vals < 0.05).flatten()
        print('Indices of the parameters that are significantly different than 0:\n', 
            significant_OLS)

        # PART C
        print('PART C')

        np.random.seed(7) # to be able to reproduce the results
        w_bootstrap_ridge = []
        for _ in range(NUM_ITER):
            # draw N samples with replacement 
            indices = np.arange(N)
            bootstrap_indices = np.random.choice(indices, N)
            # genereate the respective bootstrap labels and regressors
            y_bootstrap = Yn[bootstrap_indices]
            X_bootstrap = Xn[bootstrap_indices]
            w_ridge = ridge(y_bootstrap, X_bootstrap, lambda_optimal)
            w_bootstrap_ridge.append(w_ridge)
        w_bootstrap_ridge = np.array(w_bootstrap_ridge).T # now rows indicate regressors

        w_mean_ridge = np.mean(w_bootstrap_ridge, axis=1)
        w_std_ridge = np.std(w_bootstrap_ridge, axis=1)

        plt.figure(figure_num)
        figure_num += 1
        plt.errorbar(x_vals, w_mean_ridge, yerr=2 * w_std_ridge, 
                    ecolor='r', elinewidth=0.5, capsize=2)
        plt.title('Model weights, w (Ridge)')
        plt.xlabel('i')
        plt.ylabel('value of w_i')
        plt.show(block=False)

        z_vals = w_mean_ridge / w_std_ridge
        p_vals = 2 * (1 - norm.cdf(np.abs(z_vals)))
        significant_ridge = np.argwhere(p_vals < 0.05).flatten()
        print('Indices of the parameters that are significantly different than 0:\n', 
            significant_ridge)

        plt.show()

    elif question == '4':

        # QUESTION 4
        print('QUESTION 4\n')

        with h5py.File('hw3_data3.mat', 'r') as file:
            data_keys = list(file.keys())

        data = dict()
        with h5py.File('hw3_data3.mat', 'r') as file:
            for key in data_keys:
                data[key] = np.array(file[key]).flatten()
                print('Shape of the data associated with %s:\n' % 
                    key, np.shape(data[key]), '\n')

        # PART A 
        print('PART A\n')

        pop1 = data['pop1']
        pop2 = data['pop2']

        NUM_SAMPLES = 10000

        def bootstrap(arr, num_samples, seed=7):
            """
            Resamples an array using the bootstrap technique.
            Args:
                arr: The array that will be resampled
                num_samples: Number of samples that will be generated
                seed: The random seed to be able to reproduce the results
            Returns:
                arr_bootstrap: Numpy array containing the new samples
            """
            arr_bootstrap = []
            arr_size = np.size(arr)
            indices = np.arange(arr_size)
            np.random.seed(seed) # to be able to reproduce the results
            for _ in range(num_samples):
                bootstrap_indices = np.random.choice(indices, arr_size)
                arr_resample = arr[bootstrap_indices]
                arr_bootstrap.append(arr_resample)
            return np.array(arr_bootstrap)

        def difference_in_means(arr1, arr2, num_samples, bins=60, seed=7):
            """
            Generates a sampling distribution of the difference
            in means of two individual distributions. Uses bootstrapping
            to generate samples from the combined distribution.
            Args:
                arr1: The first distribution
                arr2: The second distribution
                num_samples: Number of samples to generate when computing
                    the sampling distribution
                bins: Number of bins in the discretized distribution
                seed: The random seed to be able to reproduce the results
            Returns:
                diff_in_means: The sampling distribution of the difference
                    in means of the given distributions
                vals: The discretized interval of values
                probs: The probabilities that the discretized interval of values
                    can be seen
            """
            arr = np.concatenate((arr1, arr2))
            arr_bootstrap = bootstrap(arr, num_samples, seed)
            samples1 = arr_bootstrap[:, :np.size(arr1)]
            samples2 = arr_bootstrap[:, np.size(arr1):]
            means1 = np.mean(samples1, axis=1)
            means2 = np.mean(samples2, axis=1)
            diff_in_means = means1 - means2
            probs, vals = np.histogram(diff_in_means, bins=bins, density=True)
            return diff_in_means, vals, probs    

        diff_in_means, vals, probs = difference_in_means(pop1, pop2, NUM_SAMPLES)

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.title('Sampling Distribution of\nDifference in means of pop1 and pop2')
        plt.xlabel('Difference in means (x)')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(diff_in_means, bins=60, density=True)
        plt.show(block=False)

        x_overline = np.mean(pop1) - np.mean(pop2)
        sigma_0 = np.std(diff_in_means)
        mu_0 = np.mean(diff_in_means)
        z = (x_overline - mu_0) / sigma_0
        p = 2 * (1 - norm.cdf(z))
        print('The z-value is:', z)
        print('The two sided p-value is:', p)

        # PART B 
        print('PART B\n')

        vox1 = data['vox1']
        vox2 = data['vox2']

        vox1_bootstrap = bootstrap(vox1, NUM_SAMPLES)
        vox2_bootstrap = bootstrap(vox2, NUM_SAMPLES)

        corr_bootstrap = np.zeros(NUM_SAMPLES)
        for i in range(NUM_SAMPLES):
            corr_bootstrap[i] = np.corrcoef(vox1_bootstrap[i], 
                                            vox2_bootstrap[i])[0, 1]

        # Function to compute the confidence interval of data samples
        def compute_confidence_interval(data, confidence):
            """
            Given the data and the confidence level, computes the confidence interval
            of the data samples.
            Args:
                data: The given data
                confidence: The confidence level, known as alpha (between 0 and 100)
            Returns:
                lower: The lowerbound of the confidence interval
                upper: The upperbound of the confidence interval
            """
            sorted_data = np.sort(data)
            lower = np.percentile(sorted_data, (100 - confidence) / 2)
            upper = np.percentile(sorted_data, confidence + (100 - confidence) / 2)
            return lower, upper

        corr_mean = np.mean(corr_bootstrap)
        corr_lower, corr_upper = compute_confidence_interval(corr_bootstrap, 95)
        print('Mean correlation value:', corr_mean)
        print('95%% confidence interval of the correlation values: (%1.5f, %1.5f)' %
            (corr_lower, corr_upper))

        corr_zero_percentage = 100 * np.size(np.where(np.isclose(corr_bootstrap, 0))) / NUM_SAMPLES
        print('Percentage of zero correlation values:', corr_zero_percentage)

        # PART C
        print('PART C')

        vox1_independent = bootstrap(vox1, NUM_SAMPLES, seed=17)
        vox2_independent  = bootstrap(vox2, NUM_SAMPLES, seed=51)

        y = np.zeros(NUM_SAMPLES)
        for i in range(NUM_SAMPLES):
            y[i] = np.corrcoef(vox1_independent [i], vox2_independent [i])[0, 1]

        plt.figure(figure_num)
        figure_num += 1
        plt.title('Sampling Distribution of\nCorrelation between vox1 and vox2')
        plt.xlabel('Correlation (y)')
        plt.ylabel('P(y)')
        plt.yticks([])
        plt.hist(y, bins=60, density=True)
        plt.show(block=False)

        y_overline = np.corrcoef(vox1, vox2)[0, 1]
        sigma_0 = np.std(y)
        mu_0 = np.mean(y)
        z = (y_overline - mu_0) / sigma_0
        p = 1 - norm.cdf(z)
        print('The z-value is:', z)
        print('The one sided p-value is:', p)

        # PART D
        print('PART D')

        building = data['building']
        face = data['face']

        diff_in_means = []
        np.random.seed(7)
        for i in range(NUM_SAMPLES):
            resample = []
            for j in range(np.size(face)):
                indices = np.random.choice(np.size(face))
                options = [0, 0]
                options.append(building[j] - face[j])
                options.append(face[j] - building[j])
                resample.append(np.random.choice(options))
            diff_in_means.append(np.mean(resample))
        diff_in_means = np.array(diff_in_means)

        plt.figure(figure_num)
        figure_num += 1
        plt.title('Sampling Distribution of\nDifference in means of building and face\n'
                '(subject populations are assumed to be same)')
        plt.xlabel('Difference in means (x)')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(diff_in_means, bins=60, density=True)
        plt.show(block=False)

        x_overline = np.mean(building) - np.mean(face)
        sigma_0 = np.std(diff_in_means)
        mu_0 = np.mean(diff_in_means)
        z = (x_overline - mu_0) / sigma_0
        p = 2 * (1 - norm.cdf(np.abs(z)))
        print('The z-value is:', z)
        print('The two sided p-value is:', p)

        # PART E
        print('PART E')

        diff_in_means, vals, probs = difference_in_means(building, face, NUM_SAMPLES)

        plt.figure(figure_num)
        figure_num += 1
        plt.title('Sampling Distribution of\nDifference in means of building and face\n'
                '(subject populations are assumed to be same)')
        plt.xlabel('Difference in means (x)')
        plt.ylabel('P(x)')
        plt.yticks([])
        plt.hist(diff_in_means, bins=60, density=True)
        plt.show(block=False)

        x_overline = np.mean(building) - np.mean(face)
        sigma_0 = np.std(diff_in_means)
        mu_0 = np.mean(diff_in_means)
        z = (x_overline - mu_0) / sigma_0
        p = 2 * (1 - norm.cdf(np.abs(z)))
        print('The z-value is:', z)
        print('The two sided p-value is:', p)

        plt.show()

efe_acer_21602217_hw3(question)