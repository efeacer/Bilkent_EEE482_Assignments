'''
Code for EEE482 homework 2, 2019 Spring. 
Author: Efe Acer
'''

import sys

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt # for plots
from scipy.io import loadmat # to be able to use .mat file in the python environment
from PIL import Image # to read .bmp image
from mpl_toolkits import mplot3d # for 3D plots

question = sys.argv[1]

def efe_acer_21602217_hw2(question):

    if question == '1' :
        
        # QUESTION 1
        print('QUESTION 1\n')

        # PART B
        print('PART B')

        # Constants
        TAU_M = 10e-3 # 10 milli seconds
        R = 1e3 # 1 kilo ohm
        I_0 = 2e-3 # 2 milli ampere
        TIME_INTERVAL = (0, 100e-3) # 0 to 100 milli seconds
        NUM_STEPS = 10000 # number of bins in the time interval
        H = (TIME_INTERVAL[1] - TIME_INTERVAL[0]) / NUM_STEPS # step_size

        v_numerical = np.zeros(NUM_STEPS) # in volts
        t = np.zeros(NUM_STEPS) # in seconds

        # Computing values in the first order difference equation
        for i in range(0, NUM_STEPS - 1):
            v_numerical[i + 1] = ((TAU_M - H) * v_numerical[i] + H * R * I_0) / TAU_M
            t[i + 1] = (i + 1) * H

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.plot(t, v_numerical)
        plt.title('Approximation of the membrane potential (v(t))'
                '\n using Euler\'s method (step size (h) = 10 micro seconds)')
        plt.ylabel('Membrane potential (V)')
        plt.xlabel('Time (s)')
        plt.show(block=False)
        
        def analytical_solution(t):
            """
            Given a time value, computes and returns the value of 
            membrane potential calculated using the analylitical 
            solution of the model involving the differential equation.
            (Uses globally defined constants)
            Args:
                t: The specified time value
            Returns:
                v(t): The membrane potential calculated according
                    to the analytical solution.
            """
            return R * I_0 * (1 - np.exp(-t / TAU_M))

        v_analytical = analytical_solution(t)

        plt.figure(figure_num)
        figure_num += 1
        plt.plot(t, v_analytical, color='green')
        plt.title('Values of the membrane potential (v(t))'
                '\n calculated according to the analytical solution')
        plt.ylabel('Membrane potential (V)')
        plt.xlabel('Time (s)')
        plt.show(block=False)

        MSE = np.sum((v_analytical - v_numerical) ** 2) / NUM_STEPS
        print('Mean Squared Error (MSE) of the numerical approximation: %.2g' % MSE)

        # PART C
        print('PART C')

        # Constants
        THETA = 1 # a threshold value of 1 V
        V_RESET = 0 # the reset voltage value

        v_threshold_numerical = np.zeros(NUM_STEPS) # in volts

        # Computing values in the first order difference equation
        for i in range(0, NUM_STEPS - 1):
            voltage = ((TAU_M - H) * v_threshold_numerical[i] + H * R * I_0) / TAU_M
            v_threshold_numerical[i + 1] = V_RESET if voltage >= THETA else voltage

        plt.figure(figure_num)
        figure_num += 1
        plt.plot(t, v_threshold_numerical)
        plt.title('Approximation of the membrane potential (v(t))'
                '\nwith threshold (theta = 1 volt)'
                '\n using Euler\'s method (step size (h) = 10 micro seconds)')
        plt.ylabel('Membrane potential (V)')
        plt.xlabel('Time (s)')
        plt.show(block=False)

        v_threshold_analytical = np.zeros(NUM_STEPS) # in volts
        t_threshold = np.array(t)

        for i in range(NUM_STEPS):
            voltage = analytical_solution(t_threshold[i])
            if voltage >= THETA:
                t_threshold[i:] -= t_threshold[i]
                voltage = analytical_solution(t_threshold[i])
            v_threshold_analytical[i] = voltage

        plt.figure(figure_num)
        figure_num += 1
        plt.plot(t, v_threshold_analytical, color='green')
        plt.title('Values of the membrane potential (v(t)) with threshold (theta'
                ' = 1 volt)\n calculated according to the analytical solution')
        plt.ylabel('Membrane potential (V)')
        plt.xlabel('Time (s)')
        plt.show(block=False)

        MSE = np.sum((v_threshold_analytical - v_threshold_numerical) ** 2) / NUM_STEPS
        print('Mean Squared Error (MSE) of the numerical approximation ' 
            '(threshold case): %.2g' % MSE)

        # PART D
        print('PART D')

        I_VALUES = np.arange(2e-3, 10e-3 + 1e-5, 1e-5) # current values in milli amperes

        interspike_intervals = np.zeros(np.size(I_VALUES))

        for i, I in enumerate(I_VALUES):
            v = np.zeros(NUM_STEPS)
            for j in range(0, NUM_STEPS - 1):
                v[j + 1] = ((TAU_M - H) * v[j] + H * R * I) / TAU_M
                if v[j + 1] >= THETA:
                    interspike_intervals[i] = (j + 1) * H
                    break

        firing_rates = 1 / interspike_intervals

        plt.figure(figure_num)
        figure_num += 1
        plt.plot(I_VALUES, firing_rates)
        plt.title('Firing rate vs input current')
        plt.xlabel('Input DC current (A)')
        plt.ylabel('Firing rate (Hz)')
        plt.show(block=False)

        # PART E
        print('PART E')

        # Constants
        GAUSSIAN_MEAN = 0 
        GAUSSIAN_STD = 4e-3 # 4 milli amperes
        NUM_STEPS = 2000
        H = (TIME_INTERVAL[1] - TIME_INTERVAL[0]) / NUM_STEPS

        v_noisy = np.zeros(NUM_STEPS) # in volts
        t_noisy = np.zeros(NUM_STEPS)
        np.random.seed(7) # set the seed to be able to recover the results

        # Computing values in the first order difference equation
        for i in range(0, NUM_STEPS - 1):
            noise =  GAUSSIAN_STD * np.random.randn() + GAUSSIAN_MEAN
            voltage = ((TAU_M - H) * v_noisy[i] + H * R * (I_0 + noise)) / TAU_M
            v_noisy[i + 1] = V_RESET if voltage >= THETA else voltage
            t_noisy[i + 1] = (i + 1) * H 

        plt.figure(figure_num)
        figure_num += 1
        plt.plot(t_noisy, v_noisy)
        plt.title('Approximation of the membrane potential (v(t))'
                '\nwith threshold (theta = 1 volt) and noise'
                '\n using Euler\'s method (step size (h) = 50 micro seconds)')
        plt.ylabel('Membrane potential (V)')
        plt.xlabel('Time (s)')
        plt.show(block=False)

        last_pair = (0, 0)
        interspike_intervals = np.zeros(np.size(I_VALUES))

        for i, I in enumerate(I_VALUES):
            intervals = [] # holds the individual interspike intervals 
            for j in range(0, NUM_STEPS - 1):
                noise = GAUSSIAN_STD * np.random.rand() + GAUSSIAN_MEAN
                v_noisy[j + 1] = ((TAU_M - H) * v_noisy[j] + H * R * (I + noise)) / TAU_M
                if v_noisy[j + 1] >= THETA:
                    v_noisy[j + 1] = V_RESET
                    interval = (j + 1) * H
                    # subtract the end point of the previous time interval if needed
                    if last_pair[0] == i: 
                        interval -= last_pair[1]
                    last_pair = (i, ((j + 1) * H))
                    intervals.append(interval)
            interspike_intervals[i] = np.mean(interval) # average interspike interval 
            
        firing_rates_noisy = 1 / interspike_intervals

        plt.figure(figure_num)
        figure_num += 1
        plt.scatter(I_VALUES, firing_rates_noisy, 
                    s=2, color='m', label='average firing rates')
        plt.plot(I_VALUES, np.poly1d(np.polyfit(I_VALUES, firing_rates_noisy, 1))
                (I_VALUES), color='y', linewidth=2, label='best fit line')
        plt.legend()
        plt.title('Firing rate vs noisy input current')
        plt.xlabel('Input noisy DC current (A)')
        plt.ylabel('Firing rate (Hz)')
        _, _, y1, y2 = plt.axis()
        plt.axis((2e-3, 10e-3, y1, y2))
        plt.show(block=False)

        plt.show()

    elif question == '2' :
        
        # QUESTION 2
        print('QUESTION 2\n')

        data = loadmat('c2p3.mat')
        counts = data['counts'].flatten()
        print('Dimension of counts:', np.shape(counts))
        stim = data['stim']
        print('Dimension of stim:', np.shape(stim), '\n')

        # PART A 
        print('PART A')

        NUM_STEPS = 10

        def STA(counts, stim, num_steps):
            """
            Given spike and stimulus data, performs Spike Triggered 
            Averaging (STA). STA is performed by computing a weighted
            averaging "num_steps" individual intervals of the stimulus.
            Returns the averages after the computation.
            Args:
                counts: The spike counts
                stim: The stimulus data
                num_steps: Number of individual intervals, in which STA 
                    will be computed
            Returns:
                averages: The resulting averages.
            """
            averages = np.zeros((np.shape(stim)[0], np.shape(stim)[1], num_steps))
            for i in range(np.size(counts)):
                for j in range(num_steps):
                    if i > j:
                        averages[:,:, j] += stim[:,:, i - 1 - j] * counts[i]
            averages /= np.sum(counts)
            return averages

        STAs = STA(counts, stim, NUM_STEPS)

        # Find the range of pixels that cover all STAs
        min_pixel = np.min(STAs)
        max_pixel = np.max(STAs)
        figure_num = 1
        for i in range(NUM_STEPS):
            plt.figure(figure_num)
            figure_num += 1
            plt.imshow(STAs[:,:, i], cmap='gray', vmin=min_pixel, vmax=max_pixel)
            step_or_steps = 'steps' if i != 0 else 'step'
            plt.title('STA %d %s before a spike' % ((i + 1), step_or_steps))
            plt.show(block=False)

        # PART B 
        print('PART B')

        row_summed_avgs = np.sum(STAs, axis=0) # sum across rows 
        col_summed_avgs = np.sum(STAs, axis=1) # sum across columns 

        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(row_summed_avgs, cmap='gray')
        plt.title('Row summed STAs')
        plt.ylabel('column pixel')
        plt.xlabel('time step')
        x_left, x_right = plt.xlim()
        plt.xticks(np.arange(x_left, x_right, 1), np.arange(1, 11))
        plt.show(block=False)
        
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(col_summed_avgs, cmap='gray')
        plt.title('Column summed STAs')
        plt.ylabel('row pixel')
        plt.xlabel('time step')
        plt.xticks(np.arange(x_left, x_right, 1), np.arange(1, 11))
        plt.show(block=False)

        # PART C 
        print('PART C')

        stim_projections = np.zeros(np.size(counts))
        for i in range(np.size(counts)):
            stim_projections[i] = np.sum(STAs[:,:, 0] * stim[:,:, i])
        stim_projections /= np.max(stim_projections)

        plt.figure(figure_num)
        figure_num += 1
        plt.hist(stim_projections, bins=100)
        plt.title('Histogram of normalized stimulus projections')
        plt.ylabel('count')
        plt.xlabel('normalized stimulus projection')
        plt.show(block=False)

        nonzero_stim_indices = np.where(counts != 0)[0]
        nonzero_stim_projs = np.zeros(np.size(nonzero_stim_indices))
        for i in range(np.size(nonzero_stim_projs)):
            nz = nonzero_stim_indices[i]
            if nz >= 1:
                nonzero_stim_projs[i] = np.sum(STAs[:,:, 0] * stim[:,:, nz - 1])
        nonzero_stim_projs /= np.max(nonzero_stim_projs)

        plt.figure(figure_num)
        figure_num += 1
        plt.hist(nonzero_stim_projs, bins=100)
        plt.title('Histogram of normalized stimulus projections for nonzero spikes')
        plt.ylabel('count')
        plt.xlabel('normalized stimulus projection for nonzero spike')
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        plt.hist([stim_projections, nonzero_stim_projs],
                color=['r', 'b'], alpha=0.8, label=['case 1', 'case2'], bins=100)
        plt.legend()
        plt.title('Comparison of the histograms')
        plt.ylabel('count')
        plt.xlabel('normalized stimulus projection for nonzero spike')
        plt.show(block=False)

        plt.show()

    elif question == '3':
        
        #'Implemented in Matlab and mailed separately.'
        print('Implemented in Matlab and mailed separately.')

    elif question == '4':

        # QUESTION 4
        print('QUESTION 4\n')

        # PART A
        print('PART A')
        
        def DOG(x, y, sigma_c, sigma_s):
            """
            Implements an on-center difference-of-gaussians (DOG) center-surround 
            filter centered at 0. 
            Args:
                x: x coordinate
                y: y coordinate
                sigma_c: Central gaussian width
                sigma_s: Surround gaussian width
            Returns: 
                result: The output of the DOG filter, D(x, y)
            """
            const_c = 1 / (2 * sigma_c ** 2)
            const_s = 1 / (2 * sigma_s ** 2)
            exp_c = np.exp(-(x ** 2 + y ** 2) * const_c)
            exp_s = np.exp(-(x ** 2 + y ** 2) * const_s)
            result = (const_c / np.pi) * exp_c - (const_s / np.pi) * exp_s
            return result

        SIGMA_C = 2
        SIGMA_S = 4
        SHAPE = (21, 21)

        def DOG_receptive_field(x, y, shape, sigma_c, sigma_s):
            """
            Samples a matrix of the specified shape using the DOG filter.
            Args:
                x: x coordinate of the receptive filter's center
                y: y coordinate of the receptive filter's center
                shape: Shape of the sampled matrix
                sigma_c: Central gaussian width for the receptive filter
                sigma_s: Surround gaussian width for the receptive filter
            Returns:
                sample: Matrix sample generated from the DOG filter
            """
            sample = np.zeros(shape)
            for i in range(- int(shape[0] / 2), 1 + int(shape[0] / 2)):
                for j in range(- int(shape[1] / 2), 1 + int(shape[1] / 2)):
                    sample[x + i + int(shape[0] / 2),
                        y + j + int(shape[1] / 2)] = DOG(x + i, y + j, sigma_c, sigma_s)
            return sample

        DOG_kernel = DOG_receptive_field(0, 0, SHAPE, SIGMA_C, SIGMA_S)

        figure_num = 1
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(DOG_kernel)
        plt.yticks(np.arange(0, SHAPE[0], int(SHAPE[0] / 5)))
        plt.title('A 21x21 pixel matrix sampled from the DOG filter')
        plt.colorbar()
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        ax_3D = plt.axes(projection='3d')
        X_PLT, Y_PLT = np.meshgrid(np.arange(- int(SHAPE[0] / 2), 1 + int(SHAPE[0] / 2)),
                                np.arange(- int(SHAPE[1] / 2), 1 + int(SHAPE[1] / 2)))
        ax_3D.plot_surface(X_PLT, Y_PLT, DOG_kernel, rstride=1, cstride=1, 
                        cmap='viridis', edgecolor='none')
        ax_3D.set_xlabel('x')
        ax_3D.set_ylabel('y')
        ax_3D.set_zlabel('DOG(x, y)')
        plt.title('The DOG kernel')
        plt.show(block=False)

        # PART B
        print('PART B')

        image = Image.open("hw2_image.bmp")
        image = np.array(image)
        # Since the image is grayscale, we only need a single color channel
        image = image[:,:, 0] 

        def convolve(image, kernel):
            """
            Given a kernel matrix, computes the 2 dimensional convolution 
            of the kernel and an image.
            Args:
                image: The given image
                kernel: The given kernel matrix to be used in the convolution
            Returns:
                result: The resulting matrix after the convolution
            """
            pad_x = int(np.shape(kernel)[0] / 2)
            pad_y = int(np.shape(kernel)[1] / 2)
            img_padding = np.zeros((2 * pad_x + np.shape(image)[0], 2 * pad_y + np.shape(image)[1]))
            img_padding[pad_x: np.shape(image)[0] + pad_x, pad_y: np.shape(image)[1] + pad_y] = image
            result = np.zeros(np.shape(image))
            for i in range(pad_x, np.shape(image)[0] + pad_x):
                for j in range(pad_y, np.shape(image)[1] + pad_y):
                    result[i - pad_x, j - pad_y] = np.sum(
                        img_padding[i - pad_x: i + pad_x + 1, j - pad_y: j + pad_y + 1] * kernel)
            return result

        DOG_image = convolve(image, DOG_kernel)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(DOG_image)
        plt.title('Resulting image after convolution with the DOG kernel')
        plt.show(block=False)

        # PART C 
        print('PART C')

        def detect_edges(filtered_image, threshold):
            """
            Given an appropriately filtered image and an optimal threshold
            value, sets the pixels to 1 if they are above the threshold, and
            to 0 if they are below the threshold.
            Args:
                filtered_image: The image that is filtered such that it is
                    ready for thresholding
                threshold: An optimal threshold value for edge detection
            Returns:
                result: Resulting image after thresholding
            """
            result = filtered_image
            result[np.where(filtered_image >= threshold)] = 1
            result[np.where(filtered_image < threshold)] = 0
            return result

        THRESHOLD = -4
        img_edges_detected = detect_edges(DOG_image, THRESHOLD)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(img_edges_detected, cmap='gray')
        plt.title('Resulting image after edge detection using DOG kernel')
        plt.show(block=False)

        # PART D
        print('PART D')

        def Gabor(x, theta, sigma_l, sigma_w, phi, lambda_):
            """
            Implements a Gabor filter with a specified orientation, and other
            specified parameters.
            filter centered at 0. 
            Args:
                x: The coordinate vector
                theta: The orientation
                sigma_l: A constant
                sigma_w: A constant
                phi: The phase angle
                lambda_: A constant
            Returns: 
                result: The output of the Gabor filter, D(x)
            """
            k = np.array([np.sin(theta), np.cos(theta)])
            k_orth = np.array([np.cos(theta), -np.sin(theta)])
            inner_k = k.dot(x)
            inner_k_orth = k_orth.dot(x)
            exp = np.exp(- (inner_k ** 2) / (2 * (sigma_l ** 2)) - (inner_k_orth ** 2) / (2 * (sigma_w ** 2)))
            result = exp * np.cos(phi + 2 * np.pi * inner_k_orth / lambda_)
            return result
        
        def Gabor_receptive_field(x, y, shape, theta, sigma_l, sigma_w, phi, lambda_):
            """
            Samples a matrix of the specified shape using the Gabor filter.
            Args:
                Args:
                    x: x coordinate for the Gabor filter
                    y: y coordinate for the Gabor filter
                    shape: Shape of the sampled matrix
                    theta: The orientation for the Gabor filter
                    sigma_l: A constant for the Gabor filter
                    sigma_w: A constant for the Gabor filter
                    phi: The phase angle for the Gabor filter
                    lambda_: A constant for the Gabor filter
            Returns:
                sample: Matrix sample generated from the Gabor filter
            """
            sample = np.zeros(shape)
            for i in range(- int(shape[0] / 2), 1 + int(shape[0] / 2)):
                for j in range(- int(shape[1] / 2), 1 + int(shape[1] / 2)):
                    sample[x + i + int(shape[0] / 2),
                        y + j + int(shape[1] / 2)] = Gabor(np.array([x + i, y + j]),
                                                            theta, sigma_l, sigma_w, 
                                                            phi, lambda_)
            return sample

        THETA_0 = 0
        THETA_30 = np.pi / 6
        THETA_60 = np.pi / 3
        THETA_90 = np.pi / 2
        SIGMA_L = 3
        SIGMA_W = 3
        PHI = 0
        LAMBDA = 6

        Gabor_kernel_90 = Gabor_receptive_field(0, 0, SHAPE, THETA_90, SIGMA_L, SIGMA_W, PHI, LAMBDA)
        
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_kernel_90)
        plt.yticks(np.arange(0, SHAPE[0], int(SHAPE[0] / 5)))
        plt.title('A 21x21 pixel matrix sampled from the Gabor filter\n(orientation is 90 degree)')
        plt.colorbar()
        plt.show(block=False)
            
        plt.figure(figure_num)
        figure_num += 1
        ax_3D = plt.axes(projection='3d')
        ax_3D.plot_surface(X_PLT, Y_PLT, Gabor_kernel_90, rstride=1, cstride=1, 
                        cmap='viridis', edgecolor='none')
        ax_3D.set_xlabel('x')
        ax_3D.set_ylabel('y')
        ax_3D.set_zlabel('D(x, y)')
        plt.title('The Gabor kernel (theta = pi / 2)')
        plt.show(block=False)

        # PART E 
        print('PART E')

        Gabor_image_90 = convolve(image, Gabor_kernel_90)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_image_90)
        plt.title('Resulting image after convolution with the Gabor kernel'
                '\n(orientation is 90 degree)')
        plt.show(block=False)

        # PART F 
        print('PART F')

        Gabor_kernel_0 = Gabor_receptive_field(0, 0, SHAPE, THETA_0, SIGMA_L, SIGMA_W, PHI, LAMBDA)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_kernel_0)
        plt.yticks(np.arange(0, SHAPE[0], int(SHAPE[0] / 5)))
        plt.title('A 21x21 pixel matrix sampled from the Gabor filter\n(orientation is 0 degree)')
        plt.colorbar()
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        ax_3D = plt.axes(projection='3d')
        ax_3D.plot_surface(X_PLT, Y_PLT, Gabor_kernel_0, rstride=1, cstride=1, 
                        cmap='viridis', edgecolor='none')
        ax_3D.set_xlabel('x')
        ax_3D.set_ylabel('y')
        ax_3D.set_zlabel('D(x, y)')
        plt.title('The Gabor kernel as a function (theta = 0)')
        plt.show(block=False)

        Gabor_kernel_30 = Gabor_receptive_field(0, 0, SHAPE, THETA_30, SIGMA_L, SIGMA_W, PHI, LAMBDA)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_kernel_30)
        plt.yticks(np.arange(0, SHAPE[0], int(SHAPE[0] / 5)))
        plt.title('A 21x21 pixel matrix sampled from the Gabor filter\n(orientation is 30 degree)')
        plt.colorbar()
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        ax_3D = plt.axes(projection='3d')
        ax_3D.plot_surface(X_PLT, Y_PLT, Gabor_kernel_30, rstride=1, cstride=1, 
                        cmap='viridis', edgecolor='none')
        ax_3D.set_xlabel('x')
        ax_3D.set_ylabel('y')
        ax_3D.set_zlabel('D(x, y)')
        plt.title('The Gabor kernel as a function (theta = 30)')
        plt.show(block=False)

        Gabor_kernel_60 = Gabor_receptive_field(0, 0, SHAPE, THETA_60, SIGMA_L, SIGMA_W, PHI, LAMBDA)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_kernel_60)
        plt.yticks(np.arange(0, SHAPE[0], int(SHAPE[0] / 5)))
        plt.title('A 21x21 pixel matrix sampled from the Gabor filter\n(orientation is 60 degree)')
        plt.colorbar()
        plt.show(block=False)

        plt.figure(figure_num)
        figure_num += 1
        ax_3D = plt.axes(projection='3d')
        ax_3D.plot_surface(X_PLT, Y_PLT, Gabor_kernel_60, rstride=1, cstride=1, 
                        cmap='viridis', edgecolor='none')
        ax_3D.set_xlabel('x')
        ax_3D.set_ylabel('y')
        ax_3D.set_zlabel('D(x, y)')
        plt.title('The Gabor kernel as a function (theta = 60)')
        plt.show(block=False)

        Gabor_image_0 = convolve(image, Gabor_kernel_0)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_image_0)
        plt.title('Resulting image after convolution with the Gabor kernel'
                '\n(orientation is 0 degree)')
        plt.show(block=False)

        Gabor_image_30 = convolve(image, Gabor_kernel_30)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_image_30)
        plt.title('Resulting image after convolution with the Gabor kernel'
                '\n(orientation is 30 degree)')
        plt.show(block=False)

        Gabor_image_60 = convolve(image, Gabor_kernel_60)
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_image_60)
        plt.title('Resulting image after convolution with the Gabor kernel'
                '\n(orientation is 60 degree)')
        plt.show(block=False)

        Gabor_image_combined = Gabor_image_0 + Gabor_image_30 + Gabor_image_60 + Gabor_image_90
        plt.figure(figure_num)
        figure_num += 1
        plt.imshow(Gabor_image_combined)
        plt.title('Resulting image after convolution with the Gabor kernel'
                '\n(0, 30, 60 and 90 degree orientation results were combined)')
        plt.show(block=False)

        plt.show()

efe_acer_21602217_hw2(question)