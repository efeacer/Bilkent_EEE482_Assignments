% QUESTION 3

% PART A 
fprintf('PART A\n');

response_magnitudes1 = zeros(50, 1);
response_magnitudes2 = zeros(50, 1);
for i = 1:50
    stimulus = zeros(1, 50);
    stimulus(i) = 1;
    fprintf('The stimulus:\n');
    disp(stimulus)
    response1 = unknownNeuron1(stimulus);
    fprintf('Response of neuron 1:\n');
    disp(response1);
    response_magnitudes1(i) = norm(response1);
    response2 = unknownNeuron2(stimulus);
    fprintf('Response of neuron 2:\n');
    disp(response2);
    response_magnitudes2(i) = norm(response2);
end
fprintf('Even the magnitudes of the responses are different for neuron 1.\n');
disp(response_magnitudes1);
fprintf('Neuron 2 elicits the same response magnitude for all stimulus.\n');
fprintf('However, the response is different as it can be seen from the previous print outs.\n');
disp(response_magnitudes2);

stimulus1 = zeros(1, 50);
stimulus1(1) = 1;
stimulus2 = zeros(1, 50);
stimulus2(2) = 1;
sample_stimulus = stimulus1 + stimulus2;
fprintf('Sample stimulus:\n');
disp(sample_stimulus);
expected_response1 = unknownNeuron1(stimulus1) + unknownNeuron1(stimulus2);
actual_response1 = unknownNeuron1(sample_stimulus);
fprintf('Check for neuron 1:\n');
is_expected1 = isequal(expected_response1, actual_response1);
disp(is_expected1);
expected_response2 = unknownNeuron2(stimulus1) + unknownNeuron2(stimulus2);
actual_response2 = unknownNeuron2(sample_stimulus);
fprintf('Check for neuron 2:\n');
is_expected2 = isequal(expected_response2, actual_response2);
disp(is_expected2);

% PART B
fprintf('PART B\n');

analyze_stimulus_frequency(); 

% PART C
fprintf('PART C\n');

analyze_stimulus_intensity(); 

% PART D
fprintf('PART D\n');

analyze_noisy_stim_frequency(1);
analyze_noisy_stim_frequency(2.5);
analyze_noisy_stim_frequency(5);

analyze_noisy_stim_intensity(1)
analyze_noisy_stim_intensity(2.5)
analyze_noisy_stim_intensity(5)

% PART B functions

function analyze_stimulus_frequency()
    frequencies = linspace(0, pi / 5);
    stimulus = zeros(100, 50);
    response_magnitudes = zeros(1, 100);
    for i = 1:100
        stimulus(i,:) = stimulus(i,:) + cos((1:50) * frequencies(i));
        response_magnitudes(i) = process_response1(stimulus(i,:));
    end
    [response_magnitude, optimal_index] = max(response_magnitudes);
    fprintf('The optimal frequency value is:\n');
    disp(frequencies(optimal_index))
    fprintf('The maximum response magnitude is:\n');
    disp(response_magnitude)
    figure
    plot(frequencies, response_magnitudes);
    title('Response Magnitude vs Temporal Frequency');
    ylabel('Response Magnitude');
    xlabel('Temporal Frequency');
end

function response_magnitude = process_response1(stimulus)
    response = unknownNeuron1(stimulus);
    response_magnitude = norm(response);
end

% PART C functions

function analyze_stimulus_intensity()
    intensities = linspace(0, 20);
    stimulus = zeros(100, 50);
    response_magnitudes = zeros(1, 100);
    for i = 1:100
        stimulus(i,:) = stimulus(i,:) + intensities(i);
        response_magnitudes(i) = process_response2(stimulus(i,:));
    end
    [response_magnitude, optimal_index] = max(response_magnitudes);
    fprintf('The optimal intensity value is:\n');
    disp(intensities(optimal_index))
    fprintf('The maximum response magnitude is:\n');
    disp(response_magnitude)
    figure
    plot(intensities, response_magnitudes);
    title('Response Magnitude vs Stimulus Intensity');
    ylabel('Response Magnitude');
    xlabel('Stimulus Intensity');
end

function response_magnitude = process_response2(stimulus)
    response = unknownNeuron2(stimulus);
    response_magnitude = norm(response);
end

% PART D functions

function analyze_noisy_stim_frequency(sigma)
    frequencies = linspace(0, pi / 5, 500);
    response_magnitudes = zeros(100, 500);
    stimulus = zeros(500, 50);
    for i = 1:100
        for j = 1:500
            stimulus(j,:) = normrnd(0, sigma, [1, 50]) + cos((1:50) * frequencies(j));
            response_magnitudes(i, j) = process_response1(stimulus(j,:));
        end
    end
    means = mean(response_magnitudes);
    figure
    hold on
    errorbar(means, std(response_magnitudes), '.r');
    plot(1:500, means, '.b');
    header = ['Response Magnitude vs Temporal Frequency (std=', num2str(sigma), ')'];
    title(header);
    ylabel('Average Response Magnitude');
    xlabel('Temporal Frequency');
    hold off
end

function analyze_noisy_stim_intensity(sigma)
    intensities = linspace(0, 20, 500);
    response_magnitudes = zeros(100, 500);
    stimulus = zeros(500, 50);
    for i = 1:100
        for j = 1:500
            stimulus(j,:) = normrnd(0, sigma, [1,50]) + intensities(j);
            response_magnitudes(i, j) = process_response2(stimulus(j,:));
        end
    end
    means = mean(response_magnitudes);
    figure
    hold on
    errorbar(means, std(response_magnitudes), '.r');
    plot(1:500, means, '.b');
    header = ['Response Magnitude vs Stimulus Intensity (std=', num2str(sigma), ')'];
    title(header);
    ylabel('Average Response Magnitude');
    xlabel('Stimulus Intensity');
    hold off
end
