clear;

global SEED;
SEED = 1018;
initialize_rng();

N = 200;
T = 40;

%% True parameters
true_param.rho = 0.3;
true_param.mu = -0.1;
true_param.beta = [1.0; 1.0];
true_param.sigma_sq = 1.0;

true_lambda = 0.4;
true_gamma = [0.3; 0.3; 0.4; 0.00; 0.00; 0.00; 0.00; 0.00];

group_index = [1, 1, 1, 2, 2, 2, 2, 2];

G = length(unique(group_index));
L = length(true_gamma);
K = length(true_param.beta);

true_param.lambdal = true_lambda * true_gamma;

var_names = {'rho', 'mu', 'beta', 'sigma_sq'};
gamma = true_param.lambdal / sum(true_param.lambdal);
true_param_array = [sum(true_param.lambdal), gamma'];

for i = 1:length(var_names)
    var = var_names{i};
    true_param_array = [true_param_array, [true_param.(var)]'];
end

%% Correlated Matrices by Spatially Correlated Positions
% First group
rho1 = 0.8;
sigma1 = ones(3, 1);

% Step 2: Construct the covariance matrix
cov_matrix1 = rho1 * ones(3) + diag((1 - rho1) * ones(1, 3));
cov_matrix1 = cov_matrix1 .* (sigma1' * sigma1); % Adjust for the variances

% Step 3: Apply the Cholesky decomposition to obtain correlated vectors
L_triangle1 = chol(cov_matrix1, 'lower');
correlated_vectors_v1 = randn(N, 3) * L_triangle1';
correlated_vectors_w1 = randn(N, 3) * L_triangle1';

% Separate the vectors
v1 = correlated_vectors_v1(:, 1);
v2 = correlated_vectors_v1(:, 2);
v3 = correlated_vectors_v1(:, 3);

w1 = correlated_vectors_w1(:, 1);
w2 = correlated_vectors_w1(:, 2);
w3 = correlated_vectors_w1(:, 3);

W1 = make_neighborsw(v1, w1, 5);
W2 = make_neighborsw(v2, w2, 5);
W3 = make_neighborsw(v3, w3, 5);

% Second group
rho2 = 0.8;
sigma2 = ones(5, 1);

cov_matrix2 = rho2 * ones(5) + diag((1 - rho2) * ones(1, 5));
cov_matrix2 = cov_matrix2 .* (sigma2' * sigma2);

L_triangle2 = chol(cov_matrix2, 'lower');
correlated_vectors_v2 = randn(N, 5) * L_triangle2';
correlated_vectors_w2 = randn(N, 5) * L_triangle2';

% Separate the vectors
v4 = correlated_vectors_v2(:, 1);
v5 = correlated_vectors_v2(:, 2);
v6 = correlated_vectors_v2(:, 3);
v7 = correlated_vectors_v2(:, 4);
v8 = correlated_vectors_v2(:, 5);

w4 = correlated_vectors_w2(:, 1);
w5 = correlated_vectors_w2(:, 2);
w6 = correlated_vectors_w2(:, 3);
w7 = correlated_vectors_w2(:, 4);
w8 = correlated_vectors_w2(:, 5);

W4 = make_neighborsw(v4, w4, 5);
W5 = make_neighborsw(v5, w5, 5);
W6 = make_neighborsw(v6, w6, 5);
W7 = make_neighborsw(v7, w7, 5);
W8 = make_neighborsw(v8, w8, 5);

Ws = cell(T + 10, L);

for t = 1:T + 10
    Ws{t, 1} = W1;
    Ws{t, 2} = W2;
    Ws{t, 3} = W3;
    Ws{t, 4} = W4;
    Ws{t, 5} = W5;
    Ws{t, 6} = W6;
    Ws{t, 7} = W7;
    Ws{t, 8} = W8;
end

%% Generate time and individual effects
% generate time effects
T_ext = T + 10;
alpha = randn(T_ext, 1);
alpha(end - T + 1) = 0.0;

% generate covariates
Xs = zeros(N, T_ext, K);

for t = 1:T_ext
    Xs(:, t, :) = normrnd(0, 2, N, K);
end

% generate individual effects
Xs_mean = mean(Xs, 2);
Cn = zeros(N, 1);

for k = 1:K
    Cn = Cn + 2 * Xs_mean(:, k);
end

Cn = Cn + randn(N, 1);


%% hyper-parameters
beta0 = zeros(K, 1);
B0 = eye(K);
c0 = zeros(N, 1);
C0 = 10 * eye(N);
alpha0 = 0;
Alpha0 = 10;

a = 0.1;
b = 0.1;

%% Batch specification
N_repeat = 500;
dim = 1 + L + 3 + K;
pe_bal_array = zeros(N_repeat, dim);
pm_bal_array = cell(N_repeat, 1);
coverage_bal = zeros(N_repeat, dim);
pspro_bal = zeros(N_repeat, dim);

%% MCMC estimation
N_iter = 20000;
N_init = N_iter * 0.02;
N_burn = N_iter * 0.2;

parfor loop = 1:N_repeat
    % generate random data
    [data, ~] = dgp(loop, true_param, N, T, Ws, Xs, Cn, alpha, group_index);
    fprintf('Current chain: %d/%d\n', loop, N_repeat);
    % initialize parameter
    param = struct('lambdal', 0.8 / L * ones(L, 1), 'rho', 0.0, 'mu', 0.0, 'beta', zeros(K, 1), 'sigma_sq', 1.0, 'Cn', Cn, 'alpha', alpha(T_ext - T + 1:end, 1));

    bal_param = param;
    bal_param.omega_sq = ones(L, 1);
    bal_param.phi_sq = ones(L, 1);

    param_array_bal = mcmc_bal(N_iter, N_init, N_burn, bal_param, data, beta0, B0, a, b, c0, C0, alpha0, Alpha0);

    % Evaluate the estimation performance

    [table_bal, pm_bal, ~, ~, pe_bal] = evaluate_simulation(true_param, param_array_bal, 0.2);

    pe_bal_array(loop, :) = pe_bal;
    pm_bal_array{loop} = pm_bal;

    for i = 1:dim
        bound_bal = hpdi(pm_bal(:, i), 0.95);

        if (bound_bal(2) > true_param_array(1, i)) && (bound_bal(1) < true_param_array(1, i))
            coverage_bal(loop, i) = 1;
        end

        if (bound_bal(1) > 0) || (bound_bal(2) < 0)
            pspro_bal(loop, i) = 1;
        end

    end

end

mean_bal = mean(pe_bal_array);
std_bal = std(pe_bal_array);
bias_bal = mean(pe_bal_array) - true_param_array;
rmse_bal = sqrt(mean((pe_bal_array - true_param_array) .^ 2));
cover_bal = sum(coverage_bal) / N_repeat;
psp_bal = sum(pspro_bal) / N_repeat;

estimation_result = table(mean_bal', std_bal', bias_bal', rmse_bal', cover_bal', psp_bal', ...
    'VariableNames', ...
    {'mean_bal', 'std_bal', 'bias_bal', 'rmse_bal', 'cover_bal', 'psp_bal'});

writetable(estimation_result, 'simu_bal_dgp1.csv')
save('simu_bal_dgp1.mat')
