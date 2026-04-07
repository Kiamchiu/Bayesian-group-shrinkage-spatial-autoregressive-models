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
sigma = ones(3, 1);

% Step 2: Construct the covariance matrix
cov_matrix = rho1 * ones(3) + diag((1 - rho1) * ones(1, 3));
cov_matrix = cov_matrix .* (sigma' * sigma); % Adjust for the variances

% Step 3: Apply the Cholesky decomposition to obtain correlated vectors
L_triangle = chol(cov_matrix, 'lower');
correlated_vectors_v = randn(N, 3) * L_triangle';
correlated_vectors_w = randn(N, 3) * L_triangle';

% Separate the vectors
v1 = correlated_vectors_v(:, 1);
v2 = correlated_vectors_v(:, 2);
v3 = correlated_vectors_v(:, 3);

w1 = correlated_vectors_w(:, 1);
w2 = correlated_vectors_w(:, 2);
w3 = correlated_vectors_w(:, 3);

W1 = make_neighborsw(v1, w1, 5);
W2 = make_neighborsw(v2, w2, 5);
W3 = make_neighborsw(v3, w3, 5);

% Second group
rho2 = 0.8;
sigma = ones(5, 1);

cov_matrix = rho2 * ones(5) + diag((1 - rho2) * ones(1, 5));
cov_matrix = cov_matrix .* (sigma' * sigma);

L_triangle = chol(cov_matrix, 'lower');
correlated_vectors_v = randn(N, 5) * L_triangle';
correlated_vectors_w = randn(N, 5) * L_triangle';

% Separate the vectors
v1 = correlated_vectors_v(:, 1);
v2 = correlated_vectors_v(:, 2);
v3 = correlated_vectors_v(:, 3);
v4 = correlated_vectors_v(:, 4);
v5 = correlated_vectors_v(:, 5);

w1 = correlated_vectors_w(:, 1);
w2 = correlated_vectors_w(:, 2);
w3 = correlated_vectors_w(:, 3);
w4 = correlated_vectors_w(:, 4);
w5 = correlated_vectors_w(:, 5);

W4 = make_neighborsw(v1, w1, 5);
W5 = make_neighborsw(v2, w2, 5);
W6 = make_neighborsw(v3, w3, 5);
W7 = make_neighborsw(v4, w4, 5);
W8 = make_neighborsw(v5, w5, 5);

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
a0 = 1;
b0 = 2;
c0 = zeros(N, 1);
C0 = eye(N);
alpha0 = 0;
Alpha0 = 1;

%% Batch specification
N_repeat = 500;
dim = 1 + L + 3 + K;
pe_gigg_array = zeros(N_repeat, dim);
pm_gigg_array = cell(N_repeat, 1);
coverage_gigg = zeros(N_repeat, dim);
pspro_gigg = zeros(N_repeat, dim);

N_iter = 20000;
N_init = N_iter * 0.02;
N_burn = N_iter * 0.2;

%% MCMC estimation
parfor loop = 1:N_repeat
    % generate random data
    [data, ~] = dgp(loop, true_param, N, T, Ws, Xs, Cn, alpha, group_index);
    fprintf('Current chain: %d/%d\n', loop, N_repeat);
    % initialize parameter
    param = struct('lambdal', 0.8 / L * ones(L, 1), 'rho', 0.0, 'mu', 0.0, 'beta', zeros(K, 1), 'sigma_sq', 1.0, 'Cn', Cn, 'alpha', alpha(T_ext - T + 1:end, 1));

    gigg_param = param;
    ag = [3, 0.5]'; % For DGP1

    gigg_param.bg = repelem(0, G)';
    gigg_param.tau_sq = 1.0;
    gigg_param.delta_sq = ones(L, 1);
    gigg_param.nu_sq = ones(G, 1);
    gigg_param.iota = 1.0;
    param_array_gigg = mcmc_gigg_sa_b(N_iter, N_init, N_burn, gigg_param, data, ag, beta0, B0, c0, C0, alpha0, Alpha0);

    % Evaluate the estimation performance
    [~, pm_gigg, ~, ~, pe_gigg] = evaluate_simulation(true_param, param_array_gigg, 0.2);
    pe_gigg_array(loop, :) = pe_gigg;
    pm_gigg_array{loop} = pm_gigg;

    for i = 1:dim
        bound_gigg = hpdi(pm_gigg(:, i), 0.95);

        if (bound_gigg(2) > true_param_array(1, i)) && (bound_gigg(1) < true_param_array(1, i))
            coverage_gigg(loop, i) = 1;
        end

        if (bound_gigg(1) > 0) || (bound_gigg(2) < 0)
            pspro_gigg(loop, i) = 1;
        end

    end

end

mean_gigg = mean(pe_gigg_array);
std_gigg = std(pe_gigg_array);
bias_gigg = mean(pe_gigg_array) - true_param_array;
rmse_gigg = sqrt(mean((pe_gigg_array - true_param_array) .^ 2));
cover_gigg = sum(coverage_gigg) / N_repeat;
psp_gigg = sum(pspro_gigg) / N_repeat;

estimation_result = table(mean_gigg', std_gigg', bias_gigg', rmse_gigg', cover_gigg', psp_gigg', ...
    'VariableNames', ...
    {'mean_gigg', 'std_gigg', 'bias_gigg', 'rmse_gigg', 'cover_gigg', 'psp_gigg'});

writetable(estimation_result, 'simu_gigg_dgp1.csv')
save('simu_gigg_dgp1.mat');
