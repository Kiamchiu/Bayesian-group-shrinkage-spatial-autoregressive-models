function [data, true_param] = dgp(loop,true_param, N, T, Ws,Xs,Cn,alpha,group_index)
% DATA STRUCTURE: index follows the corresponding position
% dist_X: T * K
% Ws: T * L
% Xs: N * T * K 
% Ys: N * T
% --------------------------------------------------------
    seed = uint32(loop ); 
    rng(seed, 'twister');

    assert((size(Ws{1}, 1) == size(Ws{1}, 2)) && (size(Ws{1}, 1) == N), 'Dimension of spatial weight matrix does not equal to N');
    assert(size(Ws, 1) == T + 10, 'Time horizon of spatial weight matrix does not equal to T')
    assert(size(Ws, 2) == length(true_param.lambdal), 'Number of spatial weight matrix does not math parameter')

    lambdal = true_param.lambdal;
    rho = true_param.rho;
    mu = true_param.mu;
    beta = true_param.beta;
    sigma_sq = true_param.sigma_sq;

    gamma = lambdal / sum(lambdal);

    K = length(beta);
    L = length(lambdal);
    G = length(unique(lambdal));

    assert(length(group_index) == L, 'grouping index is not compatible with parameters')
    assert(G <= L, 'number of groups should be bounded by number of channels')
    
    T_ext = T + 10; % additional 10 time periods to make series converge
    
    Ys = zeros(N, T_ext);

    % initialize first period of observations
    Ys(:, 1) = randn(N, 1);

    for t = 2 : T_ext
        % generate convex combination of weights
        Wct = zeros(size(Ws{1}));
        Wlambda_ct = zeros(size(Ws{1}));

        for l = 1:L
            Wct = Wct + gamma(l) * Ws{t, l};
            Wlambda_ct = Wlambda_ct + lambdal(l) * Ws{t, l};
        end

        St = eye(N) - Wlambda_ct;

        Vnt = sqrt(sigma_sq) * randn(N, 1);

        Ys(:, t) = St \ ((rho * eye(N) + mu * Wct) * Ys(:, t-1) + reshape(Xs(:, t, :), N, K) * beta + Cn + ones(N, 1) * alpha(t) + Vnt);
    end

    % take the last T periods of observations as samples
    data.Xs = Xs(:, end-T+1:end, :);
    data.Ys = Ys(:, end-T+1:end);
    data.Ws = Ws(end-T+1:end, :);
    data.group_index = group_index;
    true_param.Cn = Cn;
    true_param.alpha = alpha(end-T+1:end);
end
