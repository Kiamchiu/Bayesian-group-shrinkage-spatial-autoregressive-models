function param_array = mcmc_basgl_sa_b( ...
        N_iter, N_init, N_burn, init_param, data, ...
        beta0, B0, c0, C0, alpha0, Alpha0, ...
        d1 ...
    )

    initialize_rng();

    param_array(1) = init_param;
    param_array(N_iter).rho = NaN;

    group_index = data.group_index;

    L = length(init_param.lambdal);
    G = length(unique(group_index));

    psi_array = zeros(N_burn, L + 2);
    psi_array(1, :) = [init_param.lambdal; init_param.rho; init_param.mu];

    kappa_phi = zeros(G, 1);
    K = @(s) [max(-s - 1, -5), s + 1];

    for n = 2:N_init

        if mod(n, 100) == 0
            fprintf('current: %d-th iteration (init phase)\n', n);
        end

        param = param_array(n - 1);
        param = sample_psi(param, data, 0.01 / (L + 2) * eye(L + 2));
        psi_array(n, :) = [param.lambdal; param.rho; param.mu];
        param = sample_beta(param, data, beta0, B0);
        param = sample_sigma_sq(param, data, beta0, B0);
        param = sample_cn(param, data, c0, C0);
        param = sample_alpha(param, data, alpha0, Alpha0);

        [param, kappa_phi] = sample_nu_sq_inv_sa(n, param, group_index, kappa_phi, K);

        param = sample_tau_sq_inv(param);
        param = sample_phi_1_sq(d1, param);

        param_array(n) = param;
    end

    for n = N_init + 1:N_burn

        if mod(n, 100) == 0
            fprintf('current: %d-th iteration (burn-in)\n', n);
        end

        sigma_psi_emp = diag(diag(cov(psi_array(n - N_init:n - 1, :))));
        sigma_psi = 5.1121 / (L + 2) * sigma_psi_emp + 0.000025 / (L + 2) * eye(L + 2);

        param = param_array(n - 1);
        param = sample_psi(param, data, sigma_psi);
        psi_array(n, :) = [param.lambdal; param.rho; param.mu];
        param = sample_beta(param, data, beta0, B0);
        param = sample_sigma_sq(param, data, beta0, B0);
        param = sample_cn(param, data, c0, C0);
        param = sample_alpha(param, data, alpha0, Alpha0);

        [param, kappa_phi] = sample_nu_sq_inv_sa(n, param, group_index, kappa_phi, K);

        param = sample_tau_sq_inv(param);
        param = sample_phi_1_sq(d1, param);

        param_array(n) = param;
    end

    sigma_psi_emp = diag(diag(cov(psi_array(N_burn - N_init + 1:N_burn, :))));
    sigma_psi = 5.1121 / (L + 2) * sigma_psi_emp + 0.000025 / (L + 2) * eye(L + 2);

    for n = N_burn + 1:N_iter

        if mod(n, 100) == 0
            fprintf('current: %d-th iteration (sampling)\n', n);
        end

        param = param_array(n - 1);
        param = sample_psi(param, data, sigma_psi);
        param = sample_beta(param, data, beta0, B0);
        param = sample_sigma_sq(param, data, beta0, B0);
        param = sample_cn(param, data, c0, C0);
        param = sample_alpha(param, data, alpha0, Alpha0);

        [param, kappa_phi] = sample_nu_sq_inv_sa(n, param, group_index, kappa_phi, K);

        param = sample_tau_sq_inv(param);
        param = sample_phi_1_sq(d1, param);

        param_array(n) = param;
    end

end

function param = sample_alpha(param, data, alpha0, Alpha0)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    Cn = param.Cn;
    alpha = param.alpha;
    sigma_sq = param.sigma_sq;

    gamma = lambdal / sum(lambdal);

    Xs = data.Xs;
    Ys = data.Ys;
    Ws = data.Ws;

    [N, T, K] = size(Xs);
    L = length(lambdal);

    for t = 2:T
        Wct = zeros(size(Ws{1}));
        Wlambda_ct = zeros(size(Ws{1}));

        for l = 1:L
            Wlambda_ct = Wlambda_ct + lambdal(l) * Ws{t, l};
            Wct = Wct + gamma(l) * Ws{t, l};
        end

        St = eye(N) - Wlambda_ct;
        Ht = ones(1, N) * (St * Ys(:, t) - (rho * eye(N) + mu * Wct) * Ys(:, t - 1) - reshape(Xs(:, t, :), N, K) * beta - Cn);

        sigma_alpha_t = 1 / (1 / Alpha0 + N / sigma_sq);
        mu_alpha_t = sigma_alpha_t * (alpha0 / Alpha0 + Ht / sigma_sq);

        alpha_t = normrnd(mu_alpha_t, sqrt(sigma_alpha_t));
        alpha(t) = alpha_t;
    end

    param.alpha = alpha;
end

function param = sample_beta(param, data, beta0, B0)
    Xs = data.Xs;
    Ws = data.Ws;
    Ys = data.Ys;

    [N, T, K] = size(Xs);

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    sigma_sq = param.sigma_sq;
    Cn = param.Cn;
    alpha = param.alpha;

    gamma = lambdal / sum(lambdal);
    L = length(lambdal);

    XH_sum = zeros(K, 1);
    XX_sum = zeros(K, K);

    for t = 2:T
        % generate convex combination of weights
        Wct = zeros(size(Ws{1}));
        Wlambda_ct = zeros(size(Ws{1}));

        for l = 1:L
            Wct = Wct + gamma(l) * Ws{t, l};
            Wlambda_ct = Wlambda_ct + lambdal(l) * Ws{t, l};
        end

        St = eye(N) - Wlambda_ct;

        Xt = reshape(Xs(:, t, :), N, K);
        XH_sum = XH_sum + Xt' * (St * Ys(:, t) - (rho * eye(N) + mu * Wct) * Ys(:, t - 1) - Cn - ones(N, 1) * alpha(t)) / sigma_sq;
        XX_sum = XX_sum + Xt' * Xt / sigma_sq;
    end

    Sigma_beta = ((sigma_sq * B0) \ eye(K) + XX_sum) \ eye(K);
    T_beta = Sigma_beta * ((sigma_sq * B0) \ beta0 + XH_sum);

    param.beta = mvnrnd(T_beta, Sigma_beta)';
end

function param = sample_cn(param, data, c0, C0)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    alpha = param.alpha;
    sigma_sq = param.sigma_sq;

    gamma = lambdal / sum(lambdal);

    Xs = data.Xs;
    Ys = data.Ys;
    Ws = data.Ws;

    [N, T, K] = size(Xs);
    L = length(lambdal);

    H_sum = zeros(N, 1);

    for t = 2:T
        Wct = zeros(size(Ws{1}));
        Wlambda_ct = zeros(size(Ws{1}));

        for l = 1:L
            Wct = Wct + gamma(l) * Ws{t, l};
            Wlambda_ct = Wlambda_ct + lambdal(l) * Ws{t, l};
        end

        St = eye(N) - Wlambda_ct;
        Ht = St * Ys(:, t) - (rho * eye(N) + mu * Wct) * Ys(:, t - 1) - reshape(Xs(:, t, :), N, K) * beta - ones(N, 1) * alpha(t);
        H_sum = H_sum + Ht;
    end

    C_cov = (C0 \ eye(N) + (T - 1) / sigma_sq * eye(N)) \ eye(N);
    c_mu = C_cov * (C0 \ c0 + H_sum / sigma_sq);

    Cn = mvnrnd(c_mu, C_cov)';
    param.Cn = Cn;
end

function param = sample_phi_1_sq(d1, param)

    tau_sq_inv = param.tau_sq_inv;
    tau_sq = 1 ./ tau_sq_inv;

    L = length(tau_sq);
    b = d1 +1/2 * sum(tau_sq);
    phi_1_sq = gamrnd(L + 1, 1 / b);
    param.phi_1_sq = phi_1_sq;
end

function param_result = sample_psi(param, data, sigma_psi)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    psi = [lambdal; rho; mu];

    L = length(lambdal);

    stable = 0;

    while stable == 0
        psi_n = mvnrnd(psi, sigma_psi)';

        lambdal_n = psi_n(1:L);
        rho_n = psi_n(L + 1);
        mu_n = psi_n(L + 2);

        param_n = param;
        param_n.lambdal = lambdal_n;
        param_n.rho = rho_n;
        param_n.mu = mu_n;

        if (sum(abs(lambdal_n)) + abs(rho_n) + abs(mu_n) * sum(abs(lambdal_n / sum(lambdal_n))) < 1)

            stable = 1;

            a = min([1, exp(log_posterior(param_n, data) - log_posterior(param, data))]);
            u = rand(1, 1);

            if u <= a
                param_result = param_n;
            else
                param_result = param;
            end

        end

    end

end

function loglikelihood = log_posterior(param, data)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    sigma_sq = param.sigma_sq;
    Cn = param.Cn;
    alpha = param.alpha;
    tau_sq_inv = param.tau_sq_inv;
    nu_sq_inv = param.nu_sq_inv;
    gamma = lambdal / sum(lambdal);
    Xs = data.Xs;
    Ys = data.Ys;
    Ws = data.Ws;
    group_index = data.group_index;

    [N, T, K] = size(Xs);
    L = length(lambdal);
    G = length(unique(group_index));

    Vg_inv = cell(G, 1);

    for g = 1:G
        tau_sq_inv_g = tau_sq_inv(group_index == g);
        Vg_inv{g} = diag(nu_sq_inv(g) + tau_sq_inv_g);
    end

    Lambda = blkdiag(Vg_inv{:});

    HH_sum = 0.0;
    logdetS = 0.0;

    for t = 2:T

        Wct = zeros(size(Ws{1}));
        Wlambda_ct = zeros(size(Ws{1}));

        for l = 1:L
            Wlambda_ct = Wlambda_ct + lambdal(l) * Ws{t, l};
            Wct = Wct + gamma(l) * Ws{t, l};
        end

        St = eye(N) - Wlambda_ct;

        Ht = St * Ys(:, t) - (rho * eye(N) + mu * Wct) * Ys(:, t - 1) - reshape(Xs(:, t, :), N, K) * beta - Cn - ones(N, 1) * alpha(t);
        HH_sum = HH_sum + Ht' * Ht / (2 * sigma_sq);
        logdetS = logdetS + real(log(det(St)));
    end

    logprior =- 1 / (2 * sigma_sq) * lambdal' * Lambda * lambdal;

    loglikelihood = logprior - N * (T - 1) / 2 * log(sigma_sq) + logdetS - HH_sum;
end

function param = sample_sigma_sq(param, data, beta0, B0)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    Cn = param.Cn;
    alpha = param.alpha;
    tau_sq_inv = param.tau_sq_inv;
    nu_sq_inv = param.nu_sq_inv;
    gamma = lambdal / sum(lambdal);
    Xs = data.Xs;
    Ys = data.Ys;
    Ws = data.Ws;
    group_index = data.group_index;

    [N, T, K] = size(Xs);
    L = length(lambdal);
    G = length(unique(group_index));

    Vg_inv = cell(G, 1);

    for g = 1:G
        tau_sq_inv_g = tau_sq_inv(group_index == g);
        Vg_inv{g} = diag(nu_sq_inv(g) + tau_sq_inv_g);
    end

    Lambda = blkdiag(Vg_inv{:});

    HH_sum = zeros(1, 1);

    for t = 2:T

        Wct = zeros(size(Ws{1}));
        Wlambda_ct = zeros(size(Ws{1}));

        for l = 1:L
            Wlambda_ct = Wlambda_ct + lambdal(l) * Ws{t, l};
            Wct = Wct + gamma(l) * Ws{t, l};
        end

        St = eye(N) - Wlambda_ct;
        Ht = St * Ys(:, t) - (rho * eye(N) + mu * Wct) * Ys(:, t - 1) - reshape(Xs(:, t, :), N, K) * beta - Cn - ones(N, 1) * alpha(t);
        HH_sum = HH_sum + Ht' * Ht;
    end

    an = K + N * (T - 1) + L;
    bn = ((beta - beta0)' * (B0 \ (beta - beta0))) + HH_sum + lambdal' * Lambda * lambdal;

    sigma_sq_n = 1 / gamrnd(an / 2, 2 / bn); 

    param.sigma_sq = sigma_sq_n;
end

function param = sample_tau_sq_inv(param)
    sigma = sqrt(param.sigma_sq);
    phi_1 = sqrt(param.phi_1_sq);
    lambdal = param.lambdal;

    L = length(lambdal);

    tau_sq_inv = zeros(length(lambdal), 1);

    for l = 1:L
        mu = sigma * phi_1 / abs(lambdal(l));
        pd = makedist('InverseGaussian', 'mu', max(mu, 0.000001), 'lambda', max(param.phi_1_sq, 0.000001));

        tau_sq_inv(l) = random(pd);
    end

    param.tau_sq_inv = tau_sq_inv;
end

function [param, kappa_phi] = sample_nu_sq_inv_sa(i, param, group_index, kappa_phi, K)

    sigma_sq = param.sigma_sq;
    lambdal = param.lambdal;
    phi_2_g = param.phi_2_g;

    G = length(unique(group_index));

    % Preallocate
    phi_2_g_update = zeros(G, 1);
    nu_sq_inv_update = zeros(G, 1);

    for g = 1:G
        lambda_mu_g = lambdal(group_index == g);
        mg = length(lambda_mu_g);

        if mg == 0
            phi_2_g_update(g) = phi_2_g(g);
            nu_sq_inv_update(g) = 0;
            continue;
        end

        % Current phi and derived nu_sq_inv
        phi_curr = phi_2_g(g);

        % Sample nu_sq_inv from its full conditional
        lambda_mu_norm_sq = norm(lambda_mu_g, 2) ^ 2;
        nu_sq_inv_sample = gigrnd(-0.5, lambda_mu_norm_sq / sigma_sq, exp(2 * phi_curr));

        % Stochastic approximation update for phi_2_g
        s = 1 / i ^ 0.8;
        score = (mg + 1) - 1 ./nu_sq_inv_sample * exp(2 * phi_curr);
        phi_prop = phi_curr + s * score;

        % Boundary handling (same as BGL)
        kappa = kappa_phi(g);
        K_bounds = K(kappa);
        K_l = K_bounds(1);
        K_u = K_bounds(2);
        max_step = 3 - 2 * (1 - i ^ (-0.1));

        if phi_prop >= K_l && phi_prop <= K_u && abs(phi_prop - phi_curr) <= max_step
            % Accept SA step
            phi_2_g_update(g) = phi_prop;
            nu_sq_inv_update(g) = nu_sq_inv_sample;
        else
            % Trigger boundary handling
            kappa_phi(g) = kappa + 1;

            if phi_prop > K_u
                phi_new = unifrnd(min(phi_curr, K_u), max(phi_curr, K_u));
            elseif phi_prop < K_l
                phi_new = unifrnd(min(phi_curr, K_l), max(phi_curr, K_l));
            else
                if phi_prop > phi_curr
                    phi_new = phi_curr + max_step;
                else
                    phi_new = phi_curr - max_step;
                end

            end

            phi_2_g_update(g) = phi_new;
            nu_sq_inv_update(g) = gigrnd(-0.5, lambda_mu_norm_sq / sigma_sq, exp(2 * phi_new));
        end

    end

    param.phi_2_g = phi_2_g_update;
    param.nu_sq_inv = nu_sq_inv_update;
end
