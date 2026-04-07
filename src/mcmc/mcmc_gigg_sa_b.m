function param_array = mcmc_gigg_sa_b( ...
        N_iter, N_init, N_burn, init_param, data, ...
        ag, beta0, B0, c0, C0, alpha0, Alpha0 ...
    )

    initialize_rng(); % Set the seed for reproducibility

    param_array(1) = init_param;
    param_array(N_iter).rho = NaN; % preallocation of structs in order to save memory

    group_index = data.group_index;
    L = length(init_param.lambdal);
    Xs = data.Xs;
    Ys = data.Ys;
    Ws = data.Ws;

    psi_array = zeros(N_burn, L + 2);
    psi_array(1, :) = [init_param.lambdal; init_param.rho; init_param.mu];

    G = length(unique(group_index));
    kappa_b = zeros(G, 1);
    K = @(s) [max(-s - 1, -5), s + 1];

    for n = 2:N_init

        if mod(n, 100) == 0
            fprintf('current: %d-th iteration\n', n);
        end

        param = param_array(n - 1);
        param = sample_psi(param, group_index, Xs, Ys, Ws, 0.0001 * eye(L + 2));
        psi_array(n, :) = [param.lambdal; param.rho; param.mu];
        param = sample_tau_sq(param, group_index);
        param = sample_nu_sq(ag, param, group_index);
        [param, kappa_b] = sample_delta_sq(n, param, group_index, kappa_b, K);
        param = sample_iota(param);
        param = sample_beta(param, Xs, Ys, Ws, beta0, B0);
        param = sample_sigma_sq(param, Xs, Ys, Ws, beta0, B0);
        param = sample_cn(param, Xs, Ys, Ws, c0, C0);
        param = sample_alpha(param, Xs, Ys, Ws, alpha0, Alpha0);

        param_array(n) = param;

    end

    for n = N_init + 1:N_burn

        if mod(n, 100) == 0
            fprintf('current: %d-th iteration\n', n);
        end

        sigma_psi_emp = cov(psi_array(n - N_init:n - 1, :));
        sigma_psi = 5.1121 / (L + 2) * sigma_psi_emp + 0.000025 / (L + 2) * eye(L + 2);

        param = param_array(n - 1);
        param = sample_psi(param, group_index, Xs, Ys, Ws, sigma_psi);
        psi_array(n, :) = [param.lambdal; param.rho; param.mu];
        param = sample_tau_sq(param, group_index);
        param = sample_nu_sq(ag, param, group_index);
        [param, kappa_b] = sample_delta_sq(n, param, group_index, kappa_b, K);
        param = sample_iota(param);
        param = sample_beta(param, Xs, Ys, Ws, beta0, B0);
        param = sample_sigma_sq(param, Xs, Ys, Ws, beta0, B0);
        param = sample_cn(param, Xs, Ys, Ws, c0, C0);
        param = sample_alpha(param, Xs, Ys, Ws, alpha0, Alpha0);

        param_array(n) = param;
    end

    sigma_psi_emp = cov(psi_array(N_burn - N_init + 1:N_burn, :));
    sigma_psi = 5.1121 / (L + 2) * sigma_psi_emp + 0.000025 / (L + 2) * eye(L + 2);

    for n = N_burn + 1:N_iter

        if mod(n, 100) == 0
            fprintf('current: %d-th iteration\n', n);
        end

        param = param_array(n - 1);
        param = sample_psi(param, group_index, Xs, Ys, Ws, sigma_psi);
        param = sample_tau_sq(param, group_index);
        param = sample_nu_sq(ag, param, group_index);
        [param, kappa_b] = sample_delta_sq(n, param, group_index, kappa_b, K);
        param = sample_iota(param);
        param = sample_beta(param, Xs, Ys, Ws, beta0, B0);
        param = sample_sigma_sq(param, Xs, Ys, Ws, beta0, B0);
        param = sample_cn(param, Xs, Ys, Ws, c0, C0);
        param = sample_alpha(param, Xs, Ys, Ws, alpha0, Alpha0);

        param_array(n) = param;
    end

end

function log_posterior = log_posterior(param, Xs, Ys, Ws, group_index)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    sigma_sq = param.sigma_sq;
    Cn = param.Cn;
    alpha = param.alpha;
    tau_sq = param.tau_sq;
    nu_sq = param.nu_sq;
    delta_sq = param.delta_sq;

    gamma = lambdal / sum(lambdal);

    [N, T, K] = size(Xs);
    L = length(lambdal);
    G = length(unique(group_index));

    Nu = cell(G, 1);

    for g = 1:G
        mg = length(group_index(group_index == g));
        Nu{g} = diag(repelem(1 / nu_sq(g), mg));
    end

    Nu = blkdiag(Nu{:});
    Delta = diag(1 ./ delta_sq);

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
        logdetS = logdetS + log(det(St));
    end

    loglikelihood = logdetS - HH_sum;
    prior =- 0.5 / tau_sq * lambdal' * Nu * Delta * lambdal;
    log_posterior = loglikelihood + prior;
end

function param = sample_alpha(param, Xs, Ys, Ws, alpha0, Alpha0)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    Cn = param.Cn;
    alpha = param.alpha;
    sigma_sq = param.sigma_sq;

    gamma = lambdal / sum(lambdal);

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

function param = sample_beta(param, Xs, Ys, Ws, beta0, B0)

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

function param = sample_cn(param, Xs, Ys, Ws, c0, C0)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    alpha = param.alpha;
    sigma_sq = param.sigma_sq;

    gamma = lambdal / sum(lambdal);

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

function param_result = sample_psi(param, group_index, Xs, Ys, Ws, sigma_psi)

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

        if (sum(abs(lambdal_n)) + abs(rho_n) + sum(abs(mu_n) * abs(lambdal_n / sum(lambdal_n))) < 1)
            stable = 1;

            a = min([1, exp(log_posterior(param_n, Xs, Ys, Ws, group_index) - log_posterior(param, Xs, Ys, Ws, group_index))]);
            u = rand(1, 1);

            if u <= a
                param_result = param_n;
            else
                param_result = param;
            end

        end

    end

end

function param = sample_sigma_sq(param, Xs, Ys, Ws, beta0, B0)

    lambdal = param.lambdal;
    rho = param.rho;
    mu = param.mu;
    beta = param.beta;
    Cn = param.Cn;
    alpha = param.alpha;
    iota = param.iota;

    gamma = lambdal / sum(lambdal);

    [N, T, K] = size(Xs);
    L = length(lambdal);

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

    an = N * (T - 1) + K + 1;
    bn = 1 / iota + ((beta - beta0)' * (B0 \ (beta - beta0))) + HH_sum + 0.000001;

    sigma_sq_n = 1 / gamrnd(an / 2, 2 / bn);

    param.sigma_sq = sigma_sq_n;
end

function param = sample_tau_sq(param, group_index)

    lambdal = param.lambdal;
    nu_sq = param.nu_sq;
    delta_sq = param.delta_sq;
    iota = param.iota;

    L = length(param.lambdal);
    G = length(unique(group_index));

    Nu = cell(G, 1);

    for g = 1:G
        mg = length(group_index(group_index == g));
        Nu{g} = diag(repelem(1 / nu_sq(g), mg));
    end

    Nu = blkdiag(Nu{:});
    Delta = diag(1 ./ delta_sq);

    an = (L + 1) / 2;
    bn = 1/2 * lambdal' * Nu * Delta * lambdal + 1 / iota + 0.000001;

    tau_sq_n = 1 / gamrnd(an, 1 / bn);

    param.tau_sq = tau_sq_n;
end

function param = sample_nu_sq(ag, param, group_index)

    lambdal = param.lambdal;
    delta_sq = param.delta_sq;
    tau_sq = param.tau_sq;

    G = length(unique(group_index));
    nu_sq = zeros(G, 1);

    for g = 1:G
        g_idx = find(group_index == g);
        mg = length(g_idx);

        delta_sq_g = delta_sq(g_idx);
        lambdal_g = lambdal(g_idx);

        pn = ag(g, 1) - mg / 2;
        an = 2;
        bn = 1 / tau_sq * sum((lambdal_g .^ 2) ./ delta_sq_g) + 0.000001;

        nu_sq(g, 1) = gigrnd(pn, an, bn);
    end

    param.nu_sq = nu_sq;
end

function [param, kappa_b] = sample_delta_sq(i, param, group_index, kappa_b, K)

    lambdal = param.lambdal;
    nu_sq = param.nu_sq;
    tau_sq = param.tau_sq;
    bg = param.bg;

    L = length(lambdal);
    G = length(bg);

    delta_sq_n = zeros(L, 1);

    for l = 1:L
        bg_exp = exp(bg);
        an = bg_exp(group_index(l)) +1/2;
        bn = 1 + lambdal(l) ^ 2 / (2 * tau_sq * nu_sq(group_index(l)));

        delta_sq_n(l) = 1 / gamrnd(an, 1 / bn + 0.000001);
    end

    bg_update = zeros(G, 1);
    delta_sq_update = zeros(L, 1);

    for g = 1:G

        kappa = kappa_b(g);

        K_result = K(kappa);
        K_l = K_result(1);
        K_u = K_result(2);

        delta_sq_n_g = delta_sq_n(group_index == g);
        mg = length(delta_sq_n_g);
        s = 1 / i;
        bg_update_g = bg(g) - s * exp(bg(g)) * (mg * psi(exp(bg(g))) + sum(log(delta_sq_n_g)));

        if bg_update_g <= K_u && bg_update_g >= K_l && abs(bg_update_g - bg(g)) <= 3 - 2 * (1 - i ^ (-0.1))

            bg_update(g) = bg_update_g;
            delta_sq_update(group_index == g) = delta_sq_n(group_index == g);

        elseif bg_update_g > K_u

            kappa_b(g) = kappa + 1;

            uniform_l = min(bg(g), K_u);
            uniform_u = max(bg(g), K_u);

            bg_update_g = unifrnd(uniform_l, uniform_u);
            bg_update(g) = bg_update_g;

            g_index = find(group_index == g);

            for l = g_index
                an = exp(bg_update_g) +1/2;
                bn = 1 + lambdal(l) ^ 2 / (2 * tau_sq * nu_sq(g));

                delta_sq_n(l) = 1 / gamrnd(an, 1 / bn + 0.000001);
            end

            delta_sq_update(group_index == g) = delta_sq_n(group_index == g);

        elseif bg_update_g < K_l

            kappa_b(g) = kappa + 1;

            uniform_l = min(bg(g), K_l);
            uniform_u = max(bg(g), K_l);

            bg_update_g = unifrnd(uniform_l, uniform_u);
            bg_update(g) = bg_update_g;

            g_index = find(group_index == g);

            for l = g_index
                an = exp(bg_update_g) +1/2;
                bn = 1 + lambdal(l) ^ 2 / (2 * tau_sq * nu_sq(g));

                delta_sq_n(l) = 1 / gamrnd(an, 1 / bn + 0.000001);
            end

            delta_sq_update(group_index == g) = delta_sq_n(group_index == g);
        else
            kappa_b(g) = kappa + 1;

            if bg_update_g > bg(g)
                bg_update_g = bg(g) + 3 - 2 * (1 - i ^ (-0.1));
            else
                bg_update_g = bg(g) - 3 + 2 * (1 - i ^ (-0.1));
            end

            bg_update(g) = bg_update_g;
            g_index = find(group_index == g);

            for l = g_index
                an = exp(bg_update_g) +1/2;
                bn = 1 + lambdal(l) ^ 2 / (2 * tau_sq * nu_sq(g));

                delta_sq_n(l) = 1 / gamrnd(an, 1 / bn + 0.000001);
            end

            delta_sq_update(group_index == g) = delta_sq_n(group_index == g);
        end

    end

    param.bg = bg_update;
    param.delta_sq = delta_sq_update;
end

function param = sample_iota(param)

    tau_sq = param.tau_sq;
    sigma_sq = param.sigma_sq;

    an = 1;
    bn = 1 / tau_sq + 1 / sigma_sq + 0.000001;

    iota_n = 1 / gamrnd(an, 1 / bn);

    param.iota = iota_n;
end
