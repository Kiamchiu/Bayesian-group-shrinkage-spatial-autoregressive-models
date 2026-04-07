function param_table = construct_param_tab(param_array, burnin)

    assert(size(param_array, 2) ~= 0, 'empty parameter array');

    N_iter = length(param_array);
    N_burn = floor(N_iter * burnin);

    param_array = param_array(N_burn + 1:end);

    var_names = {'rho', 'mu', 'beta', 'sigma_sq'};

    lambda_matrix = sum([param_array.lambdal], 1)';
    gamma_matrix = cell2mat(cellfun(@(t) t / sum(t), num2cell([param_array.lambdal], 1), 'UniformOutput', false))';

    param_matrix = [lambda_matrix gamma_matrix];

    L = length(param_array(1).lambdal);
    gamma_label = [repmat("", 1, L)];

    for l = 1:L
        gamma_label(l) = "gamma" + string(l);
    end

    param_label = ["lambda", gamma_label];

    for i = 1:length(var_names)
        var = var_names{i};

        param_matrix = [param_matrix [param_array.(var)]'];

        var_len = length(param_array(1).(var));

        if var_len > 1
            sub_var_names = [];

            for j = 1:var_len
                sub_var_names = [sub_var_names, var + string(j)];
            end

            param_label = [param_label, sub_var_names];
        else
            param_label = [param_label, var];
        end

    end

    param_table = array2table(param_matrix, ...
        'VariableNames', cellstr(param_label)');
end
