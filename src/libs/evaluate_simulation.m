function [result_table, param_matrix, bias, rmse, point_estimate] = evaluate_simulation(true_param, param_array, burnin)

    param_table = construct_param_tab(param_array, burnin);
    param_matrix = table2array(param_table);

    param_label = param_table.Properties.VariableNames;

    var_names = {'rho', 'mu', 'beta', 'sigma_sq'};

    lambdal = true_param.lambdal;
    gamma = lambdal / sum(lambdal);

    true_param_array = [sum(lambdal), gamma'];

    for i = 1:length(var_names)
        var = var_names{i};
        true_param_array = [true_param_array, [true_param.(var)]'];
    end

    point_estimate = round(mean(param_matrix)', 4);
    bias = round(mean(param_matrix - true_param_array)', 4);
    rmse = round(sqrt(mean((param_matrix - true_param_array) .^ 2, 1))', 4);

    result_table = table(string(param_label)', bias, rmse, ...
        'VariableNames', {'Variable', 'Bias', 'Rmse'});
end
