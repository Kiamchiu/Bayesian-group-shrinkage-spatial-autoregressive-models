function [tpr, fpr, acc, selected_vars] = evaluate_variable_selection_sig(pm_totalsamples, active_set)
    pm_samples(:, 1:8) = pm_totalsamples(:, 2:9);

    [~, p] = size(pm_samples);

    selected = zeros(p, 1);

    for i = 1:p
        bound_np = hpdi(pm_samples(:, i), 0.50); % 95 % credible interval
        pspro_np = ((bound_np(1) > 0) || (bound_np(2) < 0));
        selected(i, 1) = pspro_np;
    end

    selected_vars = find(selected);

    true_active = ismember(1:p, active_set)';

    TP = sum(selected & true_active);
    FP = sum(selected & ~true_active);
    FN = sum(~selected & true_active);
    TN = sum(~selected & ~true_active);

    if (TP + FN) > 0
        tpr = TP / (TP + FN);
    else
        tpr = 0;
    end

    if (FP + TN) > 0
        fpr = FP / (FP + TN);
    else
        fpr = 0;
    end

    if (TP + FP + TN + FN) > 0
        acc = (TP + TN) / (TP + FP + TN + FN);
    else
        acc = 0;
    end

end
