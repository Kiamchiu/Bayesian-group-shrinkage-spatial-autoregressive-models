function rmse = rmse(exp, real)
    assert(length(exp) == length(real));

    N = length(exp);

    rmse = sqrt(sum((exp - real) .^ 2) / N);
end
