function sig = sig(param_array, sig)
    bounds = hpdi(param_array, sig);
    lower = bounds(1);
    upper = bounds(2);

    if (lower < 0) && (upper > 0)
        sig = 0;
    else
        sig = 1;
    end

end
