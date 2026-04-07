function initialize_rng()
    global SEED;

    if isempty(SEED)
        SEED = 12345;
    end

    rng(SEED, 'twister');
end
