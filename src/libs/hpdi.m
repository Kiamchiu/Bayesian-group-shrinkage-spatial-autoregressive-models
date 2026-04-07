function hdiLim = hpdi(sampleVec, credMass)

    % Determine number of elements that belong to HDI
    sortedVec = sort(sampleVec);
    ciIdx = ceil(credMass * length(sortedVec));
    nCIs = length(sortedVec) - ciIdx; % number of vector elements that make HDI

    % Determine middle of HDI to get upper and lower bound
    ciWidth = zeros(nCIs, 1);

    for ind = 1:nCIs
        ciWidth(ind) = sortedVec(ind + ciIdx) - sortedVec(ind);
    end

    [~, idxMin] = min(ciWidth);
    HDImin = sortedVec(idxMin);
    HDImax = sortedVec(idxMin + ciIdx);
    hdiLim = [HDImin, HDImax];

end
