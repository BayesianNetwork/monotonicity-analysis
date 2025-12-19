function model = ppls_rand(X, Y, nComp, maxIter, tol)

    Xmean = mean(X);
    Ymean = mean(Y);
    Xc = X - Xmean;
    Yc = Y - Ymean;

    [N, p] = size(Xc);
    [~, q] = size(Yc);

    Wx = randn(p, nComp);
    Wy = randn(q, nComp);
    Wx = orth(Wx);
    Wy = orth(Wy);

    sigmaX = eye(p);
    sigmaY = eye(q);

    loglik_old = -inf;

    for iter = 1:maxIter

        Cx = Wx * Wx' + sigmaX;
        Cy = Wy * Wy' + sigmaY;

        T = (Xc / Cx) * Wx + (Yc / Cy) * Wy;

        Wx = (Xc' * T) / (T' * T);
        Wy = (Yc' * T) / (T' * T);

        Wx = orth(Wx);
        Wy = orth(Wy);

        Ex = Xc - T * Wx';
        Ey = Yc - T * Wy';
        sigmaX = (Ex' * Ex) / N;
        sigmaY = (Ey' * Ey) / N;

        loglik = -0.5 * ( ...
        sum(sum((Ex / sigmaX) .* Ex)) + ...   % tr(Ex Σ^{-1} Ex')
        sum(sum((Ey / sigmaY) .* Ey)) + ...   % tr(Ey Σ^{-1} Ey')
        N * (log(det(sigmaX)) + log(det(sigmaY))) ...
        );

        if abs(loglik - loglik_old) < tol
            break;
        end
        loglik_old = loglik;
    end

    Tx = T;
    Ty = Yc / (Wy');

    B = Wx * Wy';

    model.Wx = Wx;
    model.Wy = Wy;
    model.Tx = Tx;
    model.Ty = Ty;
    model.B = B;
    model.Xmean = Xmean;
    model.Ymean = Ymean;

end
