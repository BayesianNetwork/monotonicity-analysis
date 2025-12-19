function model = ppls_plsinit(X, Y, nComp, maxIter, tol)

[N, p] = size(X);
[~, q] = size(Y);

Xmean = mean(X);
Ymean = mean(Y);
Xc = X - Xmean;
Yc = Y - Ymean;

[~,~,~,~,Wx_init,Wy_init] = plsregress(X, Y, nComp);
Wx = Wx_init(2:end,:);  
Wy = Wy_init;            

Wx = Wx + 0.01*randn(size(Wx));
Wy = Wy + 0.01*randn(size(Wy));

Wx = orth(Wx);  
Wy = orth(Wy);

sigmaX = eye(p);
sigmaY = eye(q);
loglik_old = -inf;


for iter = 1:maxIter
    % E-step
    Cx = Wx*Wx' + sigmaX;
    Cy = Wy*Wy' + sigmaY;
    T = (Xc / Cx) * Wx + (Yc / Cy) * Wy;

    % M-step
    Wx = (Xc'*T) / (T'*T);
    Wy = (Yc'*T) / (T'*T);
    Wx = orth(Wx);
    Wy = orth(Wy);

    Ex = Xc - T*Wx';
    Ey = Yc - T*Wy';
    sigmaX = (Ex'*Ex)/N;
    sigmaY = (Ey'*Ey)/N;

    loglik = -0.5 * ( sum(sum((Ex/sigmaX).*Ex)) + sum(sum((Ey/sigmaY).*Ey)) ...
                      + N*(log(det(sigmaX)) + log(det(sigmaY))) );
    if abs(loglik - loglik_old) < tol
        break;
    end
    loglik_old = loglik;
end

Tx = T;
Ty = Yc / (Wy');
B = Wx*Wy';

model.Wx = Wx;
model.Wy = Wy;
model.Tx = Tx;
model.Ty = Ty;
model.B = B;
model.Xmean = Xmean;
model.Ymean = Ymean;
model.loglik = loglik;

end
