function [U_Coeff, V_Coeff, Y_Sigma, Y_Tao, X_Sigma, X_Tao, Y_Error, X_Error, S_Mu, S_Sig] = Pi_QPR(YTrain, XTrain, Num_Ret)

Iota = [NaN; NaN; NaN];
Rou = [0; NaN];
XIdx_Res = find(1 - isnan(Iota));                                          
XIdx_Unr = find(isnan(Iota));                                              
YIdx_Res = find(1 - isnan(Rou));                                           
YIdx_Unr = find(isnan(Rou));                                               

[Num_Sam, Num_YVar] = size(YTrain);                                        
Num_XVar = size(XTrain, 2);                                                
Num_Ter = 10;                                                              
Y_Delta = 0.70 * ones(length(YIdx_Res), 1);                                
X_Delta = 0.70 * ones(length(XIdx_Res), 1);                                

Y_Sigma = ones(length(YIdx_Res), 1);                                       
X_Sigma = ones(length(XIdx_Res), 1);                                       
Z_Skew = zeros(Num_Sam, length(YIdx_Res));                                 
S_Skew = zeros(Num_Sam, length(XIdx_Res));                                 

X_Mu = mean(XTrain(:, XIdx_Res));                                          
N_Matrix = diag(std(XTrain(:, XIdx_Res)));                                 

YData = (YTrain - mean(YTrain))./std(YTrain);                              
XData = (XTrain - mean(XTrain))./std(XTrain);                              

[X_Proj, ~, X_Comp, ~, ~, ~, ~, ~] = plsregress(XData, YData, Num_Ret);    
H_Proj = zeros(Num_YVar, Num_Ret);                                        
for time = 1 : Num_Ret
    x_Time = X_Comp(:, time);                                              
    h_Time = (YData' * x_Time)/(x_Time' * x_Time);                         
    H_Proj(:, time) = h_Time;                                              
end

U_Coeff = H_Proj(:, 1 : Num_Ret);                                          
V_Coeff = X_Proj(:, 1 : Num_Ret);                                          
Y_Resi = YData - YData * (U_Coeff * U_Coeff');                             
X_Resi = XData - XData * (V_Coeff * V_Coeff');                             
Y_Tao_Full = (abs(mean(Y_Resi)))';                                        
Y_Tao = Y_Tao_Full(YIdx_Res);                                              
X_Tao_Full = (abs(mean(X_Resi)))';                                         
X_Tao = X_Tao_Full(XIdx_Res);                                             
Y_Error = diag(var(Y_Resi));                                               
X_Error = diag(var(X_Resi));                                               

for time= 1 : Num_Sam
    Z_Skew(time, :) = Y_Tao;                                              
    S_Skew(time, :) = X_Tao;                                              
end

for EM = 1 : Num_Ter


    S_Sig = zeros(Num_Ret, Num_Ret, Num_Sam);
    S_Mu = zeros(Num_Ret, Num_Sam);
    V_Res = V_Coeff(XIdx_Res, :);

    S_Sig4 = zeros(Num_Ret, Num_Ret);
    for num = 1 : length(XIdx_Res)
        Deno = (X_Sigma(num))^2 * (1 - (X_Delta(num))^2);
        N_Vec = N_Matrix(num, :);
        S_Sig4 = S_Sig4 + (V_Res' * (N_Vec' * N_Vec) * V_Res) / Deno;
    end
    S_Sig_First = pinv(eye(Num_Ret) + U_Coeff' * pinv(Y_Error) * U_Coeff + V_Coeff' * pinv(X_Error) * V_Coeff + S_Sig4);
    S_Sig(:, :, 1) = S_Sig_First;
    
    S_Mu3 = zeros(Num_Ret, 1);
    for num = 1 : length(XIdx_Res)
        Deno = (X_Sigma(num))^2 * (1 - (X_Delta(num))^2);
        N_Vec = N_Matrix(num, :);
        S_Mu3 = S_Mu3 + V_Res' * N_Vec' * (Iota(XIdx_Res(num)) - X_Mu(num) - X_Sigma(num) * X_Delta(num) * S_Skew(time, num)) / Deno;
    end   
    Y_Vec = (YData(1, :))';
    X_Vec = (XData(1, :))';
    S_Mu(:, 1) = S_Sig_First * (U_Coeff' * pinv(Y_Error) * Y_Vec + V_Coeff' * pinv(X_Error) * X_Vec + S_Mu3);

    for time = 2 : Num_Sam
        S_Sig4 = zeros(Num_Ret, Num_Ret);
        for num = 1 : length(XIdx_Res)
            Deno = (X_Sigma(num))^2 * (1 - (X_Delta(num))^2);
            N_Vec = N_Matrix(num, :);
            S_Sig4 = S_Sig4 + (V_Res' * (N_Vec' * N_Vec) * V_Res) / Deno;
        end
        S_Sig5 = zeros(Num_Ret, Num_Ret);
        for num = 1 : length(YIdx_Res)
            U_Vec = U_Coeff(YIdx_Res(num), :);
            Deno = (Y_Sigma(num))^2 * (1 - (Y_Delta(num))^2);
            S_Sig5 = S_Sig5 + U_Vec' * U_Vec / Deno;
        end
        S_Sig_Time = pinv(eye(Num_Ret) + U_Coeff' * pinv(Y_Error) * U_Coeff + V_Coeff' * pinv(X_Error) * V_Coeff + S_Sig4 + S_Sig5);
        S_Sig(:, :, time) = S_Sig_Time;

        S_Mu2 = zeros(Num_Ret, 1);
        for num = 1 : length(XIdx_Res)
            Deno = (X_Sigma(num))^2 * (1 - (X_Delta(num))^2);
            N_Vec = N_Matrix(num, :);
            S_Mu2 = S_Mu2 + V_Res' * N_Vec' * (Iota(XIdx_Res(num)) - X_Mu(num) - X_Sigma(num) * X_Delta(num) * S_Skew(time, num)) / Deno;
        end

        S_Mu4 = zeros(Num_Ret, 1);
        for num = 1 : length(YIdx_Res)
            U_Vec = U_Coeff(YIdx_Res(num), :);
            Deno = (Y_Sigma(num))^2 * (1 - (Y_Delta(num))^2);
            S_Pre = S_Mu(:, time - 1);
            S_Mu4 = S_Mu4 + U_Vec' * (Rou(YIdx_Res(num)) + U_Vec * S_Pre + Y_Sigma(num) * Y_Delta(num) * Z_Skew(time, num)) / Deno;  %%
        end

        Y_Vec = (YData(time, :))';
        X_Vec = (XData(time, :))';
        S_Mu(:, time) = S_Sig_Time * (V_Coeff' * pinv(X_Error) * X_Vec + S_Mu2 + U_Coeff' * pinv(Y_Error) * Y_Vec + S_Mu4);
    end
        
    for time = 1 : Num_Sam
        for num = 1 : length(XIdx_Res)
            V_Vec = V_Coeff(XIdx_Res(num), :);
            S_Vec = S_Mu(:, time);
            S_Skew_Pre = (Iota(XIdx_Res(num)) - N_Matrix(num, num) * V_Vec * S_Vec - X_Mu(num)) * X_Delta(num) / X_Sigma(num);
            S_Skew(time, num) = S_Skew_Pre + (1 - (X_Delta(num))^2) * X_Tao(num);
        end
    end

    for time = 2 : Num_Sam
        for num = 1 : length(YIdx_Res)
            U_Vec = U_Coeff(YIdx_Res(num), :);
            S_Diff = S_Mu(:, time) - S_Mu(:, time - 1);
            Z_Skew_Pre = (U_Vec * S_Diff - Rou(YIdx_Res(num))) * Y_Delta(num) / Y_Sigma(num);  %%
            Z_Skew(time, num) = Z_Skew_Pre + (1 - (Y_Delta(num)^2)) * Y_Tao(num);
        end
    end

    for num = 1 : length(XIdx_Res)
        Deno = (X_Sigma(num))^2 * (1 - (X_Delta(num))^2);
        V_Pre = 1 / X_Error(XIdx_Res(num), XIdx_Res(num)) + (N_Matrix(num, num))^2 / Deno;

        V_Mid = zeros(1, Num_Ret);
        for time = 1 : Num_Sam
            S_Vec = S_Mu(:, time);
            X_Ele = XData(time, XIdx_Res(num));
            V_Cos = N_Matrix(num, num) * (Iota(XIdx_Res(num)) - X_Mu(num) - X_Sigma(num) * X_Delta(num) * S_Skew(time, num)) * S_Vec';
            V_Mid = V_Mid + X_Ele * S_Vec' / X_Error(XIdx_Res(num), XIdx_Res(num)) + V_Cos / Deno;
        end

        V_Pos = zeros(Num_Ret, Num_Ret);
        for time = 1 : Num_Sam
            S_Vec = S_Mu(:, time);
            V_Pos = V_Pos + S_Vec * S_Vec';
        end
        V_Coeff(XIdx_Res(num), :) = 1 / V_Pre * V_Mid * pinv(V_Pos);
    end

    for num = 1 : length(XIdx_Unr)
        V_Pre = zeros(1, Num_Ret);
        V_Pos = zeros(Num_Ret, Num_Ret);
        for time = 1 : Num_Sam
            X_Ele = XData(time, XIdx_Unr(num));
            S_Vec = S_Mu(:, time);
            V_Pre = V_Pre + X_Ele * S_Vec';
            V_Pos = V_Pos + S_Vec * S_Vec';
        end
        V_Coeff(XIdx_Unr(num), :) = V_Pre * pinv(V_Pos);
    end

    for num = 1 : Num_XVar
        X_Error(num, num) = 0;
        V_Vec = V_Coeff(num, :);
        for time = 1 : Num_Sam
            X_Ele = XData(time, num);
            S_Vec = S_Mu(:, time);
            X_Error(num, num) = X_Error(num, num) + (X_Ele - V_Vec * S_Vec)^2 / Num_Sam;
        end
    end

    for num = 1 : length(YIdx_Res)
        Deno = (Y_Sigma(num))^2 * (1 - (Y_Delta(num))^2);
        U_Pre = zeros(1, Num_Ret);
        U_Pos = zeros(Num_Ret, Num_Ret);
        for time = 2 : Num_Sam
            Y_Ele = YData(time, YIdx_Res(num));
            Z_Vec = S_Mu(:, time);
            Z_Diff = S_Mu(:, time) - S_Mu(:, time - 1);
            U_Pre = U_Pre + Y_Ele * Z_Vec' / Y_Error(YIdx_Res(num), YIdx_Res(num)) + (Rou(YIdx_Res(num)) + Y_Sigma(num) * Y_Delta(num) * Z_Skew(time, num)) * Z_Diff' / Deno;  %%
            U_Pos = U_Pos + Z_Vec * Z_Vec' / Y_Error(YIdx_Res(num), YIdx_Res(num)) + Z_Diff * Z_Diff' / Deno;
        end
        U_Coeff(YIdx_Res(num), :) = U_Pre * pinv(U_Pos);
    end

    for num = 1 : length(YIdx_Unr)
        Y_Ele = YData(time, YIdx_Unr(num));
        Z_Vec = S_Mu(:, time);
        U_Pre = zeros(1, Num_Ret);
        U_Pos = zeros(Num_Ret, Num_Ret);
        for time = 2 : Num_Sam
            U_Pre = U_Pre + Y_Ele * Z_Vec';
            U_Pos = U_Pos + Z_Vec * Z_Vec';
        end
        U_Coeff(YIdx_Unr(num), :) = U_Pre * pinv(U_Pos);
    end

    Y_Error = zeros(Num_YVar, Num_YVar);
    for num = 1 : Num_YVar
        U_Vec = U_Coeff(num, :);
        for time = 1 : Num_Sam
            Y_Ele = YData(time, num);
            Z_Vec = S_Mu(:, time);
            Y_Error(num, num) = Y_Error(num, num) + (Y_Ele - U_Vec * Z_Vec)^2 / Num_Sam;
        end
    end

    Y_Tao = Approx_Tao(Z_Skew);
    X_Tao = Approx_Tao(S_Skew);

    for num = 1 : length(XIdx_Res)
        Alpha0 = Num_Sam * (1 - (X_Delta(num))^2);
        V_Vec = V_Coeff(XIdx_Res(num), :);
        Alpha1 = 0;
        for time = 1 : Num_Sam
            S_Vec = S_Mu(:, time);
            S_Ele = S_Skew(time, num);
            Alpha1 = Alpha1 + (Iota(XIdx_Res(num)) - N_Matrix(num, num) * V_Vec * S_Vec - X_Mu(num)) * X_Delta(num) * S_Ele;
        end
        Alpha2 = 0;
        for time = 1 : Num_Sam
            S_Vec = S_Mu(:, time);
            Alpha2 = Alpha2 - (Iota(XIdx_Res(num)) - N_Matrix(num, num) * V_Vec * S_Vec - X_Mu(num))^2;
        end
        Alpha = [Alpha0, Alpha1, Alpha2];
        Result = roots(Alpha);
        X_Sigma(num, 1) = max(Result);
    end

    for num = 1 : length(YIdx_Res)
        Beta0 = Num_Sam * (1 - (Y_Delta(num))^2);
        Beta1 = 0;
        for time = 2 : Num_Sam
            U_Vec = U_Coeff(YIdx_Res(num), :);
            Z_Diff = S_Mu(:, time) - S_Mu(:, time - 1);
            Z_Ele = Z_Skew(time, num);
            Beta1 = Beta1 + (Rou(YIdx_Res(num)) - U_Vec * Z_Diff) * Y_Delta(num) * Z_Ele;   %%
        end
        Beta2 = 0;
        for time = 2 : Num_Sam
            U_Vec = U_Coeff(num, :);
            Z_Diff = S_Mu(:, time) - S_Mu(:, time - 1);
            Beta2 = Beta2 - (Rou(YIdx_Res(num)) - U_Vec * Z_Diff)^2;
        end
        Beta = [Beta0, Beta1, Beta2];
        Result = roots(Beta);
        Y_Sigma(num, 1) = max(Result);
    end
end



