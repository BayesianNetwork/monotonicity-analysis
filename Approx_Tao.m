function Tao = Approx_Tao(Skew)
%% 该函数用来逼近输入或输出方程里面潜变量的均值
%% 函数输入：
%% Skew：      潜变量的估计值：样本数 * 变量数
%% 函数输出：
%% Tao：       潜变量均值的期望

Num_Var = size(Skew, 2);
Mean_Skew = mean(Skew);
Tao = zeros(Num_Var, 1);

for num = 1 : Num_Var
    Mu = Mean_Skew(num);
    Diff = [];
    for tao = 0.001 : 0.001 : 10
        ValueL = (Mu - tao) * normcdf(tao, 0, 1);
        ValueR = normpdf(tao, 0, 1);
        Diff = [Diff; abs(ValueL - ValueR)];
    end
    Tao(num, 1) = find(Diff == min(Diff))/1000;
    min(Diff)
end
end



