function [A_Par, Centers_Par, B_Par] = KohonenMap(RulesCount, FactorDim, Rules, IrisFactors, Alpha_W, Alpha_R, MaxEpoch, Eps, H_max, InitialConditions)
% RulesCount - ���������� ������� ������ (3)
% FactorDim - ���������� ��������� ������ (4)
% Rules - ������ i ������ ������������� i ������ � IrisFactors
% IrisFactors - ��������
% Alpha_W - ���������� ������ ������� ����������
% Alpha_R - ���������� ������ �������-���������� ����������
% MaxEpoch - ������������ ���������� ����
% Eps - ��������� ��������
% H_max - ������������ ���������� ��������� (3)
% InitialConditions - ��������� ������ ��������� ���� �� ����, ��
% ������������ �������� �� ������������ �������

if isempty(InitialConditions)
    oldClusterCenters = GenerateBeginClusters(FactorDim, H_max);
else
    oldClusterCenters = InitialConditions;
end
%���������� ��������� i-�� �������
neuronWins = ones(H_max, 1); %������ ������
continueEducation = true;
epochNumber = 0;
alpha_w = Alpha_W;
alpha_r = Alpha_R;
while continueEducation
    previousEpochCenters = oldClusterCenters;
    epochNumber = epochNumber + 1;
    for i = 1:length(IrisFactors(:, 1))
        currentIris = IrisFactors(i, :);
        %��������� ����������
        distance = DistFromCenter(currentIris, oldClusterCenters, neuronWins);
        %������� ���������� ~ -���������� ��������, ��� ����� ��� �������
        [~, winner_argmin] =  min(distance);
        %"�������" ���� �������, ����� ����� ���������
        distance(winner_argmin) = 1000;
        %������� ���������
        [~, rival_argmin] = min(distance);
        %����������� ���������� �����
        neuronWins(winner_argmin) = neuronWins(winner_argmin) + 1;
        %������������� ������� ���������
        newClusterCenters = oldClusterCenters;
        newClusterCenters(winner_argmin,:) = newClusterCenters(winner_argmin,:) + alpha_w*(currentIris - newClusterCenters(winner_argmin, :));
        newClusterCenters(rival_argmin,:) = newClusterCenters(rival_argmin,:) - alpha_r*(currentIris - newClusterCenters(rival_argmin, :));
        %������ ������� ��� ��������� �������
        oldClusterCenters = newClusterCenters;
    end
    %��������� ����������� ������
    alpha_w = alpha_w - alpha_w*(epochNumber/MaxEpoch);
    alpha_r = alpha_r - alpha_r*(epochNumber/MaxEpoch);
    %������� ������
    currentEpochCenters = newClusterCenters;
    %������� ��������
    currentPrec = CalcPrecision(H_max, currentEpochCenters, previousEpochCenters);
    if (epochNumber > MaxEpoch) || (currentPrec < Eps)
        continueEducation = false;
    end
end
%������� ���������� ��������
for i=1:length(currentEpochCenters(:,1))
    for j = 1:length(currentEpochCenters(1,:))
        if currentEpochCenters(i, j) < 0 || currentEpochCenters(i, j) > 1
            currentEpochCenters(i,:) = [];
            break;
        end
    end
end
K = length(currentEpochCenters(:,1));
%display(K); %������� ��������� ��������
%������� ��������� ��� ������� ������������� �����
Centers_Par = currentEpochCenters; %Kx4 �����������
A_Par = zeros(size(Centers_Par));
B_Par = zeros(K,1);
for i = 1:K
    neighbors = zeros(K-1, 4); %������
    n = 1;
    for j = 1:K
        if j ~= i
            neighbors(n,:)= Centers_Par(j,:);
            n = n+1;
        end
    end;
    %� ������� ������ ���� ����� i-�� ��������
    %���� ���������� �� i�� �� �������
    distances = SquareDist(Centers_Par(i,:), neighbors);
    %������� ���������
    [nearest_dist] =  min(distances);
    %��������� ������ ����������� �� ����������, �������� �� r
    A_Par(i,:) = nearest_dist/1.5; 
end
% Rules - ������ i ������ ������������� i ������ � IrisFactors Nx1
% IrisFactors - �������� Nx4
for i=1:K
    %prod(A) - ������������ ���� ��������� ������� �
    Alphas = zeros(length(Rules),1);
    for j = 1:length(Rules)
        Alphas(j) = prod(MuGauss(IrisFactors(j,:), Centers_Par(i,:), A_Par(i,:)));
    end
    Alphas_dot_y = Alphas.*Rules;
    B_Par(i) = sum(Alphas_dot_y)/sum(Alphas);
end

end


function [Centers] = GenerateBeginClusters(FactorDim, H_max)
Centers = rand(H_max, FactorDim);
end

function [metric] = SquareDist(x, c)
metric = zeros(length(c(:,1)),1);
for k = 1:length(metric)
    metric(k) = (x-c(k, :))*(x-c(k, :))';
end
end

function [dist] = DistFromCenter(currentIris, centers, neuronWins)
winsSum = sum(neuronWins);
metric = SquareDist(currentIris, centers);
dist = zeros(length(centers(:, 1)), 1);
for i = 1:length(dist)
    dist(i) = (neuronWins(i)/winsSum)*(metric(i));
end;
end

function [prec] = CalcPrecision(H_max, newClusterCenters, oldClusterCenters)
euclideNorm = zeros(H_max,1);
for i = 1:H_max
    euclideNorm(i) = (newClusterCenters(i, :) - oldClusterCenters(i, :))*(newClusterCenters(i, :) - oldClusterCenters(i, :))';
end
prec = sum(euclideNorm);
prec = prec/H_max;
end
%������� �������������� ��
function [answer] = MuGauss(X, C, A)
    answer = exp(-(X-C).^2./(2*A));
end

