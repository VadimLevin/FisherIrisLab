function [bestGlobalPos, CurrentEps] = ParticalSwarmOptimization(RulesCount, FactorDim, InitCount, InertionCoef, MemoryCoef, CoopCoef, Eps, MaxIter, InitialConditions, ModelParameters, IrisRules, IrisFactors)
% RulesCount - ���������� �������
% FactorDim - ���������� ��������� ������
% InitCount - ���������� ������ � ���
% InertionCoef - ���������� ������� ������ (W)
% MemoryCoef - ���������� ������ (C1)
% CoopCoef - ������ �������������� (�2)
% Eps - ����������� �������� �������
% MaxIter - ������������ ���������� ��������
% InitialConditions - ��������� ������������� ������ ���� ������, ���� ���
% ModelParameters - ��������� ������ W=<a, c, b> ��� 1�� �������
% IrisRules - ������ i ������ ������������� i ������ � IrisFactors
% IrisFactors - ��������
TaskDimension = RulesCount*(2*FactorDim+1);% 3*(2*4+1) = 27

oldSwarmPosition = zeros(InitCount, TaskDimension);%���������� ��������� ������ � ���
newSwarmPosition = zeros(InitCount, TaskDimension);%������� ��������� ������ � ���
%������ ��������� �������������
if isempty(InitialConditions)
    %���� ������� ��� W=<a, c, b>
    oldSwarmPosition(1,:) = ModelParameters;
    %������ ��������, ������ ���������� �� 0 �� 1.
    oldSwarmPosition(2:InitCount,:) = rand(InitCount-1, TaskDimension);
else
    oldSwarmPosition = InitialConditions;
end;
%��������� �������� �������
oldV = zeros(InitCount, TaskDimension);
%������ ��������� ������ ������� 
bestIndividPos = oldSwarmPosition;
%������ ��������� ���
bestGlobalPos = FindBestGlobalPosition(InitCount, oldSwarmPosition, IrisFactors, IrisRules);
[A_par, C_par, B_par] = DecodeParameters(bestGlobalPos);
CurrentEps = Verification(IrisFactors, IrisRules, A_par, C_par, B_par);

i = 0;

while i <= MaxIter && CurrentEps > Eps
    for j = 1:InitCount
        %��������� ��������
        newV(j,:) = CalculateVelocity(oldV(j,:), oldSwarmPosition(j, :), bestIndividPos(j,:), bestGlobalPos, InertionCoef, MemoryCoef, CoopCoef);
        %��������� ���������
        newSwarmPosition(j,:) = oldSwarmPosition(j, :) + newV(j,:);
        %����� �� �������� ��������� ��������� �������?
        [bestPos, whichBetter] = ChooseBestPosition(IrisFactors, IrisRules, newSwarmPosition(j,:), bestIndividPos(j,:));
        if whichBetter == 1
            bestIndividPos(j,:) = bestPos;
            %����� �� �������� ��������� ��������� ���?
            [bestPos, whichGlobal] = ChooseBestPosition(IrisFactors, IrisRules, bestIndividPos(j,:), bestGlobalPos);
            if whichGlobal == 1
                bestGlobalPos = bestPos;
                %������� �������� � ����� ������ ����������
                [A_par, C_par, B_par] = DecodeParameters(bestGlobalPos);
                CurrentEps = Verification(IrisFactors, IrisRules, A_par, C_par, B_par);
                
            end
        end
    end
    oldSwarmPosition = newSwarmPosition;
    oldV = newV;
   
    i=i+1;
end;
end
function [newVelocity] = CalculateVelocity(currentVel, currentPos, BestIP, BestGP, InertionCoef, MemoryCoef, CoopCoef)
% currentVel - ������� �������� ������� (1xd), N-���������� ������, d-����������� ������������
% currentPos - ������� ������� ������� (1xd) 
% BestIP - ������� ������ ��������� ��� ������� (1xd)
% BestGP - ������� ������ ���������� ��������� (1�d)
    newVelocity = InertionCoef*currentVel;
    newVelocity = newVelocity + MemoryCoef*rand(size(currentVel)).*(BestIP - currentPos);
    newVelocity = newVelocity + CoopCoef*rand(size(currentVel)).*(BestGP - currentPos);
end

%������� �������������� ��
function [answer] = MuGauss(X, C, A)
    answer = exp(-(X-C).^2./(2*A));
end
%���������� ������������� �����
function [answer] = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par)
% IrisFactors - �������������� ������
% IrisRules - ��� ����� i ������ ������������� i-�� ������ � IrisFactors
% A_par - A_ik
% C_par - ������
% B_par - B_0k
K = 3;
Norm = zeros(length(IrisRules),1);
for j = 1:length(IrisRules)
    Alphas = zeros(K,1);
    for i=1:K
        %prod(A) - ������������ ���� ��������� ������� �
        Alphas(i) = prod(MuGauss(IrisFactors(j,:), C_par(i,:), A_par(i,:)));
    end
    Alphas_dot_B = Alphas.*B_par;
    response = sum(Alphas_dot_B)/sum(Alphas);
    Norm(j) = (response - IrisRules(j))^2;
end
answer = sum(Norm)/length(Norm);
end

function [A_par, C_par, B_par] = DecodeParameters(currentPartPos)
    A_par = zeros(3, 4);
    C_par = zeros(3, 4);
    B_par = zeros(3, 1);
    A_par(1,:) = currentPartPos(1:4);
    A_par(2,:) = currentPartPos(5:8);
    A_par(3,:) = currentPartPos(9:12);
    C_par(1,:) = currentPartPos(13:16);
    C_par(2,:) = currentPartPos(17:20);
    C_par(3,:) = currentPartPos(21:24);
    
    B_par(1:3) = currentPartPos(25:27);
end

%����� ���������� ��������� ���
function [bestGlobalPos] = FindBestGlobalPosition(InitCount, SwarmPosition, IrisFactors, IrisRules)

currentPartPos = SwarmPosition(1,:);
%������������� ����������
[A_par, C_par, B_par] = DecodeParameters(currentPartPos);
%������� ������� ������������� ����� � �������� ��� �������� �������
%���������
bestGlobalValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);
bestGlobalPos = currentPartPos;
for i = 2:InitCount
    currentPartPos = SwarmPosition(i,:);
    %������������� ����������
    [A_par, C_par, B_par] = DecodeParameters(currentPartPos);
    %�������� ������� ������������� ����� ��� ���� �������
    currentPartValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);
    %���� �������� �����, �� ������� ��������� ���� ������� - ������
    %���������� ���
    if currentPartValue < bestGlobalValue
        bestGlobalPos = currentPartPos;
    end
end
end

%����� ��������� ������� �� ����
function [bestPos, type] = ChooseBestPosition(IrisFactors, IrisRules, firstPart, secondPart)
currentPartPos = firstPart;
%������������� ����������
[A_par, C_par, B_par] = DecodeParameters(currentPartPos);
firstValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);

currentPartPos = secondPart;
%������������� ����������
[A_par, C_par, B_par] = DecodeParameters(currentPartPos);
secondValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);

if firstValue < secondValue
    bestPos = firstPart;
    type = 1;
else
    bestPos = secondPart;
    type = 2;
end 
end

%�����������
function [answer] = Verification(IrisFactors, IrisRules, A_par, C_par, B_par)
% IrisFactors - �������������� ������
% IrisRules - ��� ����� i ������ ������������� i-�� ������ � IrisFactors
% A_par - A_ik
% C_par - ������
% B_par - B_0k
K = 3;
Norm = zeros(length(IrisRules),1);
response = zeros(length(IrisRules),1);
for j = 1:length(IrisRules)
    Alphas = zeros(K,1);
    for i=1:K
        %prod(A) - ������������ ���� ��������� ������� �
        Alphas(i) = prod(MuGauss(IrisFactors(j,:), C_par(i,:), A_par(i,:)));
    end
    Alphas_dot_B = Alphas.*B_par;
    response(j) = sum(Alphas_dot_B)/sum(Alphas);
end
CorrectAnswers = 0;
for i = 1:length(IrisRules)
    if abs(response(i) - IrisRules(i))< 0.5
        CorrectAnswers = CorrectAnswers + 1;
    end
end
answer = CorrectAnswers/length(IrisRules);
end

