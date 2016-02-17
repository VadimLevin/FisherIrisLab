function [bestGlobalPos, CurrentEps] = ParticalSwarmOptimization(RulesCount, FactorDim, InitCount, InertionCoef, MemoryCoef, CoopCoef, Eps, MaxIter, InitialConditions, ModelParameters, IrisRules, IrisFactors)
% RulesCount - количество классов
% FactorDim - количество признаков класса
% InitCount - количество частиц в рое
% InertionCoef - коэфициент инерции частиц (W)
% MemoryCoef - коэфициент памяти (C1)
% CoopCoef - фактор сотрудничества (С2)
% Eps - необходимая точность решения
% MaxIter - максимальное количество итераций
% InitialConditions - Начальное распределение частиц либо пустое, либо нет
% ModelParameters - Параметры модели W=<a, c, b> для 1ой частицы
% IrisRules - классы i строка соответствует i строке в IrisFactors
% IrisFactors - признаки
TaskDimension = RulesCount*(2*FactorDim+1);% 3*(2*4+1) = 27

oldSwarmPosition = zeros(InitCount, TaskDimension);%предыдущее положение частиц в рое
newSwarmPosition = zeros(InitCount, TaskDimension);%текущее положение частиц в рое
%задаем начальное распределение
if isempty(InitialConditions)
    %одна частица это W=<a, c, b>
    oldSwarmPosition(1,:) = ModelParameters;
    %другие случайно, каждая координата от 0 до 1.
    oldSwarmPosition(2:InitCount,:) = rand(InitCount-1, TaskDimension);
else
    oldSwarmPosition = InitialConditions;
end;
%начальная скорость нулевая
oldV = zeros(InitCount, TaskDimension);
%лучшее положение каждой частицы 
bestIndividPos = oldSwarmPosition;
%лучшее положения роя
bestGlobalPos = FindBestGlobalPosition(InitCount, oldSwarmPosition, IrisFactors, IrisRules);
[A_par, C_par, B_par] = DecodeParameters(bestGlobalPos);
CurrentEps = Verification(IrisFactors, IrisRules, A_par, C_par, B_par);

i = 0;

while i <= MaxIter && CurrentEps > Eps
    for j = 1:InitCount
        %обновляем скорость
        newV(j,:) = CalculateVelocity(oldV(j,:), oldSwarmPosition(j, :), bestIndividPos(j,:), bestGlobalPos, InertionCoef, MemoryCoef, CoopCoef);
        %обновляем положение
        newSwarmPosition(j,:) = oldSwarmPosition(j, :) + newV(j,:);
        %Нужно ли обновить наилучшее положение частицы?
        [bestPos, whichBetter] = ChooseBestPosition(IrisFactors, IrisRules, newSwarmPosition(j,:), bestIndividPos(j,:));
        if whichBetter == 1
            bestIndividPos(j,:) = bestPos;
            %нужно ли обновить наилучшее положение роя?
            [bestPos, whichGlobal] = ChooseBestPosition(IrisFactors, IrisRules, bestIndividPos(j,:), bestGlobalPos);
            if whichGlobal == 1
                bestGlobalPos = bestPos;
                %Подсчет точности с новым лучшим положением
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
% currentVel - текущая скорость частицы (1xd), N-количество частиц, d-размерность пространства
% currentPos - текущая позиция частицы (1xd) 
% BestIP - текущее лучшее положение для частицы (1xd)
% BestGP - текущее лучшее глобальное положение (1хd)
    newVelocity = InertionCoef*currentVel;
    newVelocity = newVelocity + MemoryCoef*rand(size(currentVel)).*(BestIP - currentPos);
    newVelocity = newVelocity + CoopCoef*rand(size(currentVel)).*(BestGP - currentPos);
end

%Функция принадлежности мю
function [answer] = MuGauss(X, C, A)
    answer = exp(-(X-C).^2./(2*A));
end
%Функционал эмпирического риска
function [answer] = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par)
% IrisFactors - характеристики ирисов
% IrisRules - тип ириса i строка соответствует i-ой строке в IrisFactors
% A_par - A_ik
% C_par - Центры
% B_par - B_0k
K = 3;
Norm = zeros(length(IrisRules),1);
for j = 1:length(IrisRules)
    Alphas = zeros(K,1);
    for i=1:K
        %prod(A) - произведение всех элементов массива А
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

%поиск наилучшего положения роя
function [bestGlobalPos] = FindBestGlobalPosition(InitCount, SwarmPosition, IrisFactors, IrisRules)

currentPartPos = SwarmPosition(1,:);
%Декодирование параметров
[A_par, C_par, B_par] = DecodeParameters(currentPartPos);
%подсчет функции эмперического риска и значения для текущего лучшего
%положения
bestGlobalValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);
bestGlobalPos = currentPartPos;
for i = 2:InitCount
    currentPartPos = SwarmPosition(i,:);
    %Декодирование параметров
    [A_par, C_par, B_par] = DecodeParameters(currentPartPos);
    %значение функции эмперического риска для этой частицы
    currentPartValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);
    %если значение лучше, то считаем положение этой частицы - лучшим
    %положением роя
    if currentPartValue < bestGlobalValue
        bestGlobalPos = currentPartPos;
    end
end
end

%выбор наилучшей частицы из двух
function [bestPos, type] = ChooseBestPosition(IrisFactors, IrisRules, firstPart, secondPart)
currentPartPos = firstPart;
%Декодирование параметров
[A_par, C_par, B_par] = DecodeParameters(currentPartPos);
firstValue = EmpirikRiskFunction(IrisFactors, IrisRules, A_par, C_par, B_par);

currentPartPos = secondPart;
%Декодирование параметров
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

%Верификация
function [answer] = Verification(IrisFactors, IrisRules, A_par, C_par, B_par)
% IrisFactors - характеристики ирисов
% IrisRules - тип ириса i строка соответствует i-ой строке в IrisFactors
% A_par - A_ik
% C_par - Центры
% B_par - B_0k
K = 3;
Norm = zeros(length(IrisRules),1);
response = zeros(length(IrisRules),1);
for j = 1:length(IrisRules)
    Alphas = zeros(K,1);
    for i=1:K
        %prod(A) - произведение всех элементов массива А
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

