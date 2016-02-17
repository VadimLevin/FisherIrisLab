function [A_Par, Centers_Par, B_Par] = KohonenMap(RulesCount, FactorDim, Rules, IrisFactors, Alpha_W, Alpha_R, MaxEpoch, Eps, H_max, InitialConditions)
% RulesCount - количество классов цветов (3)
% FactorDim - количество признаков класса (4)
% Rules - классы i строка соответствует i строке в IrisFactors
% IrisFactors - признаки
% Alpha_W - коэфициент сдвига нейрона победителя
% Alpha_R - коэфициент сдвига нейрона-ближайшего конкурента
% MaxEpoch - максимальное количество эпох
% Eps - требуемая точность
% H_max - максимальное количество кластеров (3)
% InitialConditions - начальные центры кластеров если не даны, то
% используется алгоритм на формирование центров

if isempty(InitialConditions)
    oldClusterCenters = GenerateBeginClusters(FactorDim, H_max);
else
    oldClusterCenters = InitialConditions;
end
%Количество выигрышей i-го нейрона
neuronWins = ones(H_max, 1); %массив единиц
continueEducation = true;
epochNumber = 0;
alpha_w = Alpha_W;
alpha_r = Alpha_R;
while continueEducation
    previousEpochCenters = oldClusterCenters;
    epochNumber = epochNumber + 1;
    for i = 1:length(IrisFactors(:, 1))
        currentIris = IrisFactors(i, :);
        %вычисляем расстояние
        distance = DistFromCenter(currentIris, oldClusterCenters, neuronWins);
        %находим победителя ~ -игнорируем значение, нам важен кто выиграл
        [~, winner_argmin] =  min(distance);
        %"удаляем" этот элемент, чтобы найти соперника
        distance(winner_argmin) = 1000;
        %находим соперника
        [~, rival_argmin] = min(distance);
        %увеличиваем количество побед
        neuronWins(winner_argmin) = neuronWins(winner_argmin) + 1;
        %корректировка центров кластеров
        newClusterCenters = oldClusterCenters;
        newClusterCenters(winner_argmin,:) = newClusterCenters(winner_argmin,:) + alpha_w*(currentIris - newClusterCenters(winner_argmin, :));
        newClusterCenters(rival_argmin,:) = newClusterCenters(rival_argmin,:) - alpha_r*(currentIris - newClusterCenters(rival_argmin, :));
        %делаем расчеты для остальных цветков
        oldClusterCenters = newClusterCenters;
    end
    %уменьшаем коэфициенты сдвига
    alpha_w = alpha_w - alpha_w*(epochNumber/MaxEpoch);
    alpha_r = alpha_r - alpha_r*(epochNumber/MaxEpoch);
    %текущие центры
    currentEpochCenters = newClusterCenters;
    %подсчет точности
    currentPrec = CalcPrecision(H_max, currentEpochCenters, previousEpochCenters);
    if (epochNumber > MaxEpoch) || (currentPrec < Eps)
        continueEducation = false;
    end
end
%удаляем избыточные кластеры
for i=1:length(currentEpochCenters(:,1))
    for j = 1:length(currentEpochCenters(1,:))
        if currentEpochCenters(i, j) < 0 || currentEpochCenters(i, j) > 1
            currentEpochCenters(i,:) = [];
            break;
        end
    end
end
K = length(currentEpochCenters(:,1));
%display(K); %сколько кластеров осталось
%считаем параметры для функции эмпирического риска
Centers_Par = currentEpochCenters; %Kx4 размерность
A_Par = zeros(size(Centers_Par));
B_Par = zeros(K,1);
for i = 1:K
    neighbors = zeros(K-1, 4); %соседи
    n = 1;
    for j = 1:K
        if j ~= i
            neighbors(n,:)= Centers_Par(j,:);
            n = n+1;
        end
    end;
    %В соседях центры всех кроме i-го кластера
    %ищем расстояния от iго до соседей
    distances = SquareDist(Centers_Par(i,:), neighbors);
    %Находим ближайший
    [nearest_dist] =  min(distances);
    %Заполняем строку расстоянием до ближайшего, деленное на r
    A_Par(i,:) = nearest_dist/1.5; 
end
% Rules - классы i строка соответствует i строке в IrisFactors Nx1
% IrisFactors - признаки Nx4
for i=1:K
    %prod(A) - произведение всех элементов массива А
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
%Функция принадлежности мю
function [answer] = MuGauss(X, C, A)
    answer = exp(-(X-C).^2./(2*A));
end

