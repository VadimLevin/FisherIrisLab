function Main
%Data - прономированный массив входных данных
[Data] = Preprocessor('irisData.txt');
%Количество циклов опытов
V = 5;
%Перемешаем входные данные для дальнейшей обработки
shuffledData = Data(randperm(size(Data,1)),:);
%Точность
Score = 0.0;
for i = 1:V
    %Cформируем 2 выборки тестовую и тренировочную
    %В тестовую войдет отрезок от i*(общее_количество/количество тестов) до
    %(i+1)*(общее_количество/количество тестов)
    TB_l = (i-1)*(length(shuffledData(:,1))/V) + 1;%границы отрезка левая
    TB_r = i*(length(shuffledData(:,1))/V);%правая
    TestSelection = [];
    TestSelection = shuffledData(TB_l:TB_r, :);
    %В обучающую выборку войдут все, кроме тех, которые вошли в тестувую
    TrainSelection = [];
    if TB_l > 1
        TrainSelection = shuffledData(1:TB_l-1, :);
    end
    if TB_r < length(shuffledData(:,1))
        TrainSelection = [TrainSelection; shuffledData(TB_r+1:length(shuffledData(:,1)),:)];
    end
    %Правила
    Rules = TrainSelection(:, length(TrainSelection(1,:))); %Последний столбец
    %Признаки(Характеристики 4шт)
    IrisFactors = TrainSelection(:, 1:(length(TrainSelection(1,:))-1)); %Все кроме последнего
    RulesCount = 3;%количество различных классов - 3 типа цветов
    FactorDim = 4;%у каждого 4 характеристики
    %Теперь запускаем метод Кохонена
    Alpha_W = 0.07;
    Alpha_R = 0.0007;
    MaxEpoch = 7;
    Eps = 10^(-7);
    H_max = 3;
    InitialConditions = [];
    
    [A_Par, C_Par, B_Par] = KohonenMap(RulesCount, FactorDim, Rules, IrisFactors, Alpha_W, Alpha_R, MaxEpoch, Eps, H_max, InitialConditions);
    %Кодируем в W=<a, c, b> для роя частиц
    TaskDimension = RulesCount*(2*FactorDim+1);
    W = zeros(TaskDimension,1);
    W(1:4) = A_Par(1);
    W(5:8) = A_Par(2);
    W(9:12) = A_Par(3);
    W(13:16) = C_Par(1);
    W(17:20) = C_Par(2);
    W(21:24) = C_Par(3);
    W(25:27) = B_Par;
    %Запускаем метод параметрической оптимизации - рой частиц
    InitCount = 20;
    InertionCoef = 0.7;
    MemoryCoef = 1.49445;
    CoopCoef = 1.49445;
    Eps = 10^(-3);
    MaxIter = 20;
    %bestGlobalPos - W после оптимизации, bestGlobalValue - значение
    %функции эмперического риска для bestGlobalPos
    [bestGlobalPos, bestGlobalValue] = ParticalSwarmOptimization(RulesCount, FactorDim, InitCount, InertionCoef, MemoryCoef, CoopCoef, Eps, MaxIter, [], W, Rules, IrisFactors);
    IrisRules = TestSelection(:, length(TestSelection(1,:))); %Последний столбец
    %Признаки(Характеристики 4шт)
    IrisFactors = TestSelection(:, 1:(length(TestSelection(1,:))-1)); %Все кроме последнего
    [A_par, C_par, B_par] = DecodeParameters(bestGlobalPos);
    TestPrec = Verification(IrisFactors, IrisRules, A_par, C_par, B_par);
    display(TestPrec);
    Score = Score+TestPrec;
    display(i);
end;
CrossValidationPrecision = Score/V;
display(CrossValidationPrecision);
end

%Функция принадлежности мю
function [answer] = MuGauss(X, C, A)
    answer = exp(-(X-C).^2./(2*A));
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