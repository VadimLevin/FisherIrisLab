function [normalizedData] = Preprocessor(fileName)
%====��������� ��� �������
    inputData = readtable(fileName,...
    'Delimiter',',','ReadVariableNames',false)
%==== ������� �������� ������, ������ ��� ������ ������ ���� - double
    irisMatrix = zeros(height(inputData), width(inputData));
    for i = 1:length(irisMatrix(:,1))
        irisMatrix(i, 1) = inputData{i,1};
        irisMatrix(i, 2) = inputData{i,2};
        irisMatrix(i, 3) = inputData{i,3};
        irisMatrix(i, 4) = inputData{i,4};
        irisName = inputData{i, 5};
        if strcmp(irisName{:}, 'Iris-setosa')
            irisMatrix(i,5) = 0.5;
        end;
        if strcmp(irisName{:}, 'Iris-versicolor')
            irisMatrix(i,5) = 1.5;
        end;
        if strcmp(irisName{:}, 'Iris-virginica')
            irisMatrix(i,5) = 2.5;
        end;
    end;
%=====������������ ������� ������
MinMax = []; % N x 2: 1 [i,1] - min_i, [i,2] - max_i
% ====��� ������ ���������� � ������� irisMatrix ������� ������� � ��������
for i=1:(length(irisMatrix(1, :))-1)
    MinMax=[MinMax; [min(irisMatrix(:, i)), max(irisMatrix(:, i))];];
end;
%====���������� ������� �������� ����� ���������� �������
normalizedData = irisMatrix;
for i = 1:(length(irisMatrix(1, :))-1)
    for j = 1:length(irisMatrix(:, 1))
        normalizedData(j, i) = (normalizedData(j, i) - MinMax(i, 1))/(MinMax(i, 2)-MinMax(i, 1));
    end;
end;
end