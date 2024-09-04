
clc, clear, close all , warning off all

%Data = importdata('Inputj.mat');
%%A = A(1:5,:);
Data = randint(70000,3,[-225 225]);
Data = double(Data);
niteration = 10;
L = TasomNeuralNetwok2(Data,niteration);
