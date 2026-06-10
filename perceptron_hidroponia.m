clc;
clear;
close all;

%% LEITURA DOS DADOS

treino = readtable('Dados/hidroponia_treinamento.csv');
teste = readtable('Dados/hidroponia_teste.csv');

X_train = table2array(treino(:,1:5));
Y_train = table2array(treino(:,6));

X_test = table2array(teste(:,1:5));
Y_test = table2array(teste(:,6));

% Transforma classe 0 em -1
Y_train(Y_train==0) = -1;
Y_test(Y_test==0) = -1;

% NORMALIZAÇÃO

Xmin = min(X_train);
Xmax = max(X_train);

X_train = (X_train-Xmin)./(Xmax-Xmin);
X_test = (X_test-Xmin)./(Xmax-Xmin);

% ADICIONANDO BIAS

X_train = [ones(size(X_train,1),1) X_train];
X_test = [ones(size(X_test,1),1) X_test];

[n_amostras,n_atributos] = size(X_train);

%% PARÂMETROS DO PERCEPTRON

eta = 0.01;
epocas = 100;

w = rand(n_atributos,1)-0.5;

erro_epoca = zeros(epocas,1);
pesos_hist = zeros(epocas,n_atributos);

%% TREINAMENTO

for epoca = 1:epocas

    erro_total = 0;

    for i = 1:n_amostras

        x = X_train(i,:)';

        % Soma ponderada
        u = w'*x;

        % Função degrau
        if u >= 0
            y = 1;
        else
            y = -1;
        end

        % Erro
        erro = Y_train(i)-y;

        % Atualização dos pesos
        w = w + eta*erro*x;

        erro_total = erro_total + abs(erro);

    end

    erro_epoca(epoca) = erro_total;
    pesos_hist(epoca,:) = w';

    if erro_total == 0
        fprintf('\nConvergência alcançada na época %d\n',epoca);
        break
    end

end

%% TESTE

n_testes = size(X_test,1);

Y_pred = zeros(n_testes,1);

for i = 1:n_testes

    u = w'*X_test(i,:)';

    if u >= 0
        Y_pred(i) = 1;
    else
        Y_pred(i) = -1;
    end

end

%% MÉTRICAS

acertos = sum(Y_pred==Y_test);
erros = n_testes-acertos;

acuracia = 100*acertos/n_testes;

fprintf('\n=====================================\n');
fprintf('RESULTADOS DO PERCEPTRON\n');
fprintf('=====================================\n');

fprintf('Taxa de aprendizagem: %.3f\n',eta);
fprintf('Número de épocas: %d\n',epoca);

fprintf('\nNúmero de amostras de treinamento: %d\n',size(X_train,1));
fprintf('Número de amostras de teste: %d\n',n_testes);

fprintf('\nAcertos: %d\n',acertos);
fprintf('Erros: %d\n',erros);

fprintf('Taxa de erro: %.2f %%\n',100-acuracia);
fprintf('Acurácia = %.2f %%\n',acuracia);

fprintf('\nPesos finais:\n');
disp(w)

%% TABELA DE RESULTADOS

Resultado = table(Y_test,Y_pred,...
'VariableNames',{'Classe_Real','Classe_Prevista'});

disp(Resultado)

% GRÁFICO 1 - CONVERGÊNCIA

figure

plot(erro_epoca(1:epoca),'b','LineWidth',2)

xlabel('Épocas')
ylabel('Erro Total')
title('Convergência do Perceptron')

grid on

%% GRÁFICO 2 - EVOLUÇÃO DOS PESOS

figure

plot(pesos_hist(1:epoca,:),'LineWidth',2)

xlabel('Épocas')
ylabel('Valor dos Pesos')
title('Evolução dos Pesos')

legend('Bias',...
'Temperatura Água',...
'Temperatura Ambiente',...
'Umidade',...
'pH',...
'EC',...
'Location','best')

grid on

%% GRÁFICO 3 - CLASSE REAL X PREVISTA

figure

plot(Y_test,'bo-','LineWidth',2)
hold on

plot(Y_pred,'r*-','LineWidth',2)

xlabel('Amostras')
ylabel('Classe')
title('Classe Real x Classe Prevista')

legend('Real','Prevista')

grid on

%% GRÁFICO 4 - CUBO 3D DAS AMOSTRAS

figure

idx_neg = Y_train == -1;
idx_pos = Y_train == 1;

scatter3(X_train(idx_neg,2),...
         X_train(idx_neg,5),...
         X_train(idx_neg,6),...
         80,...
         'r',...
         'filled')
hold on

scatter3(X_train(idx_pos,2),...
         X_train(idx_pos,5),...
         X_train(idx_pos,6),...
         80,...
         'b',...
         'filled')

xlabel('Temperatura da Água')
ylabel('pH')
zlabel('Condutividade Elétrica')

title('Distribuição 3D das Amostras')

legend('Classe -1 (Qualidade Ruim)',...
       'Classe 1 (Qualidade Boa)',...
       'Location','best')

grid on
box on
rotate3d on
view(45,30)