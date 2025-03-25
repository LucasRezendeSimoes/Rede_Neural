import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import time

print('Carregando Arquivo de teste')

iteracoes = 7000
camadas = [(15, 7, 2), (30, 15, 8, 2), (35, 30, 24, 14, 8, 3)]
execucoes = 10

arquivos = ['teste2.npy', 'teste3.npy', 'teste4.npy', 'teste5.npy']
ativacao_por_arquivo = {
    'teste2.npy': 'tanh',
    'teste3.npy': 'relu',
    'teste4.npy': 'relu',
    'teste5.npy': 'relu'
}

arquivos = list(ativacao_por_arquivo.keys())

#--------------------------------------------------------------------------
for arquivo_teste in arquivos:
    print(f'\nExecutando para o arquivo: {arquivo_teste}\n')
    inicio = time.time()
    arquivo = np.load(arquivo_teste)
    x = arquivo[0]
    y = np.ravel(arquivo[1])

    func_ativacao = ativacao_por_arquivo[arquivo_teste]
    print(f"Usando função de ativação '{func_ativacao}' para este arquivo.")

    melhor = {
        "Gráfico1": (),
        "Gráfico2": (),
        "Gráfico3": (),
        "Camadas": 0,
        "Execução": 0,
        "Erro": float('inf')
    }

    for j, camada in enumerate(camadas):
        erros = []

        for i in range(execucoes):
            print(f"Simulação {j + 1}  Execução {i + 1} com camadas {camada}")
            regr = MLPRegressor(hidden_layer_sizes=camada,
                                max_iter=iteracoes,
                                activation=func_ativacao,
                                solver='adam',
                                learning_rate='adaptive',
                                n_iter_no_change=iteracoes)

            print('Treinando RNA')
            regr = regr.fit(x, y)

            print('Preditor')
            y_est = regr.predict(x)
            plt.figure(figsize=[14, 7])

            # Gráfico do curso original
            plt.subplot(1, 3, 1)
            plt.title("Curso original")
            plt.plot(x, y)

            # Gráfico de aprendizagem
            plt.subplot(1, 3, 2)
            plt.title("Aprendizagem")
            plt.plot(regr.loss_curve_)

            # Gráfico do regressor
            plt.subplot(1, 3, 3)
            plt.title("Regressor")
            plt.plot(x, y, linewidth=1, color='red', label="Real")
            plt.plot(x, y_est, linewidth=2, label="Estimado")
            plt.legend()

            plt.show()

            if regr.best_loss_ < melhor["Erro"]:
                melhor["Erro"] = regr.best_loss_
                melhor["Gráfico"] = plt
                melhor["Camadas"] = camada
                melhor["Execução"] = i + 1

            erros.append(regr.best_loss_)

        # Média e desvio padrão dos erros para esta camada
        media = np.mean(erros)
        desvio = np.std(erros)
        print(f"\033[34mCamadas: {camada} -> Média do Erro: {media}, Desvio Padrão: {desvio}\033[0m")

    print("\nMelhor configuração para este teste:")
    print(f"Camadas: {melhor['Camadas']}, Execução: {melhor['Execução']}, Erro: {melhor['Erro']}")
    melhor["Gráfico"].show()

    fim = time.time()
    print(f"Tempo de execução para {arquivo_teste}: {fim - inicio} segundos\n")
