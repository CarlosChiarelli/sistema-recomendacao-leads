# junto com pre-processamento é a que mais toma tempo
# demais classes são para auxiliar essa 

# algoritmos e otimização de hiperparâmetros

from pandas import Series, DataFrame

from preprocessamento import Preprocessamento
from fonte_dados import FonteDados 
from metricas import Metricas

class Experimentos:

    def __init__(self):
        from numpy.random import seed
        seed(42)
        '''
        self.modelos_testados = {'logistica' : LogisticRegression(),
                                  'knn': KNeighborsClassifier(),
                                  'decision_tree' : DecisionTreeClassifier(), 
                                  'random_forest': RandomForestClassifier(), 
                                  'svm': SVC(),
                                  'adaboost': AdaBoostClassifier(),
                                  'xgb': XGBClassifier(),
                                  'catboost': CatBoostClassifier()}
        '''                                    
        self.dic_modelos = None    
        self.metricas_models = {}
        self.y_treino = None
        
    def treinamento_modelo(self, X_train, y_train):
        '''
        Train the model with especified experiments
        :param X_train: pd.DataFrame with train data
        :param y_train: pd.Series with train labels
        :return: Dict with trained model
        '''
        for alg in self.modelos_testados.keys():  
            # treina cada modelo 
            print('Treinando o modelo ', alg)
            modelo = self.modelos_testados[alg]
            print(modelo, '\n\n')    
            modelo.fit(X_train, y_train)

            if self.dic_modelos is None:
                self.dic_modelos = {alg : modelo}
            else: 
                self.dic_modelos.update({alg:modelo})

        return self.dic_modelos

    # como experimento é focado em um tipos de modelo
    # tenho apenas um run_experiment 
    # executo ela após mapear tudo, então já coleto resultados
    # e passo para próxima iteração do processo
    def exec_experimentos(self):
        '''
        Run especified experiments
        :return: Dict with metrics
        '''

        preproc = Preprocessamento()

        print('Leitura dos dados', '\n\n')
        #train_df = DataSource().read_data(etapa_treino = True)
        #test_df, y_test = DataSource().read_data(etapa_treino = False)
        #y_test = y_test['SalePrice']
        X_treino = FonteDados().leitura_dados()
        #X_teste, y_teste = FonteDados().leitura_dados(etapa_treino=False)
        
        print('Pré-processamento dados de treino', '\n\n')
        #X_train, y_train = pre.process(train_df, etapa_treino = True)
        #X_treino = pre_proc.processo(X_treino)
        X_treino = preproc.processo_part1(X_treino, perc_miss=.10)

        # separando pedaço para teste
        X_teste = X_treino.iloc[0:5, :]

        X_treino = preproc.processo_part2(X_treino)

        #print('Pré-processamento dados de teste', '\n\n')
        #X_teste = pre_proc.processo(X_teste, etapa_treino=False)

        #print('Pré-processamento do target', '\n\n')
        #y_treino = pre_proc.processo(y_treino, target=True)
        #y_teste = pre_proc.processo(y_teste, etapa_treino=False, target=True)
        

        return X_treino, X_teste
'''
        print('Treinamento dos modelos')
        # fazemos o ravel para corrigir a dimensão do vetor a ser previsto
        modelos = Experimentos().treinamento_modelo(X_treino, y_treino)

        print('Executando métricas')
        for model in modelos.keys():

            print('\n', model)
            y_pred = modelos[model].predict(X_teste)

            # vejo as métricas (sempre calculadas no teste)
            #Metricas().calcula_classif(y_teste, Series(y_pred))
            metrics = Metricas().calcula_classif(y_teste, Series(y_pred))
            DataFrame.from_dict(metrics, orient= 'index').to_csv('../saida/metrica_'+model+'.csv')

            self.metricas_models[model] = metrics

        return metrics
                    
'''