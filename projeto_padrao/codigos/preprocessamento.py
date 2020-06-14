# demanda muito tempo
# são muitos testes para ver ganho de performance 
# nas métricas do modelo 

#from funcoesProprias import dfExploracao
#import category_encoders as ce
#import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from pandas import Series, DataFrame, concat
import numpy as np

def configuraJupyter(nCols=1000, nLinhas=1000):
    '''
    :ações: seta mostrar todas colunas e tema plots para fundo dark
     '''
    from matplotlib.pyplot import style
    from pandas import set_option

    # tema para fundo dark
    style.use('classic')

    # mostrar todas colunas no comando head
    options = {'display': {'max_columns': nCols,
                           'max_rows': nLinhas}}

    for category, option in options.items():
        for op, value in option.items():
            set_option(f'{category}.{op}', value) 

def dfExploracao(df, quantUnicos=False):
    '''
    :ações: retorna df com nomes, tipos, percentual de NAs e contagem unicos da coluna
    '''        
    from pandas import DataFrame

    explora = DataFrame(data = {'colunas':list(df.columns),
                                'tipos':list(df.dtypes),
                                'na_perct':df.isna().sum() / df.shape[0]})

    if quantUnicos:
        explora['quantUnicos'] = df.nunique()

    # removendo indice
    explora.reset_index(inplace=True)
    explora.drop('index', axis=1, inplace=True)

    return explora


class Preprocessamento:

    def __init__(self):
        # criando objetos que quero salvar com a classe
        # para aproveitá-los depois
        self.df_nomes_tipos_treino = None
        #self.feature_names = None
        #self.std_scaler = None
        #self.categoric_features = None
        #self.numeric_features = None
        #self.catb = None
        #self.scaler = None
        #self.train_features = None

        self.vars_categ = None
        self.vars_num = None
        self.vars_bin = None
        self.input_varsNum = None
        self.input_varsCat = None

        self.norm_prePca = None
        self.padron_posPca = None
        self.oneHotEnc = None
        self.padron_posPca = None
        self.moda_obj = {}

        self.proc_part1 = {'cols_rem': None,
                           'limitesRenom_outros': None,
                           'cols_manipu': None,
                           'cols_bin': None}

    def processo_part1(self, df, etapa_treino=True, perc_miss=.10):
        '''
        Primeira etapa do pré-processamento dos dados
        1. remoção de colunas com mais de 50% de missings
        2. classes de categóricos nominais com baixa contagem serão agrupados como 'OUTROS' ou agrupados com a menor contagem
        3. colunas com variância muito baixa (desbalanceamento muito alto) serão removidas (exe: coluna booleana com 10% de TRUE)
        4. transforma binárias para inteiros

        :param df: Pandas DataFrame
        :param etapa_treino: Booleano
        :return: nada (altera o DF inserido)
        '''

        # 1. missings e colunas com baixa variancia
        if etapa_treino:

            #explora = Preprocessamento().dfExploracao(df)
            explora = dfExploracao(df)
            self.df_nomes_tipos_treino = explora.copy()
            
            self.perc_miss_rm = perc_miss
            self.proc_part1['cols_rem'] = explora[explora['na_perct'] > perc_miss]['colunas'].tolist()
        
            # cols com baixa variancia
            self.proc_part1['cols_rem'] = self.proc_part1['cols_rem'] + ['dt_situacao',
                                                                        'fl_spa',
                                                                        'fl_antt',
                                                                        'de_indicador_telefone', 
                                                                        'fl_simples_irregular',
                                                                        # numerica, porem ela já existe discretizada em intervalos
                                                                        'idade_empresa_anos',
                                                                        'fl_me', 
                                                                        'fl_sa',
                                                                        'fl_epp', 
                                                                        'fl_ltda', 
                                                                        'fl_st_especial',
                                                                        'fl_veiculo',
                                                                        'fl_matriz',
                                                                        'sg_uf_matriz']#,
                                                                        #'vl_total_veiculos_pesados_grupo',
                                                                        #'vl_total_veiculos_leves_grupo']
        
        print('Remoção das colunas', '\n')
        df.drop(self.proc_part1['cols_rem'], axis=1, inplace=True)

        # 2. renomeando categóricas nominais com baixa contagem
        def converteMenorClasseOutros(df, coluna, limite):
            '''
            : param df: Dataframe a ser manipulado
            : param coluna: coluna alvo
            : param limite: abaixo desse limite, a categoria é renomeada para OUTROS
            '''
            lgl = df[coluna].value_counts() < limite
            valores = lgl.index

            valores = valores[np.where(lgl)].tolist()
            df[coluna] = ['OUTROS' if x in valores else x for x in df[coluna]]

        # limites para categorizar
        if etapa_treino:
            self.proc_part1['limitesRenom_outros'] = {'de_natureza_juridica':29e3,
                                                      'natureza_juridica_macro':4e3,
                                                      'de_ramo':21e3,
                                                      'setor':30e3,
                                                      'nm_divisao':14e3,
                                                      'nm_segmento':20e3}
                                                      #'sg_uf_matriz':10e3
                                                      #'nm_meso_regiao':15e3,
                                                      #'nm_micro_regiao':15e3}
        
        print('Categorizando categóricas nominais como OUTROS', '\n')
        for nomeCol, limiteCol in self.proc_part1['limitesRenom_outros'].items():
            converteMenorClasseOutros(df, nomeCol, limiteCol)

        # 3. correção de colunas especificas
        if etapa_treino:
            self.proc_part1['cols_manipu'] = ['fl_rm',
                                              'de_saude_tributaria',
                                              'de_nivel_atividade',
                                              'de_faixa_faturamento_estimado',
                                              'de_faixa_faturamento_estimado_grupo',
                                              'idade_emp_cat',
                                              'de_saude_rescencia',
                                              'vl_faturamento_estimado_aux',
                                              'vl_faturamento_estimado_grupo_aux',
                                              'qt_filiais',
                                              'vl_total_veiculos_leves_grupo',
                                              'vl_total_veiculos_pesados_grupo',
                                              'nu_meses_rescencia']

        print('Manipulação de colunas especificas', '\n')
        # coluna fl_rm
        df['fl_rm'] = [True if x == 'SIM' else False for x in df['fl_rm']]

        # coluna de_saude_tributaria 
        df['de_saude_tributaria'] = [x if x != 'VERMELHO' else 'LARANJA' for x in df['de_saude_tributaria']]

        # coluna de_nivel_atividade
        df['de_nivel_atividade'] = ['BAIXA' if x == 'MUITO BAIXA' else x for x in df['de_nivel_atividade']]
        
        dic_aux = {'MEDIA':1,
                   'ALTA':2,
                   'BAIXA':0}
        
        df['de_nivel_atividade'] = df['de_nivel_atividade'].map(dic_aux).fillna(df['de_nivel_atividade'])
        
        # colunas de_faixa_faturamento_estimado  e  de_faixa_faturamento_estimado_grupo
        dic_aux = {'DE R$ 81.000,01 A R$ 360.000,00':1,             
                   'ATE R$ 81.000,00':0,                             
                   'DE R$ 360.000,01 A R$ 1.500.000,00':2,           
                   'DE R$ 1.500.000,01 A R$ 4.800.000,00':2,         
                   'DE R$ 4.800.000,01 A R$ 10.000.000,00':2,         
                   'DE R$ 10.000.000,01 A R$ 30.000.000,00':2,        
                   'SEM INFORMACAO': np.nan,                                
                   'DE R$ 30.000.000,01 A R$ 100.000.000,00':2,        
                   'DE R$ 100.000.000,01 A R$ 300.000.000,00':2,      
                   'DE R$ 300.000.000,01 A R$ 500.000.000,00':2,       
                   'DE R$ 500.000.000,01 A 1 BILHAO DE REAIS':2,       
                   'ACIMA DE 1 BILHAO DE REAIS':2}
        
        df['de_faixa_faturamento_estimado'] = df['de_faixa_faturamento_estimado'].map(dic_aux).fillna(df['de_faixa_faturamento_estimado'])
        df['de_faixa_faturamento_estimado_grupo'] = df['de_faixa_faturamento_estimado_grupo'].map(dic_aux).fillna(df['de_faixa_faturamento_estimado_grupo'])
        
        df['de_faixa_faturamento_estimado'] = [x if x != 'SEM INFORMACAO' else np.nan for x in df['de_faixa_faturamento_estimado']]

        # colunas idade_emp_cat
        dic_aux = {'1 a 5':1,
                   '5 a 10':2,
                   '> 20':5,
                   '10 a 15':3,
                   '<= 1':0,
                   '15 a 20':4}
        
        df['idade_emp_cat'] = df['idade_emp_cat'].map(dic_aux).fillna(df['idade_emp_cat'])
        
        # coluna de_saude_rescencia
        df['de_saude_rescencia'] = ['SEM INFORMACAO' if x == 'ATE 3 MESES' or x == 'ATE 6 MESES' else x for x in df['de_saude_rescencia']]
        
        dic_aux = {'ACIMA DE 1 ANO':1,
                   'ATE 1 ANO':0,
                   'SEM INFORMACAO': np.nan}
        
        df['de_saude_rescencia'] = df['de_saude_rescencia'].map(dic_aux).fillna(df['de_saude_rescencia'])
        df['de_saude_rescencia'] = [x if x != 'SEM INFORMACAO' else np.nan for x in df['de_saude_rescencia']]

        # colunas vl_faturamento_estimado_aux e vl_faturamento_estimado_grupo_aux
        def intervalo(x):
            valor = 210000
            
            if x < valor:
                return 0
            elif x == valor:
                return 1
            elif x > valor:
                return 2
            else:
                return np.nan

        df['vl_faturamento_estimado_aux'] = df['vl_faturamento_estimado_aux'].apply(intervalo)
        df['vl_faturamento_estimado_grupo_aux'] = df['vl_faturamento_estimado_grupo_aux'].apply(intervalo) 

        # coluna qt_filiais, vl_total_veiculos_leves_grupo, vl_total_veiculos_pesados_grupo
        df['qt_filiais'] = [0 if x == 0 else 1 for x in df['qt_filiais']]
        df['vl_total_veiculos_leves_grupo'] = [0 if x == 0 else 1 for x in df['vl_total_veiculos_leves_grupo']]
        df['vl_total_veiculos_pesados_grupo'] = [0 if x == 0 else 1 for x in df['vl_total_veiculos_pesados_grupo']]


        # colunas nu_meses_rescencia 
        def intervalo(x):
            inf, sup = 21, 27
            
            if x < inf:
                return 0
            elif x >= inf and x <= sup:
                return 1
            elif x > sup:
                return 2
            else:
                return np.nan

        df['nu_meses_rescencia'] = df['nu_meses_rescencia'].apply(intervalo)
                
        # 4. binárias para inteiros    
        if etapa_treino:
            explora = dfExploracao(df, quantUnicos=True)
            self.proc_part1['cols_bin'] = explora[explora['quantUnicos']==2]['colunas']

        print('Transformando colunas binárias', '\n')
        for col in self.proc_part1['cols_bin']:
            df[col] = df[col].astype(bool).astype(int)


    def processo_part2(self, df, etapa_treino=True):
        '''
        Primeira etapa do pré-processamento dos dados
        1. preenche missings (objeto - moda, float - mediana)
        2. one-hot-encoder das dummies
        3. padronizacao
        4. PCA (95% variancia explicada)

        :param df: Pandas DataFrame
        :param etapa_treino: Booleano
        :return: nada (altera o DF inserido)
        '''
        if etapa_treino:
            print('Salvando tipo das variáveis', '\n')
            explora = dfExploracao(df)
            self.vars_bin = explora[explora['tipos'] == 'int32']['colunas']
            self.vars_num = explora[(explora['tipos'] == 'int64') | (explora['tipos'] == 'float64')]['colunas']
            self.vars_categ = explora[explora['tipos'] == 'object']['colunas']

            print('Preenchimento dos missings das numéricas', '\n')
            self.input_varsNum = SimpleImputer(strategy='median')
            df[self.vars_num] = self.input_varsNum.fit_transform(df[self.vars_num])

            print('Preenchimento dos missings das categoricas', '\n')
            for col in self.vars_categ:
                self.moda_obj[col] = df[col].mode()[0]
                df[col] = df[col].fillna(self.moda_obj[col])

            print('Codificação one-hot-encoder', '\n')
            self.oneHotEnc = OneHotEncoder()
            aux = self.oneHotEnc.fit_transform(df[self.vars_categ])
            df = concat([df.reset_index(drop=True).drop(self.vars_categ, axis=1), DataFrame(aux.toarray())], axis=1)
            print(df.shape)
            del aux

            print('Normalização das numéricas', '\n')
            self.norm_prePca = MinMaxScaler()
            df[self.vars_num] = self.norm_prePca.fit_transform(df[self.vars_num])

        else:
            print('Preenchimento dos missings das numéricas', '\n')
            df[self.vars_num] = self.input_varsNum.transform(df[self.vars_num])

            print('Preenchimento dos missings das categoricas', '\n')
            for col in self.vars_categ:
                df[col] = df[col].fillna(self.moda_obj[col])

            print('Codificação one-hot-encoder', '\n')
            aux = self.oneHotEnc.transform(df[self.vars_categ])
            df = concat([df.reset_index(drop=True).drop(self.vars_categ, axis=1), DataFrame(aux.toarray())], axis=1)
            del aux

            print('Normalização pré PCA', '\n')
            df[self.vars_num] = self.norm_prePca.transform(df[self.vars_num])

        return df







    def processo(self, df_input, etapa_treino=True, perc_miss=.5, target=False):
        '''
        Processo para treinamento do modelo
        1. Discretiza e corrige variaveis: tipo_escola, Q025, sexo, ano_de_conclusao, raça
        2. Remove colunas por completude e significado
        3. Rotula categóricas ordinais

        :param df: Pandas DataFrame
        :param etapa_treino: Boolean
        :return: Pandas Data Frame processado

        '''

        # diferenciar etapa TREINO/TESTE
        # pois fit_transform Treino e apenas transform no Teste


        if etapa_treino:
            # CALCULO tudo se for etapa de TREINO

            print('Definindo colunas a serem removidas')
            # colunas com muitos NAs
            self.perc_miss_rm = perc_miss
            cols_miss = df.isnull().mean() >= self.perc_miss_rm
            cols_miss = df.columns[cols_miss].tolist()

            # colunas irrelevantes
            cols_sem_relevancia = [True if x.startswith('CO_') or x.startswith('SG') or x.startswith('IN_') else False for x in df.columns]
            cols_sem_relevancia = df.columns[cols_sem_relevancia].tolist()
            
            cols_presenca = [True if 'PRESENCA' in x else False for x in df.columns]
            cols_presenca = df.columns[cols_presenca].tolist()

            # colunas para remover
            self.cols_rem = cols_miss + cols_sem_relevancia + cols_presenca + ['TP_NACIONALIDADE']

            print('Variáveis removidas (significância e completude', self.perc_miss_rm*100, '%)', '\n')
            df.drop(self.cols_rem, axis=1, inplace=True)

            # salvando colunas e tipos do treino
            print('Salvando tipos e nomes das colunas de treino', '\n')
            self.df_nomes_tipos_treino = dfExploracao(df)[['colunas', 'tipos']]

            # tipos categóricas ordinais
            self.vars_categ_ord = self.df_nomes_tipos_treino[self.df_nomes_tipos_treino['tipos'] == 'object']['colunas']
            self.vars_numericas = self.df_nomes_tipos_treino[self.df_nomes_tipos_treino['tipos'] != 'object']['colunas']

            print('Rotulação das categóricas ordinais', '\n')
            for coluna in self.vars_categ_ord:
                rotula_temp = LabelEncoder()
                df[coluna] = rotula_temp.fit_transform(df[coluna])
                self.categ_ordinal.append(rotula_temp)

            print('Preenchimento dos missings das numéricas', '\n')
            self.imputador_miss = SimpleImputer(strategy='constant', fill_value=0)
            df[self.vars_numericas] = self.imputador_miss.fit_transform(df[self.vars_numericas])

            print('Normalização dos dados (robusto)', '\n')
            self.normalizador = RobustScaler()
            df[self.vars_numericas] = self.normalizador.fit_transform(df[self.vars_numericas])


            # testar uma outra hora (codificação em variáveis dummies)
            # processar categoricas (catBoost um dos melhores para categoricas)
            #self.catb = ce.CatBoostEncoder(cols=self.categoric_features)
            #df[self.categoric_features] = self.catb.fit_transform(df[self.categoric_features], y=y)

            #return df[self.categoric_features + self.numeric_features], y
        
        else:
            # APLICO tudo se for etapa de TESTE

            print('Variáveis removidas (significância e completude ', self.perc_miss_rm*100, '%)', '\n')
            df.drop(self.cols_rem, axis=1, inplace=True)

            print('Rotulação das categóricas ordinais', '\n')
            for coluna, rotulador in zip(self.vars_categ_ord, self.categ_ordinal):
                df[coluna] = rotulador.transform(df[coluna])

            print('Preenchimento dos missings das numéricas', '\n')
            df[self.vars_numericas] = self.imputador_miss.transform(df[self.vars_numericas])

            print('Normalização dos dados (robusto)', '\n')
            df[self.vars_numericas] = self.normalizador.transform(df[self.vars_numericas])

            #df[self.categoric_features] = self.catb.transform(df[self.categoric_features])

            #return df[self.categoric_features + self.numeric_features]
        
        return df

    def reducao_dim(self, x_treino, n_dim=2, metodo='pca'):
        '''
        : ações: redução de dimensionalidade com TSNE
        : param x_treino: Dataframe para converter 
        : return: DF reduzido
        '''
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        if metodo == 'pca':
            pca = PCA(n_components=n_dim)
            pca.fit_transform(x_treino)
            x_reduzido = pca.transform(x_treino)
            return x_reduzido

        else:
            tsne = TSNE(n_components=n_dim)
            tsne_results = tsne.fit_transform(x_treino)
            return tsne_results


    def balanceamento_oversampling(self, x_treino, y_treino):
        '''
        : ações: transforma dados desbalanceados em balanceados (aumenta classe minoritaria)
        : param x_treino: Dataframe para converter
        : return: DF balanceado
        '''
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(sampling_strategy ="minority")

        X_smote, y_smote = smote.fit_resample(x_treino, y_treino['IN_TREINEIRO'])

        print('Dimensões antes:', (x_treino.shape, y_treino.shape), ' |  Dimensões depois:', (X_smote.shape, y_smote.shape))
        return X_smote, y_smote
