# demanda muito tempo
# são muitos testes para ver ganho de performance 
# nas métricas do modelo 

#from funcoesProprias import dfExploracao
#import category_encoders as ce
#import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        #self.padron_posPca = None
        self.oneHotEnc = None
        self.padron_posPca = None
        self.moda_obj = {}
        self.pca = None
        self.linhasTsne = None

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

        return df


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
        #np.random.seed(0)

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
            #print(df.shape)
            del aux

            print('Normalização das numéricas', '\n')
            self.norm_prePca = MinMaxScaler()
            df[self.vars_num] = self.norm_prePca.fit_transform(df[self.vars_num])

            print('Redução de dimensionalidade', '\n')
            self.pca = PCA(n_components=0.95, random_state=42)
            df = self.pca.fit_transform(df)

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

            print('Redução de dimensionalidade', '\n')            
            df = self.pca.transform(df)

        return df


    def reducao_viz(self, df, etapa_treino=True, n_dim=2, perct_linhas=.05, perplexity=10):
        '''
        : ações: redução de dimensionalidade com TSNE
        : param x_treino: Dataframe para converter 
        : return tsne_results: DF reduzido
        : return sel: linhas selecionadas
        '''
        from datetime import datetime
        from time import sleep

        passado=datetime.now()

        if etapa_treino:
            # selecionando linhas aleatorias de acordo com percentual definidio
            np.random.seed(42)
            self.linhasTsne = np.random.choice(df.shape[0], size = int(df.shape[0]*perct_linhas), replace=False)
            self.linhasTsne = np.sort(self.linhasTsne)
            print('Dimensão do treino:', df[self.linhasTsne].shape)

            # perplexity defini quão próximo vão ficar os pontos das n dimensões em 2 dimensões            
            tSNE = TSNE(n_components=n_dim, random_state=42, perplexity = perplexity)
            print('Treinando T-sne:', tSNE, '\n')
            tsne_results = tSNE.fit_transform(df[self.linhasTsne])            

        
        agora=datetime.now()
        print('Tempo execucao:', agora-passado, '\n')
        return tsne_results



