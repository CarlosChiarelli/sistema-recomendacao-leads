# classe 1 
# aqui faço o split TREINO/TESTE

from pandas import read_csv

class FonteDados:

    def __init__(self):
    	# criando variáveis
    	# definir os caminhos dados de treino e teste 
        self.caminho_dados = '../dados/estaticos_market.csv'
        self.dic_camin_testes = {
        							1:'../dados/estaticos_portfolio1.csv',
 									2:'../dados/estaticos_portfolio2.csv',
 									3:'../dados/estaticos_portfolio3.csv'
 								}

    # definir se a etapa é dados de treino ou teste    
    def leitura_dados(self, etapa_treino=True, numTeste=1):
        '''
            Lê os dados da fonte de dados
            :param etapa_treino: booleano especificando se é treino (falso é teste).
            :return: pd.DataFrame com valores, caso seja teste retorna lista com testes
        '''

        id_col = 'id'

        # selecionando colunas que existem na submissao
        df = read_csv(self.caminho_dados, index_col=id_col)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        if etapa_treino:
        	print('Dimensões:', df.shape)
        	return df


