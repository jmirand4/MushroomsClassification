import random
import math

#valor exemplificativo para a velocidade de aprendizagem (tambem lhe podíamos ter chamado new)
alpha = 0.2

#------------------CÓDIGO GENÉRICO PARA CRIAR, TREINAR E USAR UMA REDE COM UMA CAMADA ESCONDIDA------------
"""Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
tal como foi discutido na teorica, as saidas destas estas unidades estao sempre a -1.
Por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
def make(nx, nz, ny):
    #a rede neuronal é um dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    return nn

#Funcao de activacao (sigmoide)
def sig(inp):
    return 1.0/(1.0 + math.exp(-inp))

"""Função que recebe uma rede nn e um padrao de entrada inp (uma lista) 
e faz a propagacao da informacao para a frente ate as saidas"""
def forward(nn, inp):
    #copia a informacao do vector de entrada in para a listavector de inputs da rede nn  
    nn['x']=inp.copy()
    nn['x'].append(-1)
    
    #calcula a activacao da unidades escondidas
    nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1)
    
    #calcula a activacao da unidades de saida
    nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
"""Funcao que recebe uma rede nn com as activacoes calculadas e a lista output de saidas pretendidas e calcula os erros
na camada escondida e na camada de saida"""
def error(nn, output):
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
"""Funcao que recebe uma rede com as activacoes e erros calculados e actualiza as listas de pesos"""
def update(nn):
    nn['wzx'] = [[w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    
"""Funcao que realiza uma iteracao de treino para um dado padrao de entrada inp com saida desejada output"""
def iterate(i, nn, inp, output):
    forward(nn, inp)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, inp, output, nn['y']))

#-----------------------CÓDIGO QUE PERMITE CRIAR E TREINAR REDES PARA APRENDER AS FUNÇÕES BOOLENAS--------------------
"""Funcao que cria uma rede 2x2x1 e treina a função lógica AND
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_and(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
"""Funcao que cria uma rede 2x2x1 e treina um OR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_or(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

"""Funcao que cria uma rede 2x2x1 e treina um XOR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""
def train_xor(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net
    
    
#-------------------------CÓDIGO QUE IRÁ PERMITIR CRIAR UMA REDE PARA APRENDER A CLASSIFICAR COGUMELOS---------    
"""Funcao principal do nosso programa para classificar cogumelos: cria os conjuntos de treino e teste, chama
a funcao que cria e treina a rede e, por fim, a funcao que a testa.
A funcao recebe como argumento o ficheiro correspondente ao dataset que deve ser usado e o numero de epocas que deve ser considerado no treino"""
def run_mushrooms(file, epocas):
    #train_size = 300 Definido no MAIN
    global train_size
    test_size = 1000
    train_set, test_set = build_sets(file, train_size, test_size)

    #Dados de treino
    neural_net = train_mushrooms(train_set, epocas)

    #Testar 
    test_mushrooms(neural_net, test_set)
    

"""Funcao que cria os conjuntos de treino e de de teste a partir dos dados
armazenados em f (mushrooms.csv). A funcao le cada linha, tranforma-a numa lista de valores e 
chama a funcao translate para a colocar no formato adequado para o padrao de treino. 
Estes padroes são colocados numa lista.
A função recebe como argumentos o nº de exemplos que devem ser considerados no conjunto de treino --->x e
o nº de exemplos que devem ser considerados no conjunto de teste ------> y
Finalmente, devolve duas listas, uma com x padroes (conjunto de treino)
e a segunda com y padrões (conjunto de teste). Atenção que x+y não pode ultrapassar o nº de cogumelos 
disponível no dataset"""
def build_sets(f,x,y):
    #Ler dataset
    with open(f, 'r') as file:
        lines = file.readlines()

    #Remover 1º linha
    lines = lines[1:]

    #Misturar linhas
    random.shuffle(lines)

    # Criar conjuntos de treino e teste
    train_data = lines[:x]
    test_data = lines[x:x + y]
    
    train_set = [translate(line.strip().split(',')) for line in train_data]
    test_set = [translate(line.strip().split(',')) for line in test_data]

    return train_set, test_set

"""A função translate recebe cada lista de valores simbólicos transforma-a num padrão de treino. 
Cada padrão é uma lista com o seguinte formato [padrao_de_entrada, classe_do_cogumelo, padrao_de_saida]
O enunciado do trabalho explica de que forma deve ser obtido o padrão de entrada
"""
def translate(lista):
    #Transformar as classes em números binários
    cap_shape_dict = {'b': [1, 0, 0, 0, 0, 0],
                  'c': [0, 1, 0, 0, 0, 0],
                  'x': [0, 0, 1, 0, 0, 0],
                  'f': [0, 0, 0, 1, 0, 0],
                  'k': [0, 0, 0, 0, 1, 0],
                  's': [0, 0, 0, 0, 0, 1]}

    cap_surface_dict = {'f': [1, 0, 0, 0],
                    'g': [0, 1, 0, 0],
                    'y': [0, 0, 1, 0],
                    's': [0, 0, 0, 1]}

    cap_color_dict = {'n': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'b': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  'c': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  'g': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  'r': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  'p': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  'u': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  'e': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  'w': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    bruises_dict = {'t': [1, 0],
                'n': [0, 1]}

    odor_dict = {'a': [1, 0, 0, 0, 0, 0, 0, 0, 0],
             'l': [0, 1, 0, 0, 0, 0, 0, 0, 0],
             'c': [0, 0, 1, 0, 0, 0, 0, 0, 0],
             'y': [0, 0, 0, 1, 0, 0, 0, 0, 0],
             'f': [0, 0, 0, 0, 1, 0, 0, 0, 0],
             'm': [0, 0, 0, 0, 0, 1, 0, 0, 0],
             'n': [0, 0, 0, 0, 0, 0, 1, 0, 0],
             'p': [0, 0, 0, 0, 0, 0, 0, 1, 0],
             's': [0, 0, 0, 0, 0, 0, 0, 0, 1]}

    gill_attachment_dict = {'a': [1, 0, 0, 0],
                        'd': [0, 1, 0, 0],
                        'f': [0, 0, 1, 0],
                        'n': [0, 0, 0, 1]}

    gill_spacing_dict = {'c': [1, 0, 0],
                    'w': [0, 1, 0],
                    'd': [0, 0, 1]}

    gill_size_dict = {'b': [1, 0],
                      'n': [0, 1]}
    
    stalk_shape_dict = {'e': [1, 0],
                        't': [0, 1]}
    
    stalk_root_dict = {'b': [1, 0, 0, 0, 0, 0, 0],
                       'c': [0, 1, 0, 0, 0, 0, 0],
                       'u': [0, 0, 1, 0, 0, 0, 0],
                       'e': [0, 0, 0, 1, 0, 0, 0],
                       'z': [0, 0, 0, 0, 1, 0, 0],
                       'r': [0, 0, 0, 0, 0, 1, 0],
                       '?': [0, 0, 0, 0, 0, 0, 1]}
    
    stalk_surface_above_ring_dict = {'f': [1, 0, 0, 0],
                                     'y': [0, 1, 0, 0],
                                     'k': [0, 0, 1, 0],
                                     's': [0, 0, 0, 1]}
    
    stalk_surface_below_ring_dict = {'y': [1, 0, 0, 0],
                                     's': [0, 1, 0, 0],
                                     'k': [0, 0, 1, 0],
                                     's': [0, 0, 0, 1]}    
    
    
    stalk_color_above_ring_dict = {'n': [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   'b': [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                   'c': [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   'g': [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   'o': [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   'p': [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   'e': [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                   'w': [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                   'y': [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    
    stalk_color_below_ring_dict = {'n': [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                   'b': [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                   'c': [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   'g': [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                   'o': [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                   'p': [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                   'e': [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                   'w': [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                   'y': [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    
    veil_type_dict = {'p': [1, 0],
                      'u': [0, 1]}
    
    veil_color_dict = {'n': [1, 0, 0, 0],
                       'o': [0, 1, 0, 0],
                       'w': [0, 0, 1, 0],
                       'y': [0, 0, 0, 1]}
    
    ring_number_dict = {'n': [1, 0, 0],
                       'o': [0, 1, 0],
                       't': [0, 0, 1]}
    
    ring_type_dict = {'c': [1, 0, 0, 0, 0, 0, 0 ,0],
                     'e': [0, 1, 0, 0, 0, 0, 0 ,0],
                     'f': [0, 0, 1, 0, 0, 0, 0 ,0],
                     'l': [0, 0, 0, 1, 0, 0, 0 ,0],
                     'n': [0, 0, 0, 0, 1, 0, 0 ,0],
                     'p': [0, 0, 0, 0, 0, 1, 0 ,0],
                     's': [0, 0, 0, 0, 0, 0, 1 ,0],
                     'z': [0, 0, 0, 0, 0, 0, 0 ,1]}    
    
    spore_print_color_dict = {'k': [1, 0, 0, 0, 0, 0, 0, 0, 0],
                              'n': [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              'b': [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              'h': [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              'r': [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              'o': [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              'u': [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              'w': [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              'y': [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    
    population_dict = {'a': [1, 0, 0, 0, 0, 0],
                       'c': [0, 1, 0, 0, 0, 0],
                       'n': [0, 0, 1, 0, 0, 0],
                       's': [0, 0, 0, 1, 0, 0],
                       'v': [0, 0, 0, 0, 1, 0],
                       'y': [0, 0, 0, 0, 0, 1]}
    
    habitat_dict = {'g': [1, 0, 0, 0, 0, 0, 0],
                    'l': [0, 1, 0, 0, 0, 0, 0],
                    'm': [0, 0, 1, 0, 0, 0, 0],
                    'p': [0, 0, 0, 1, 0, 0, 0],
                    'u': [0, 0, 0, 0, 1, 0, 0],
                    'w': [0, 0, 0, 0, 0, 1, 0],
                    'd': [0, 0, 0, 0, 0, 0, 1]}  
    
    #Juntar os dicionarios
    attributes_mapping = [cap_shape_dict, cap_surface_dict, cap_color_dict,
                        bruises_dict, odor_dict, gill_attachment_dict,
                        gill_spacing_dict, gill_size_dict, stalk_shape_dict,
                        stalk_root_dict, stalk_surface_above_ring_dict,
                        stalk_surface_below_ring_dict, stalk_color_above_ring_dict,
                        stalk_color_below_ring_dict, veil_type_dict, veil_color_dict,
                        ring_number_dict, ring_type_dict, spore_print_color_dict,
                        population_dict, habitat_dict]
  
    

    #Converter atributos para vetores binários
    input_pattern = []
    for value, attribute_mapping in zip(lista[1:], attributes_mapping):
        if value in attribute_mapping:
            input_pattern.extend(attribute_mapping[value])


    #Converter classe do cogumelo para vetor binário
    output_pattern = [1, 0] if lista[0] == 'e' else [0, 1]

    return [input_pattern, lista[0], output_pattern]

"""Cria a rede e chama a funçao iterate para a treinar. A função recebe como argumento o conjunto de treino
e o número de épocas que irão ser usadas para fazer o treino"""
def train_mushrooms(training_set, epocas):
    global n_camada_escondida
   
    nn = make(len(training_set[0][0]),n_camada_escondida , 2)  # Trocar o numeros escondidos no MAIN

    i = 1
    for epoca in range(epocas):
        for pattern in training_set:
            iterate(i, nn, pattern[0], pattern[2])
            i+=1

    return nn

"""Recebe o padrao de saida da rede e devolve a classe com que a rede classificou o cogumelo.
Devolve a classe que corresponde ao indice da saida com maior valor."""
def retranslate(out):
    #Converte a saída da rede para a classe "principal"
    return 'e' if out[0] > out[1] else 'p'

"""Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste.
Para cada padrao do conjunto de teste chama a funcao forward e determina a classe do cogumelo
que corresponde ao maior valor da lista de saida. A classe determinada pela rede deve ser comparada com a classe real,
sendo contabilizado o número de respostas corretas. A função calcula a percentagem de respostas corretas"""    
def test_mushrooms(net, test_set):
    #Predicts corretas
    correct_predictions = 0

    for i, pattern in enumerate(test_set, start=1):
        forward(net, pattern[0])
        output = net['y']
        predicted_class = retranslate(output)

        #Classe 'p' ou 'e'
        true_class = pattern[1]

        print(f'The network thinks mushrooms number {i} is {predicted_class}, it should be {true_class}')

        if predicted_class == true_class:
            correct_predictions += 1

    #Taxa de sucesso
    success_rate = (correct_predictions / len(test_set)) * 100
    print(f"\nSuccess rate: {success_rate:.2f}%")

if __name__ == "__main__":
        epocas = 1 #Ajustar enunciado
        train_size = 6000 #variavel global
        # test_size = 1000 Default pelo enunciado
        n_camada_escondida=20 #variavel global
        run_mushrooms('mushrooms.csv',epocas)
       