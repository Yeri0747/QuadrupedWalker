import numpy as np
import math
from random import shuffle, random, sample, randint, randrange, uniform, choice
from copy import deepcopy
from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
import matplotlib.pyplot as plt

class Individual:
    """ Implementa el individuo del AG. Un individuo tiene un cromosoma que es una lista de NUM_ITEMS elementos (genes),
       cada gen i puede asumir dos posibles alelos: 0 o 1 (no incluir/incluir en la mochila el item i del pool) """

    def __init__(self, chromosome):  # el constructor recibe un cromosoma
        self.chromosome = chromosome[:]  
        self.fitness = -1  # -1 indica que el individuo no ha sido evaluado

    def crossover_onepoint(self, other):
        "Retorna dos nuevos individuos del cruzamiento de un punto entre individuos self y other "
        c = randrange(len(self.chromosome))
        ind1 = Individual(self.chromosome[:c] + other.chromosome[c:])
        ind2 = Individual(other.chromosome[:c] + self.chromosome[c:])        
        return [ind1, ind2]   
    
    
    def crossover_uniform(self, other):
        chromosome1 = []
        chromosome2 = []
        "Retorna dos nuevos individuos del cruzamiento uniforme entre self y other "
        for i in range(len(self.chromosome)):
            if uniform(0, 1) < 0.5:
                chromosome1.append(self.chromosome[i])
                chromosome2.append(other.chromosome[i])
            else:
                chromosome1.append(other.chromosome[i])
                chromosome2.append(self.chromosome[i])
        chromosome1[3]=choice([0,0.1,-0.1])
        chromosome2[3]=choice([0,0.1,-0.1])
        ind1 = Individual(chromosome1)
        ind2 = Individual(chromosome2)
        
        return [ind1, ind2] 

    def mutation_flip(self):
        "Cambia aleatoriamente el alelo de un gen."
        new_chromosome = deepcopy(self.chromosome)
        mutGene = randrange(0,len(new_chromosome))   # escoge un gen para mutar
        if mutGene == 3:
            new_chromosome[mutGene]=choice([0,0.1,-0.1])
        elif mutGene == 0:
            new_chromosome[mutGene] = (uniform(0,50))
        elif mutGene == 2 or mutGene ==4:
                valor = random()
                if valor > 0:
                    new_chromosome[mutGene] = (valor-0.1)
                else:
                    new_chromosome[mutGene] = (valor+0.1)
        else:
            new_chromosome[mutGene] = random()
        
        return Individual(new_chromosome)

def init_population(pop_size, chromosome_size):
    #Inicializa una poblacion de pop_size individuos, cada cromosoma de individuo de tamaño chromosome_size.
    population = []
    for i in range(pop_size):
        new_chromosome = []
        for i in range(5):
            if i==3:
                new_chromosome.append(choice([0.1,0,-0.1]))
            elif i==0:
                new_chromosome.append(uniform(0,100))
            elif i == 2 or i ==4:
                valor = random()
                if valor > 0:
                    new_chromosome.append(valor-0.1)
                else:
                    new_chromosome.append(valor+0.1)
            else:
                new_chromosome.append(random())
            #np.random.shuffle(new_chromosome)
            
        population.append( Individual(new_chromosome) )
    return population   

def get_fitness(chromosome):
    
    # speed = chromosome[0]
    # t = chromosome[1]
    # amplitude1 = chromosome[2]
    # steering_amplitude = chromosome[3]
    # amplitufe2 = chromosome[4]
    # time_step = chromosome[5]
    
    randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    environment = minitaur_gym_env.MinitaurBulletEnv(render=False,
                                                   motor_velocity_limit=np.inf,
                                                   pd_control_enabled=True,
                                                   hard_reset=False,
                                                   env_randomizer=randomizer,
                                                   on_rack=False)


    sum_reward = 0
    for step_counter in range(50):        
        t = step_counter * chromosome[1]
        a1 = math.sin(t * chromosome[0]) * (chromosome[2] + chromosome[3])
        a2 = math.sin(t * chromosome[0] + math.pi) * (chromosome[2] - chromosome[3])
        a3 = math.sin(t * chromosome[0]) * chromosome[4]
        a4 = math.sin(t * chromosome[0] + math.pi) * chromosome[4]
        action = [a1, a2, a2, a1, a3, a4, a4, a3]
        _, reward, done, _ = environment.step(action)
        sum_reward += reward
        if done:
            break        
    environment.reset()
    fitness = np.zeros(2) # objetivos
    fitness[0] =sum_reward
    fitness[1] = chromosome[0]    
    return fitness


def evaluate_population(population):
    """ Evalua una poblacion de individuos con la funcion get_fitness """
    pop_size = len(population)

    for i in range(pop_size):
        if population[i].fitness == -1:    # evalua solo si el individuo no esta evaluado
            population[i].fitness = get_fitness(population[i].chromosome)

def build_offspring_population(population, crossover, mutation, pmut):     
    """ Construye una poblacion hija con los operadores de cruzamiento y mutacion pasados
        crossover:  operador de cruzamiento
        mutation:   operador de mutacion
        pmut:       taza de mutacion
    """
    pop_size = len(population)
    
    ## Selecciona parejas de individuos (mating_pool) para cruzamiento con el metodo de la ruleta
    mating_pool = []
    for i in range(int(pop_size/2)): 
        # escoje dos individuos diferentes aleatoriamente de la poblacion
        permut = np.random.permutation( pop_size )
        mating_pool.append( (population[permut[0]], population[permut[1]] ) ) 
        
    ## Crea la poblacion descendencia cruzando las parejas del mating pool 
    offspring_population = []
    for i in range(len(mating_pool)): 
        if crossover == "onepoint":
            offspring_population.extend( mating_pool[i][0].crossover_onepoint(mating_pool[i][1]) ) # cruzamiento 1 punto
        elif crossover == "uniform":
            offspring_population.extend( mating_pool[i][0].crossover_uniform(mating_pool[i][1]) ) # cruzamiento uniforme
        else:
            raise NotImplementedError

    ## Aplica el operador de mutacion con probabilidad pmut en cada hijo generado
    for i in range(len(offspring_population)):
        if uniform(0, 1) < pmut: 
            if mutation == "flip":
                offspring_population[i] = offspring_population[i].mutation_flip() # cambia el alelo de un gen
            else:
                raise NotImplementedError   
                
    return offspring_population

def get_crowding_distances(fitnesses):
    """
    La distancia crowding de un individuo es la diferencia del fitness mas proximo hacia arriba menos el fitness mas proximo 
    hacia abajo. El valor crowding total es la suma de todas las distancias crowdings para todos los fitness
    """
    
    pop_size = len(fitnesses[:, 0])
    num_objectives = len(fitnesses[0, :])

    # crea matriz crowding. Filas representan individuos, columnas representan objectives
    crowding_matrix = np.zeros((pop_size, num_objectives))

    # normalisa los fitnesses entre 0 y 1 (ptp es max - min)
    normalized_fitnesses = (fitnesses - fitnesses.min(0)) / fitnesses.ptp(0)

    for col in range(num_objectives):   # Por cada objective
        crowding = np.zeros(pop_size)

        # puntos extremos tienen maximo crowding
        crowding[0] = 1
        crowding[pop_size - 1] = 1

        # ordena los fitness normalizados del objectivo actual
        sorted_fitnesses = np.sort(normalized_fitnesses[:, col])
        sorted_fitnesses_index = np.argsort(normalized_fitnesses[:, col])

        # Calcula la distancia crowding de cada individuo como la diferencia de score de los vecinos
        crowding[1:pop_size - 1] = (sorted_fitnesses[2:pop_size] - sorted_fitnesses[0:pop_size - 2])

        # obtiene el ordenamiento original
        re_sort_order = np.argsort(sorted_fitnesses_index)
        sorted_crowding = crowding[re_sort_order]

        # Salva las distancias crowdingpara el objetivo que se esta iterando
        crowding_matrix[:, col] = sorted_crowding

    # Obtiene las distancias crowding finales sumando las distancias crowding de cada objetivo 
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances

def select_by_crowding(population, num_individuals):
    """
    Selecciona una poblacion de individuos basado en torneos de pares de individuos: dos individuos se escoge al azar
    y se selecciona el mejor segun la distancia crowding. Se repite hasta obtener num_individuals individuos
    """    
    population = deepcopy(population)
    pop_size = len(population)
    
    num_objectives = len(population[0].fitness)
    
    # extrae los fitness de la poblacion en la matriz fitnesses
    fitnesses = np.zeros([pop_size, num_objectives])
    for i in range(pop_size): fitnesses[i,:] = population[i].fitness
        
    # obtiene las  distancias  crowding
    crowding_distances = get_crowding_distances(fitnesses)   
    
    population_selected = []   # poblacion escogida

    for i in range(num_individuals):  # por cada individuo a seleccionar

        # escoje dos individuos aleatoriamente de la poblacion no escogida aun
        permut = np.random.permutation( len(population) )
        ind1_id = permut[0]
        ind2_id = permut[1]

        # Si ind1_id es el mejor
        if crowding_distances[ind1_id] >= crowding_distances[ind2_id]:

            # traslada el individuo ind1 de population a la lista de individuos seleccionados
            population_selected.append( population.pop(ind1_id) )
            # remueve la distancia crowding del individuo seleccionado
            crowding_distances = np.delete(crowding_distances, ind1_id, axis=0)
            
        else:  # Si ind2_id es el mejor
            
            # traslada el individuo ind2 de population a la lista de individuos seleccionados
            population_selected.append( population.pop(ind2_id) )
            # remueve la distancia crowding del individuo seleccionado
            crowding_distances = np.delete(crowding_distances, ind2_id, axis=0)

    return (population_selected)



def get_paretofront_population(population):
    """
    Obtiene de population la poblacion de individups de la frontera de Pareto, 
    """
    population = deepcopy(population)
    pop_size = len(population)
    
    # todos los individuos son inicialmente asumidos como la frontera de Pareto
    pareto_front = np.ones(pop_size, dtype=bool)
    
    for i in range(pop_size): # Compara cada individuo contra todos los demas
        for j in range(pop_size):
            # Chequea si individuo 'i' es dominado por individuo 'j'
            #if all(population[j].fitness >= population[i].fitness) and any(population[j].fitness > population[i].fitness):
            #if str(all(population[j].fitness >= population[i].fitness)) and str(any(population[j].fitness > population[i].fitness)):
            if all(np.asarray(population[j].fitness) >= np.asarray(population[i].fitness)) and any(np.asarray(population[j].fitness) > np.asarray(population[i].fitness)):
                # j domina i -> señaliza que individuo 'i' como no siendo parte de la frontera de Pareto
                pareto_front[i] = 0
                break   # Para la busqueda para 'i' (no es necesario hacer mas comparaciones)

    paretofront_population = []
    for i in range(pop_size):  # construye la lista de individuos de la frontera de Pareto 
        if pareto_front[i] == 1: paretofront_population.append(population[i])
        
    return paretofront_population


def build_next_population(population, min_pop_size, max_pop_size):
    """
    Construye la poblacion de la siguiente generacion añadiendo sucesivas fronteras de Pareto hasta 
    tener una poblacion de al menos min_pop_size individuos. Reduce la frontera de Pareto con el metodo de
    crowding distance si al agregar la frontera excede el tamaño maximo de la poblacion (max_pop_size)
    """
    population = deepcopy(population)
    pareto_front = []
    next_population = []
    
    while len(next_population) < min_pop_size:   # mientras la poblacion no tenga el tamaño minimo
        # obtiene la poblacion frontera de Pareto actual
        paretofront_population = get_paretofront_population(population)
        
        # si poblacion actual + paretofront excede el maximo permitido -> reduce paretofront con el metodo de crowding
        combined_population_size = len(next_population) + len(paretofront_population)
        if  combined_population_size > max_pop_size:
            paretofront_population = select_by_crowding( paretofront_population, max_pop_size-len(next_population) ) 
        
        # Adiciona la frontera de Pareto (original o reducida) a la poblacion en construccion
        next_population.extend( paretofront_population )
    
        # remueve de population los individuos que fueron agregados a next_population 
        for i in range( len(paretofront_population) ):
            for j in range( len(population) ):
                if all( np.asarray(paretofront_population[i].chromosome) == np.asarray(population[j].chromosome) ):
                    del(population[j])
                    break
                    
    return next_population

def main():
    ## Hiperparametros del algoritmo genetico

    NUM_ITEMS = 5        # numero de items

    #POP_SIZE = 50
    MIN_POP_SIZE = 10
    MAX_POP_SIZE = 10
    CHROMOSOME_SIZE = NUM_ITEMS
    GENERATIONS = 20   # numero de generaciones
    PMUT = 0.35         # tasa de mutacion

    

    P = init_population( MAX_POP_SIZE, CHROMOSOME_SIZE )   # Crea  una poblacion inicial
    #  evalua la poblacion inicial
    evaluate_population(P)

    ## CODIGO PRINCIPAL DEL  ALGORITMO GENETICO  NSGA-II

    ## Ejecuta los ciclos evolutivos 
    for g in range(GENERATIONS):   # Por cada generacion
        
        if g %10 == 0:
            print ('Generacion {} (de {}) '.format(g, GENERATIONS))
        
        ## genera y evalua la poblacion hija    
        Q = build_offspring_population(P, "uniform", "flip", PMUT)
        evaluate_population(Q)
        
        ## une la poblacion padre y la poblacion hija
        P.extend(Q) 
        
        ## Construye la poblacion de la siguiente generacion
        P = build_next_population(P, MIN_POP_SIZE, MAX_POP_SIZE)

    # Obtiene la poblacion de la frontera de pareto final 
    pareto_front_population = get_paretofront_population(P)
    ## Plotea los individuos de la frontera de Pareto final
    pop_size = len(pareto_front_population)
    num_objectives = len(pareto_front_population[0].fitness)
        
    # extrae los fitness de la poblacion en la matriz fitnesses
    fitnesses = np.zeros([pop_size, num_objectives])
    for i in range(pop_size): fitnesses[i,:] = pareto_front_population[i].fitness

    x = fitnesses[:, 0]
    y = fitnesses[:, 1]
    plt.xlabel('Objectivo A - Reward')
    plt.ylabel('Objectivo B - Velocidad')
    plt.scatter(x,y)
    plt.savefig('pareto.png')
    for i in range(len(pareto_front_population)):
        a = pareto_front_population[i].chromosome
        b = pareto_front_population[i].fitness         
        print("Solucion", a, " - ", b)
    plt.show()

if __name__ == '__main__':
  main()
 #https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/