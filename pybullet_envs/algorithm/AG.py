import sys
import time
import math
import numpy as np
from random import shuffle, random, sample, randint, randrange, uniform, choice
from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
from copy import deepcopy
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
        ind1 = Individual(chromosome1)
        ind2 = Individual(chromosome2)
        return [ind1, ind2] 

    def mutation_flip(self):
        "Cambia aleatoriamente el alelo de un gen."
        new_chromosome = deepcopy(self.chromosome)
        mutGene = randrange(0,len(new_chromosome))   # escoge un gen para mutar
        if new_chromosome[mutGene] == 0:
            new_chromosome[mutGene] = 1
        else:
            new_chromosome[mutGene] = 0
        return Individual(new_chromosome)

    def mutation_inversion(self):
        """
        Invierte el orden de todos los genes comprendidos entre 2 puntos 
        seleccionados al azar en el cromosoma
        """
        new_chromosome = deepcopy(self.chromosome)
        cut_1 = randrange(0,len(new_chromosome))
        cut_2 = randrange(0,len(new_chromosome))
        
        min_cut = min(cut_1,cut_2)
        max_cut = max(cut_1,cut_2)
        
        aux = new_chromosome[min_cut:max_cut]
        aux.reverse()
        
        new_chromosome2 = new_chromosome[:min_cut]
        new_chromosome2.extend(aux)
        new_chromosome2.extend(new_chromosome[max_cut:])
        
        ## ESCRIBIR AQUI SU CODIGO
        
        return Individual(new_chromosome2)

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
    # fitness = np.zeros(2) # objetivos
    # fitness[0] =sum_reward
    # fitness[1] = chromosome[0]    
    return sum_reward + chromosome[0]

def evaluate_population(population):
    """ Evalua una poblacion de individuos con la funcion get_fitness """
    pop_size = len(population)

    for i in range(pop_size):
        if population[i].fitness == -1:    # evalua solo si el individuo no esta evaluado
            population[i].fitness = get_fitness(population[i].chromosome)

def select_parents_roulette(population):
    popsize = len(population)
    
    # Escoje el primer padre
    sumfitness = sum([indiv.fitness for indiv in population])  # suma total del fitness de la poblacion
    pickfitness = uniform(0, sumfitness)   # escoge un numero aleatorio entre 0 y sumfitness
    cumfitness = 0     # fitness acumulado
    for i in range(popsize):
        cumfitness += population[i].fitness
        if cumfitness > pickfitness: 
            iParent1 = i
            break
     
    # Escoje el segundo padre, desconsiderando el primer padre
    sumfitness = sumfitness - population[iParent1].fitness # retira el fitness del padre ya escogido
    pickfitness = uniform(0, sumfitness)   # escoge un numero aleatorio entre 0 y sumfitness
    cumfitness = 0     # fitness acumulado
    for i in range(popsize):
        if i == iParent1: continue   # si es el primer padre 
        cumfitness += population[i].fitness
        if cumfitness > pickfitness: 
            iParent2 = i
            break        
    return (population[iParent1], population[iParent2])


def select_parents_tournament(population, tournament_size):
    # Escoje el primer padre
    list_indiv=[]
    x1 = np.random.permutation(len(population) )
    y1= x1[0:tournament_size]
    for i in range(tournament_size):
        list_indiv.append(population[y1[i]].fitness)
    
    iParent1=np.argmax(list_indiv)
    
    # Escoje el segundo padre, desconsiderando el primer padre   
    x2 = np.delete(x1, iParent1)
    x2 = np.random.permutation(x2)
    list_indiv=[]
    y2= x2[0:tournament_size]
    for i in range(tournament_size):
        list_indiv.append(population[y2[i]].fitness)
    iParent2=np.argmax(list_indiv)
    
    return (population[x1[iParent1]],population[x2[iParent2]])

def select_survivors_ranking(population, offspring_population, numsurvivors):
    next_population = []
    population.extend(offspring_population) # une las dos poblaciones
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:numsurvivors]
    for i in range(numsurvivors):
        next_population.append(population[isurvivors[i]])
    return next_population

def select_survivors_steady(population, offspring_population, numind2replace):
    next_population = []
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)
    osurvivors = sorted(range(len(offspring_population)), key=lambda i: offspring_population[i].fitness, reverse=True)
    
    
    for i in range(len(population)-numind2replace):
        next_population.append(population[isurvivors[i]])
    for j in range(numind2replace):
        next_population.append(offspring_population[osurvivors[j]])
    
    return next_population



def genetic_algorithm(population, ngen=100, pmut=0.1, 
                      crossover="onepoint", mutation="flip", 
                      selection_parents_method="roulette", 
                      selection_survivors_method="ranking"):
    """Algoritmo Genetico para el problema de la mochila
        items:      pool de items a escoger para la mochila. 
                    Debe ser una lista de objetos de clase Item
        max_weight: maximo peso que puede soportar la mochila
        ngen:       maximo numero de generaciones 
        pmut:       tasa de mutacion
        crossover:  operador de cruzamiento
        mutation:   operador de mutacion
        selection_parents_method: método de selección de padres para cruzamiento
        selection_survivors_method: método de selección de sobrevivientes 
    """
    
    pop_size = len(population)
    evaluate_population(population)  # evalua la poblacion inicial
    ibest = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:1]  # mejor individuo
    bestfitness = [population[ibest[0]].fitness]  # fitness del mejor individuo
    print("Poblacion inicial, best_fitness = {}".format(population[ibest[0]].fitness))
    
    for g in range(ngen):   # Por cada generacion

        ## Selecciona parejas de individuos (mating_pool) para cruzamiento con el metodo de la ruleta
        mating_pool = []
        for i in range(int(pop_size/2)):
            if selection_parents_method == "roulette":
                mating_pool.append(select_parents_roulette(population))
            elif selection_parents_method == "tournament":
                mating_pool.append(select_parents_tournament(population, 3))
            else:
                raise NotImplementedError
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
                elif mutation == "inversion":
                    offspring_population[i] = offspring_population[i].mutation_inversion() # invierte todos los genes entre 2 puntos al azar
                else:
                    raise NotImplementedError   
        
        ## Evalua la poblacion descendencia creada
        evaluate_population(offspring_population)   # evalua la poblacion descendencia
        
        ## Selecciona individuos para la sgte. generación 
        if selection_survivors_method == "ranking":
            population = select_survivors_ranking(population, offspring_population, pop_size) #metodo de ranking
        elif selection_survivors_method == "steady":
            population = select_survivors_steady(population, offspring_population, int(pop_size/2)) #metodo steady-state
        else:
            raise NotImplementedError
            
        ## Almacena la historia del fitness del mejor individuo
        ibest = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:1]
        bestfitness.append(population[ibest[0]].fitness)
        
        if (g % 50 == 0):  # muestra resultados cada 10 generaciones
            print("generacion {}, (Mejor fitness = {})".format(g, population[ibest[0]].fitness))
        
    #print("Mejor individuo en la ultima generacion = {} (fitness = {})".format(population[ibest[0]].chromosome, population[ibest[0]].fitness))
    print("Mejor individuo en la ultima generacion (fitness = {})".format( population[ibest[0]].fitness))
    return population[ibest[0]], bestfitness  # devuelve el mejor individuo y la lista de mejores fitness x gen

def main():
    POP_SIZE = 5       # numero de individuos
    GENERATIONS = 5   # numero de generaciones
    PMUT = 0.1         # tasa de mutacion

    CROSSOVER = "uniform"
    #CROSSOVER = "onepoint"

    MUTATION = "flip"
    #MUTATION = "inversion"

    P_SELECTION = "roulette"
    #P_SELECTION = "tournament"

    S_SELECTION = "ranking"
    #S_SELECTION = "steady"


    ## Inicializa una poblacion inicial de forma aleatoria
    population = init_population(POP_SIZE, 5)

    # Evolue la poblacion con el algoritmo genetico (cruzamiento 'onepoint', )
    best_ind, bestfitness = genetic_algorithm(population, GENERATIONS, PMUT, 
                                          crossover=CROSSOVER,
                                          mutation=MUTATION, 
                                          selection_parents_method=P_SELECTION,
                                          selection_survivors_method=S_SELECTION)
    
    plt.plot(bestfitness)
    plt.savefig('AG-Monoobjetivo-'+CROSSOVER+'-'+MUTATION+'-'+P_SELECTION+'-'+S_SELECTION+'.png')
    #plt.show()
    print("Población: {}\tGeneraciones: {}\tTasa de mutacion: {}\n\nCROSSOVER: {}\nMUTATION: {}\nP_SELECTION: {}\nS_SELECTION: {}".format(POP_SIZE,GENERATIONS,PMUT,CROSSOVER,MUTATION,P_SELECTION,S_SELECTION))
    print(best_ind.chromosome)

if __name__ == '__main__':
  main()