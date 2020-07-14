import matplotlib.pyplot as plt
from pybullet_envs.bullet import minitaur_gym_env
from pybullet_envs.bullet import minitaur_env_randomizer
import math
import numpy as np
from random import shuffle, random, sample, randint, randrange, uniform, choice, seed
from copy import deepcopy

class Individual:
    "Clase abstracta para individuos de un algoritmo evolutivo."

    def __init__(self, chromosome):
        self.chromosome = chromosome

    def crossover(self, other):
        "Retorna un nuevo individuo cruzando self y other."
        raise NotImplementedError
        
    def mutate(self):
        "Cambia los valores de algunos genes."
        raise NotImplementedError

        

class Individual_minitaur(Individual):
    "Clase que implementa el individuo en el problema de las n-reinas."

    def __init__(self, chromosome):
        self.chromosome = chromosome[:]
        self.fitness = -1

    def crossover_onepoint(self, other):
        "Retorna dos nuevos individuos del cruzamiento de un punto entre self y other "
        c = randrange(len(self.chromosome))
        ind1 = Individual_minitaur(self.chromosome[:c] + other.chromosome[c:])
        ind2 = Individual_minitaur(other.chromosome[:c] + self.chromosome[c:])
        return [ind1, ind2]   
    
    def crossover_uniform(self, other):
        "Retorna un nuevo individuo cruzando self y other."
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
        ind1 = Individual_minitaur(chromosome1)
        ind2 = Individual_minitaur(chromosome2)
        
        return [ind1, ind2]     

    def mutate_position(self):
        "Cambia aleatoriamente la posicion de una reina."
        mutated_ind = Individual_minitaur(self.chromosome[:])
        indexPos = randint(0, len(mutated_ind.chromosome)-1)
        newPos = randint(0, len(mutated_ind.chromosome)-1)
        mutated_ind.chromosome[indexPos] = newPos
        return mutated_ind
    
    def mutate_swap(self):
        "Intercambia la posicion de dos genes."
        mutated_ind = Individual_minitaur(self.chromosome[:])
        indexOne = random.randint(0,len(mutated_ind.chromosome)-1)
        indexTwo = random.randint(0,len(mutated_ind.chromosome)-1)
        temp = mutated_ind.chromosome[indexOne]
        mutated_ind.chromosome[indexOne] = mutated_ind.chromosome[indexTwo]
        mutated_ind.chromosome[indexTwo] = temp
        return mutated_ind

    def mutate_flip(self):
        "Cambia aleatoriamente el alelo de un gen."
        new_chromosome = deepcopy(self.chromosome)
        mutGene = randrange(0,len(new_chromosome))   # escoge un gen para mutar
        if mutGene == 3:
            new_chromosome[mutGene]=choice([0,0.1,-0.1])
        elif mutGene == 0:
            new_chromosome[mutGene] = uniform(0,100)
        elif mutGene == 2 or mutGene ==4:
                valor = random()
                if valor > 0:
                    new_chromosome[mutGene] = (valor-0.1)
                else:
                    new_chromosome[mutGene] = (valor+0.1)
        else:
            new_chromosome[mutGene] = random()

        return Individual_minitaur(new_chromosome)

def fitness_minitaur(chromosome, steps):
    """Retorna el fitness de un cromosoma """
    # speed = chromosome[0]
    # time_step = chromosome[1]
    # amplitude1 = chromosome[2]
    # steering_amplitude = chromosome[3]
    # amplitufe2 = chromosome[4]
    
    randomizer = (minitaur_env_randomizer.MinitaurEnvRandomizer())
    environment = minitaur_gym_env.MinitaurBulletEnv(render=False,
                                                   motor_velocity_limit=np.inf,
                                                   pd_control_enabled=True,
                                                   hard_reset=False,
                                                   env_randomizer=randomizer,
                                                   on_rack=False)


    sum_reward = 0
    for step_counter in range(steps):        
        t = step_counter * chromosome[1]
        a1 = math.sin(t * chromosome[0]) * (chromosome[2] + chromosome[3])
        a2 = math.sin(t * chromosome[0] + math.pi) * (chromosome[2] - chromosome[3])
        a3 = math.sin(t * chromosome[0]) * chromosome[4]
        a4 = math.sin(t * chromosome[0] + math.pi) * chromosome[4]
        action = [a1, a2, a2, a1, a3, a4, a4, a3]
        _, reward, done, _ = environment.step(action)
        sum_reward += reward
        if done:
            sum_reward += 5   #castigo por caerse
            break        
    environment.reset()
    fitness = sum_reward + chromosome[0] # objetivos
    return fitness

def evaluate_population(population, fitness_fn, steps):
    """ Evalua una poblacion de individuos con la funcion de fitness pasada """
    popsize = len(population)
    for i in range(popsize):
        if population[i].fitness == -1:    # si el individuo no esta evaluado
            population[i].fitness = fitness_fn(population[i].chromosome, steps)

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
     
    # Escoje el segundo padre, desconsiderando el padre ya escogido
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


def select_survivors(population, offspring_population, numsurvivors):
    next_population = []
    population.extend(offspring_population) # une las dos poblaciones
    isurvivors = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:numsurvivors]
    for i in range(numsurvivors): next_population.append(population[isurvivors[i]])
    return next_population



def genetic_algorithm(population, fitness_fn, ngen=100, pmut=0.1, steps = 300):
    "Algoritmo Genetico "
    
    popsize = len(population)
    evaluate_population(population, fitness_fn, steps)  # evalua la poblacion inicial
    ibest = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:1]
    bestfitness = [population[ibest[0]].fitness]
    print("Poblacion inicial, best_fitness = {}".format(population[ibest[0]].fitness))
    
    for g in range(ngen):   # Por cada generacion
        
        ## Selecciona las parejas de padres para cruzamiento 
        mating_pool = []
        for i in range(int(popsize/2)): mating_pool.append(select_parents_roulette(population)) 
        
        ## Crea la poblacion descendencia cruzando las parejas del mating pool con Recombinación de 1 punto
        offspring_population = []
        for i in range(len(mating_pool)): 
            #offspring_population.extend( mating_pool[i][0].crossover_onepoint(mating_pool[i][1]) )
            offspring_population.extend( mating_pool[i][0].crossover_uniform(mating_pool[i][1]) )

        ## Aplica el operador de mutacion con probabilidad pmut en cada hijo generado
        for i in range(len(offspring_population)):
            if uniform(0, 1) < pmut: 
                #offspring_population[i] = offspring_population[i].mutate_swap()
                offspring_population[i] = offspring_population[i].mutate_flip()
        
        ## Evalua la poblacion descendencia
        evaluate_population(offspring_population, fitness_fn, steps)  # evalua la poblacion inicial
        
        ## Selecciona popsize individuos para la sgte. generación de la union de la pob. actual y  pob. descendencia
        population = select_survivors(population, offspring_population, popsize)

        ## Almacena la historia del fitness del mejor individuo
        ibest = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)[:1]
        bestfitness.append(population[ibest[0]].fitness)
        print("generacion {}, best_fitness = {}".format(g, population[ibest[0]].fitness))
    
    return population[ibest[0]], bestfitness  




def genetic_search_minitaur(fitness_fn, chromosome_size=10, pop_size=10, ngen=100, pmut=0.5, steps = 300):
    
    population = []

    ## Crea la poblacion inicial con cromosomas aleatorios
    for i in range(pop_size):
        new_chromosome = []
        for i in range(chromosome_size):
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
            
        population.append( Individual_minitaur(new_chromosome) )
   
        
    
    return genetic_algorithm(population, fitness_fn, ngen, pmut, steps)
        




# busca solucion para el problema de 10 reinas. Usa 100 individuos aleatorios, 100 generaciones y taza de mutación de 0.5

def main():
    CHROMOSOME_SIZE = 5
    GENERATIONS = 10   # numero de generaciones
    PMUT = 0.6         # tasa de mutacion
    STEPS = 500 #pasos en el entorno
    POP_SIZE = 15

    best_ind, bestfitness = genetic_search_minitaur(fitness_minitaur, CHROMOSOME_SIZE, POP_SIZE, GENERATIONS, PMUT, STEPS)
    print("Mejor resultado: ", best_ind.chromosome)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.plot(bestfitness)
    plt.savefig('AG-monobjetivo_10_500.png')
    plt.show()


if __name__ == '__main__':
  main()