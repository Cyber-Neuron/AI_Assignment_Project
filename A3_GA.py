# encoding=utf-8
'''
Created on Apr 11, 2017

@author: dan
'''


import math
import random
import matplotlib.pyplot as plt
import numpy as np

class Gene():
    def __init__(self, chromosome_length, amount, mutation_limit=10, isEarlyStop=False):
        
        self.c_length = chromosome_length
        self.population = self.init_population(chromosome_length, amount)
        # the trigger of mutation
        self.last_winner = 0
        self.early_stop = isEarlyStop
        self.mutation_num = 0
        # the limit of mutation
        self.mutation_limit = mutation_limit

    def reproduction(self, elitism_rate=0.2, survived_rate=0.5, mutation_rate=0.01, cross_over_rate=0.6):
        self.mutation_rate = mutation_rate
        # select parent
        parents = self.selection(elitism_rate, survived_rate)
        # mating and crossover
        self.mating(parents, cross_over_rate)
        # mutation
        self.mutation(mutation_rate)
    def create_chromosome(self, length):
        """
        Generate binary code as chromosome
        """
        chromosome = 0
        for i in xrange(length):
            chromosome |= (1 << i) * random.randint(0, 1)
        return chromosome

    def init_population(self, chromosome_length, amount):
        self.chromosome_length = chromosome_length
        self.amount = amount
        return [self.create_chromosome(chromosome_length) for i in xrange(amount)]

    def fitness(self, x):
        # convert binary to decimal
        # x = self.decode(chromosome)
        # fitness function +(5*np.sin(x/8))*(np.cos(x/19))
        return (-1.0 / 10000) * (x - 1023) * x + (5 * np.sin(x / 8)) * (np.cos(x / 19))

    def selection(self, elitism_rate, survived_rate):
        """
        keep the strongest one and randomly dorp others
        """
        # Find the most valuable ones
        
        ranked = self.get_strongest()
        # choose the top n strongest one
        retain_length = int(len(ranked) * elitism_rate)
        parents = ranked[:retain_length]
        # drop someone randomly via survived_rate
        for chromosome in ranked[retain_length:]:
            if self.lucky(survived_rate):
                parents.append(chromosome)
        return parents
    def get_strongest(self):
        # Each chromosome looks like this (16115.027477040323, 16) (165.02303255663, 111)
        strongest = [(self.fitness(self.decode(chromosome)), chromosome) for chromosome in self.population]
        # rank the population by fitness result
        strongest = [x[1] for x in sorted(strongest, reverse=True)]
        return strongest
    def mating(self, parents, cross_over_rate):
        """
        Mating and crossover
        """
        
        # newborn will be added into whole population and reproduce with others. 
        children = []
        # the number of newborn
        new_count = len(self.population) - len(parents)
        # reproduce children
        while len(children) < new_count:
            # choose parents randomly to reproduce
            
            parent_a = random.randint(0, len(parents) - 1)
            parent_b = random.randint(0, len(parents) - 1)
            if parent_a != parent_b:
                # choose chromosome break point randomly from parents and generate new chromosome
                cross_pos = random.randint(0, self.c_length)
                mask = 0
               
                if self.lucky(cross_over_rate):
                    for i in xrange(cross_pos):
                        mask |= (1 << i) 
                    parent_a = parents[parent_a]
                    parent_b = parents[parent_b]
                    # The child will obtain parents gene
                    child = ((parent_a & mask) | 
                            (parent_b & ~mask)) & ((1 << self.c_length) - 1)
                    children.append(child)
            else:
                if parent_a == 0 and parent_b == 0:
                    # to prevent the chromosome die out.
                    print 'The population was terminated!'
                    self.population = self.init_population(self.chromosome_length, self.amount)
                    return
        # update the whole population
        self.population = parents + children
    def lucky(self, rate):
        """
        randomly return true or false to control the mutation or survivors
        the value of rnd will be 1 or 2. the rate control the probability of 1 occurs
        """
        rnd = np.random.choice(np.arange(1, 3), p=[rate, 1 - rate])
        return rnd == 1
    def mutation(self, mutation_rate):
        """
        mutation, randomly choose one for mutation 
        """
        for i in xrange(len(self.population)):
            if self.lucky(mutation_rate):
                j = random.randint(0, self.c_length - 1)
                self.population[i] ^= 1 << j


    def decode(self, chromosome):
        """
        convert binary to decimal
        """
        return chromosome * 1023.0 / (2 ** self.c_length - 1)
    
    def output(self, name):
        """
        output current result
        """
        
        ranked = self.get_strongest()
        print(ranked[0])
        rst = self.decode(ranked[0])
        # if the result won't change then trigger mutation manually
        if self.last_winner == rst:
            if self.mutation_num % 2 == 0:
                self.mutation(self.mutation_rate)
            self.mutation_num += 1
            print name, " mutated ", self.mutation_num, "times."
            if self.mutation_num > self.mutation_limit:
                print "Too many mutations!"
                return -1
            if self.fitness(rst) > 31 and self.early_stop:
                print "Got it!"
                print 'f(x)=', self.fitness(rst), ': x=', rst
                return -1
        self.last_winner = rst
        
        print 'f(x)=', self.fitness(rst), ': x=', rst
        
        return self.fitness(rst)     


if __name__ == '__main__':
    """ 
    The length of chromosome will be 10 as 2^10=1023，the amount of chromosome will be 10. 
    The smaller of amount you choose the more detail of evolution you will see.
    The mutation limit is designed to prevent resource exhaust.
    """
    chromosome_length = 10
    pop_amount = 50
    mutation_limit = 2000
    epoch = 50
    isEarlyStop = False
    crossover_rate = [0.2, 0.4, 0.6, 0.8]
    
    
    geneA = Gene(chromosome_length, pop_amount, mutation_limit, isEarlyStop)
    geneB = Gene(chromosome_length, pop_amount, mutation_limit, isEarlyStop)
    geneC = Gene(chromosome_length, pop_amount, mutation_limit, isEarlyStop)
    geneD = Gene(chromosome_length, pop_amount, mutation_limit, isEarlyStop)
    history = {}
    history['A'] = []
    history['B'] = []
    history['C'] = []
    history['D'] = []
    
    # reproduce n iterations
    for i, cross_over_rate in enumerate(crossover_rate):
        for x in xrange(epoch):
        
            geneA.reproduction(elitism_rate=0.2, survived_rate=0.7, mutation_rate=0.01, cross_over_rate=cross_over_rate)
            geneB.reproduction(elitism_rate=0.2, survived_rate=0.7, mutation_rate=0.05, cross_over_rate=cross_over_rate)
            geneC.reproduction(elitism_rate=0.2, survived_rate=0.7, mutation_rate=0.1, cross_over_rate=cross_over_rate)
            geneD.reproduction(elitism_rate=0.2, survived_rate=0.7, mutation_rate=0.2, cross_over_rate=cross_over_rate)
            oa = geneA.output('A')
            ob = geneB.output('B')
            oc = geneC.output('C')
            od = geneD.output('D')
            terminate = (oa == -1 or ob == -1 or oc == -1 or od == -1)
            if not terminate:
                history['A'].append(oa)
                history['B'].append(ob)
                history['C'].append(oc)
                history['D'].append(od)
            else:
                break
            # print x ,history
        plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow', 'black', 'pink', 'lightblue'])
        plt.plot(history['A'], '.')
        plt.plot(history['B'], '.')
        plt.plot(history['C'], '.')
        plt.plot(history['D'], '.')
        average = (np.array(history['A']) + np.array(history['B']) + np.array(history['C']) + np.array(history['D'])) / 4
        best = np.amax([np.array(history['A']), np.array(history['B']), np.array(history['C']), np.array(history['D'])], axis=0)
        worst = np.amin([np.array(history['A']), np.array(history['B']), np.array(history['C']), np.array(history['D'])], axis=0)
        
        plt.plot(average)
        plt.plot(best)
        plt.plot(worst)
        plt.xlabel('Epoch')
        plt.ylabel('Max f(x)')
        plt.title('cross_over_rate ' + str(cross_over_rate) + '; population: ' + str(pop_amount))
        plt.legend(['mutation_rate:0.01', 'mutation_rate:0.05', 'mutation_rate:0.1', 'mutation_rate:0.2', 'Average', 'Best', 'Worst'], loc='upper left')
        plt.show()
