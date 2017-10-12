import random
import numpy as np

class Chromosome:
    def __init__(self, bitstring):
        self.bitstring = bitstring
        self.fitness = 999

    def __lt__(self, other):
        return (self.fitness < other.fitness)

    def __eq__(self, other):
        return (self.fitness == other.fitness)

    def __gt__(self, other):
        return(self.fitness > other.fitness)

    def __le__(self, other):
        return(self.fitness <= other.fitness)

    def __cmp__(self, other):
        if self.fitness < other.fitness:
            return -1
        elif self.fitness > other.fitness:
            return 1
        else:
            return 0

    def __repr__(self):
        return ' '.join([str(i) for i in self.bitstring])

    def printSolution(self):
        return self.__repr__()

    def evaluate(self, objectivefn):
        self.fitness = objectivefn(self.bitstring)

    def singlePointCrossover(self, other):
        c = Chromosome(self.bitstring[0:-1] + other.bitstring[1:])
        c2 = Chromosome(other.bitstring[0:-1] + self.bitstring[1:])
        return [c, c2]

    def mutate(self, prob):
        if random.random() < prob:
            index = random.randint(0, len(self.bitstring)-1)
            self.bitstring[index] = random.uniform(-5.12, 5.12)

class AlgoritmoGenetico:
    def __init__(self, popsize=100, iterations=50, elitism=0.8, stringlen=2, mutRate=0.4, problemParams=None):
        self.popsize = popsize
        self.iterations = iterations
        self.fractionKept = elitism
        self.inputLen = stringlen
        self.mutationRate = mutRate
        self.poplist = []

    def makeInitialPopulation(self):
        self.poplist = [Chromosome([random.uniform(-5.12, 5.12)
                                    for i in range(0, self.inputLen)])
                        for j in range(0, self.popsize)]

    def assignFitness(self):
        for i in range(0,(self.popsize)):
            self.poplist[i].fitness = self.objective(self.poplist[i].bitstring[0],self.poplist[i].bitstring[1])

    def objective(self,x,y):
        return 20 + x**2 + y**2 -10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

    def chooseChromosome(self):
        c1 = random.choice(self.poplist)
        c2 = random.choice(self.poplist)
        if c1.fitness <= c2.fitness:
            return c1
        else:
            return c2

    def makeChildren(self):
        p1 = self.chooseChromosome()
        p2 = self.chooseChromosome()
        kids = p1.singlePointCrossover(p2)
        kids[0].mutate(self.mutationRate)
        kids[1].mutate(self.mutationRate)
        return kids

    def runAG(self):
        self.makeInitialPopulation()

        for i in range(1, self.iterations+1):
            ### Gera o fitness de cada elemento da populacao
            self.assignFitness()
            print("Geracao: " + str(i))
            print('population', self.poplist)
            ### Encontra a melhor solucao da geracao atual
            best = min(self.poplist)
            #print ('Melhor solucao atual: ' + best.printSolution() + ", Fitness: " +str(best.fitness))
            print(str(best.fitness))
            newpop = [best]
            ### Geracao de novos individuos
            for j in range(1, int((1.0 - self.fractionKept) * self.popsize)):
                newpop.extend(self.makeChildren())

            ### Seleciona os novos individuos
            while len(newpop) < self.popsize:
                newpop.append(self.chooseChromosome())
            self.poplist = newpop
        ### Apresenta a melhor solucao final
        self.assignFitness()
        best = min(self.poplist)
        print('Melhor solucao final: ' + best.printSolution() + ", Fitness: " +str(best.fitness))
        print(str(best.fitness))

if __name__ == '__main__':
    AG = AlgoritmoGenetico()
    AG.runAG()