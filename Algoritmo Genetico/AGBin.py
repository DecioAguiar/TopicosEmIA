import random
import struct
import numpy as np

def vec_to_str(v):
	str_bin = ''
	for i in range(0, len(v)):
		str_bin += str(v[i])
	return str_bin

def float_to_bin2(f):
    """ Convert a float into a binary string. """
    ba = struct.pack('>d', f)
    ba = bytearray(ba)  # convert string to bytearray - not needed in py3
    s = ''.join('{:08b}'.format(b) for b in ba)
    str_bin = s[:-1].lstrip('0') + s[0] # strip all leading zeros except for last
    x = []
    for i in range(0,len(str_bin)):
    	x.append(int(str_bin[i]))
    return x

def int_to_bytes(n, minlen=0):  # helper function
    """ Int/long to byte string. """
    nbits = n.bit_length() + (1 if n < 0 else 0)  # plus one for any sign bit
    nbytes = (nbits+7) // 8  # number of whole bytes
    b = bytearray()
    for _ in range(nbytes):
        b.append(n & 0xff)
        n >>= 8
    if minlen and len(b) < minlen:  # zero pad?
        b.extend([0] * (minlen-len(b)))
    return bytearray(reversed(b))  # high bytes first

def float_to_bin(f):
    """ Convert a float into a binary string. """
    ba = struct.pack('>d', f)
    ba = bytearray(ba)  # convert string to bytearray - not needed in py3
    s = ''.join('{:08b}'.format(b) for b in ba)
    return s[:-1].lstrip('0') + s[0] # strip all leading zeros except for last

def bin_to_float(b):
    bf = int_to_bytes(int(b, 2), 8)
    return struct.unpack('>d', bf)[0]

class Chromosome:
    def __init__(self, bitstring, decimal):
        self.bitstring = bitstring
        self.decimal = decimal
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
        #return ' '.join([str(i) for i in self.bitstring])
        return str(self.fitness)

    def printSolution(self):
        return self.__repr__()

    def evaluate(self, objectivefn):
        self.fitness = objectivefn(self.bitstring)

    def singlePointCrossover(self, other):
        novos = []
        pos = random.randint(0, len(self.bitstring))
        str1 = vec_to_str(self.bitstring[0:pos] + other.bitstring[pos:])
        str2 = vec_to_str(other.bitstring[0:pos] + self.bitstring[pos:])
        d = bin_to_float(str1)
        d2 = bin_to_float(str2)
        if( (d >= -5.12) and (d <= 5.12) ):
            c = Chromosome(self.bitstring[0:pos] + other.bitstring[pos:],d)
            novos.append(c)
        if( (d2 >= -5.12) and (d2 <= 5.12) ):
            c2 = Chromosome(other.bitstring[0:pos] + self.bitstring[pos:],d2)
            novos.append(c2)
        return novos


    def mutate(self, prob):
        if random.random() < prob:
            index = random.randint(0, len(self.bitstring) - 1)
            self.bitstring[index] ^= 1
            str_bin = vec_to_str(self.bitstring)
            self.decimal = bin_to_float(str_bin)

class AlgoritmoGenetico:
    def __init__(self, popsize=20, iterations=10, elitism=0.8, stringlen=64, mutRate=0.4, problemParams=None):
        self.popsize = popsize
        self.iterations = iterations
        self.fractionKept = elitism
        self.inputLen = stringlen
        self.mutationRate = mutRate
        self.poplist = []
        self.poplist2 = []

    def makeInitialPopulation(self):
        for i in range(0, self.popsize):
            decimal = random.uniform(-5.12, 5.12)
            str_bin = float_to_bin2(decimal)
            self.poplist.append(Chromosome(str_bin, decimal))

        for j in range(0, self.popsize):
            decimal = random.uniform(-5.12, 5.12)
            str_bin = float_to_bin2(decimal)
            self.poplist2.append(Chromosome(str_bin, decimal))

    def assignFitness(self):
        for i in range(0,(self.popsize)):
            self.poplist[i].fitness = self.objective2(self.poplist[i].decimal,self.poplist[i].decimal)
            self.poplist2[i].fitness = self.objective2(self.poplist2[i].decimal,self.poplist[i].decimal)

    def objective(self,x):
        return 2*x-(20*np.pi*np.sin(2*np.pi*x))

    def objective2(self,x,y):
        return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

    def chooseChromosome(self):
        new = random.sample(self.poplist,2)
        c1 = new[0]
        c2 = new[1]
        if c1.fitness <= c2.fitness:
            return c1
        else:
            return c2

    def chooseChromosome2(self):
        new = random.sample(self.poplist2, 2)
        c1 = new[0]
        c2 = new[1]
        if c1.fitness <= c2.fitness:
            return c1
        else:
            return c2

    def makeChildren(self):
        p1 = self.chooseChromosome()
        p2 = self.chooseChromosome()
        kids = p1.singlePointCrossover(p2)
        for i in range(0, len(kids)):
            kids[i].mutate(self.mutationRate)
        return kids

    def makeChildren2(self):
        p1 = self.chooseChromosome2()
        p2 = self.chooseChromosome2()
        kids = p1.singlePointCrossover(p2)
        for i in range(0, len(kids)):
            kids[i].mutate(self.mutationRate)
        return kids

    def runAG(self):
        self.makeInitialPopulation()

        for i in range(1, self.iterations+1):
            ### Gera o fitness de cada elemento da populacao
            self.assignFitness()
            print("Geracao: " + str(i))
            print('population', self.poplist)
            print('population2', self.poplist2)
            ### Encontra a melhor solucao da geracao atual
            best = min(self.poplist)
            best2 = min(self.poplist2)
            solucao = self.objective2(best.decimal, best2.decimal)
            print ('Melhor solucao atual: ' +str(solucao))
            #print(str(solucao))

            newpop = [best]
            newpop2 = [best2]
            ### Geracao de novos individuos
            for j in range(1, int((1.0 - self.fractionKept) * self.popsize)):
                newpop.extend(self.makeChildren())

            for j in range(1, int((1.0 - self.fractionKept) * self.popsize)):
                newpop2.extend(self.makeChildren2())

            ### Seleciona os novos individuos
            while len(newpop) < self.popsize:
                newpop.append(self.chooseChromosome())
            self.poplist = newpop
            while len(newpop2) < self.popsize:
                newpop2.append(self.chooseChromosome2())
            self.poplist2 = newpop2

        ### Apresenta a melhor solucao final
        self.assignFitness()
        best = min(self.poplist)
        best2 = min(self.poplist2)
        solucao = self.objective2(best.decimal, best2.decimal)
        print('Melhor solucao atual: ' + str(solucao))
        #print(str(solucao))
if __name__ == '__main__':
    AG = AlgoritmoGenetico()
    AG.runAG()
