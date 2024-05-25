# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        z = sum(self.values())
        if z != 0:
            for key in self:
                self[key] /= z

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        # Si no hay distancia ruidosa registrada:
        if noisyDistance == None:
            # Si la posición del fantasma coincide con la posición de la cárcel:
            if ghostPosition == jailPosition:
                # Devuelve 1 (probabilidad alta de observar al fantasma en la cárcel).
                return 1
            else:
                # De lo contrario, devuelve 0 (probabilidad baja de observar al fantasma en cualquier otro lugar).
                return 0

        # Si la posición del fantasma coincide con la posición de la cárcel:
        if ghostPosition == jailPosition:
            # Si no hay distancia ruidosa registrada:
            if noisyDistance == None:
                # Devuelve 1 (probabilidad alta de observar al fantasma en la cárcel).
                return 1
            else:
                # De lo contrario, devuelve 0 (probabilidad baja de observar al fantasma en cualquier otro lugar).
                return 0

        # Calcula la distancia real entre la posición de Pacman y la posición del fantasma.
        trueDistance = manhattanDistance(pacmanPosition, ghostPosition)

        # Calcula la probabilidad de observación basada en la distancia ruidosa y la distancia real.
        observationProb = busters.getObservationProbability(noisyDistance, trueDistance)

        # Retorna la probabilidad de observación calculada.
        return observationProb


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()

        for position in self.allPositions:
            observationProb = self.getObservationProb(observation, pacmanPosition, position, jailPosition)
            self.beliefs[position] *= observationProb

        self.beliefs.normalize()
        
    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        newBeliefs = DiscreteDistribution()
        for oldPos in self.allPositions:
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            for newPos, prob in newPosDist.items():
                newBeliefs[newPos] += self.beliefs[oldPos] * prob
        self.beliefs = newBeliefs
        self.beliefs.normalize()


    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # Calcula el número de posiciones legales disponibles en el juego.
        numLegalPositions = len(self.legalPositions)

        # Itera sobre el número de partículas (representaciones de posibles ubicaciones de los fantasmas).
        for i in range(self.numParticles):
            # Calcula el índice de la partícula en función del número de posiciones legales disponibles.
            particleIndex = i % numLegalPositions
            # Añade una nueva partícula (posición legal) a la lista de partículas.
            self.particles.append(self.legalPositions[particleIndex])


    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # Obtiene la posición actual de Pacman en el juego.
        pacmanPosition = gameState.getPacmanPosition()

        # Obtiene la posición de la cárcel en el juego.
        jailPosition = self.getJailPosition()

        # Inicializa una distribución de peso discreta para las partículas.
        weightDistribution = DiscreteDistribution()

        # Itera sobre cada partícula en la lista de partículas.
        for particle in self.particles:
            # Si la partícula no está en la distribución de peso, la agrega con peso 0.
            if particle not in weightDistribution:
                weightDistribution[particle] = 0
            # Calcula el peso de la partícula basado en la observación actual.
            weightDistribution[particle] += self.getObservationProb(observation, pacmanPosition, particle, jailPosition)

        # Normaliza la distribución de pesos para que la suma de los pesos sea 1.
        weightDistribution.normalize()

        # Si la suma total de pesos es 0, indica que la distribución de pesos no es válida y se reinicializa uniformemente.
        if weightDistribution.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Si la distribución de pesos es válida, se generan nuevas partículas utilizando muestreo ponderado.
            newParticles = []
            for i in range(len(self.particles)):
                # Selecciona una muestra de la distribución de pesos.
                sample = weightDistribution.sample()
                # Agrega la muestra a la lista de nuevas partículas.
                newParticles.append(sample)
            # Actualiza la lista de partículas con las nuevas partículas generadas.
            self.particles = newParticles


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # Inicializa una lista vacía para almacenar las nuevas partículas.
        newParticles = []

        # Itera sobre cada partícula en la lista de partículas existente.
        for particle in self.particles:
            # Obtiene la distribución de posición para la partícula actual en función del estado del juego.
            newPosDist = self.getPositionDistribution(gameState, particle)
            # Realiza un muestreo de la distribución de posición para obtener una nueva posición.
            sample = newPosDist.sample()
            # Agrega la nueva posición a la lista de nuevas partículas.
            newParticles.append(sample)

        # Actualiza la lista de partículas con las nuevas partículas generadas.
        self.particles = newParticles


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        # Inicializa una distribución discreta para contar la frecuencia de cada partícula.
        dist = DiscreteDistribution()

        # Itera sobre cada partícula en la lista de partículas.
        for particle in self.particles:
            # Incrementa el conteo de la frecuencia de la partícula en la distribución.
            dist[particle] += 1

        # Normaliza la distribución para que la suma de las probabilidades sea 1.
        dist.normalize()

        # Retorna la distribución de frecuencia normalizada.
        return dist



class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # Inicializa una lista para almacenar las posiciones de los fantasmas.
        ghostPosLists = []

        # Para cada fantasma en el juego, añade la lista de posiciones legales a ghostPosLists.
        for i in range(self.numGhosts):
            ghostPosLists.append(self.legalPositions)

        # Calcula el producto cartesiano de todas las posiciones posibles para los fantasmas.
        cartesianProduct = itertools.product(*ghostPosLists)

        # Convierte el objeto iterador en una lista y luego mezcla aleatoriamente el orden de las combinaciones.
        cartesianProduct = list(cartesianProduct)
        random.shuffle(cartesianProduct)

        # Itera sobre el número de partículas deseado.
        for i in range(self.numParticles):
            # Calcula el índice de la lista a partir del índice actual y la longitud de cartesianProduct.
            listsIndex = i % len(cartesianProduct)
            # Añade una combinación de posiciones de fantasmas a la lista de partículas.
            self.particles.append(cartesianProduct[listsIndex])


    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # Obtiene la posición actual de Pacman en el juego.
        pacmanPosition = gameState.getPacmanPosition()

        # Inicializa una distribución de peso discreta para las partículas.
        weightDistribution = DiscreteDistribution()

        # Itera sobre cada partícula en la lista de partículas.
        for particle in self.particles:
            # Si la partícula no está en la distribución de peso, la agrega con peso 0.
            if particle not in weightDistribution:
                weightDistribution[particle] = 0
            
            # Inicializa un producto que almacenará la probabilidad de observación conjunta para cada fantasma.
            product = 1
            # Calcula la probabilidad de observación conjunta para cada fantasma en la partícula.
            for i in range(self.numGhosts):
                product *= self.getObservationProb(observation[i], pacmanPosition, particle[i], self.getJailPosition(i))
            
            # Añade el producto al peso correspondiente de la partícula en la distribución de pesos.
            weightDistribution[particle] += product

        # Normaliza la distribución de pesos para que la suma de los pesos sea 1.
        weightDistribution.normalize()

        # Si la suma total de pesos es 0, indica que la distribución de pesos no es válida y se reinicializa uniformemente.
        if weightDistribution.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # Si la distribución de pesos es válida, se generan nuevas partículas utilizando muestreo ponderado.
            newParticles = []
            for i in range(len(self.particles)):
                # Selecciona una muestra de la distribución de pesos.
                sample = weightDistribution.sample()
                # Agrega la muestra a la lista de nuevas partículas.
                newParticles.append(sample)
            # Actualiza la lista de partículas con las nuevas partículas generadas.
            self.particles = newParticles

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            # Itera sobre cada fantasma en el juego.
            for i in range(self.numGhosts):
                # Obtiene la distribución de probabilidad de la nueva posición del fantasma i,
                # dada la información actual del juego y la posición anterior del fantasma en la partícula.
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                # Realiza un muestreo de la distribución de probabilidad para obtener una nueva posición para el fantasma i.
                sample = newPosDist.sample()
                # Almacena la nueva posición del fantasma i en la nueva partícula.
                newParticle[i] = sample

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
