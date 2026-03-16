function signedRand() {
	return (Math.random() * 2) - 1;
}

function sigmoid(x: number): number {
	return 1 / (1 + Math.exp(-x));
}

function ReLU(x: number) {
	return Math.max(x, 0);
}

function gaussianRandom(mean: number, deviation: number) {
	let u = 1 - Math.random();
	let v = Math.random();
	let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
	return z * deviation + mean;
}

enum ActivationFunction {
	SIGMOID,
	TANH,
	RELU,
	NONE,
}

enum NodeLayer {
	INPUT,
	HIDDEN,
	OUTPUT
}

class NeuralNode {
	id: number;
	layer: NodeLayer;
	activation: ActivationFunction;
	bias: number;
	constructor(id: number, layer: NodeLayer, activation: ActivationFunction, bias: number) {
		this.id = id;
		this.layer = layer;
		this.activation = activation;
		this.bias = bias;
	}

	copy() {
		return new NeuralNode(this.id, this.layer, this.activation, this.bias);
	}
}

class NeuralConnection {
	inNode: number;
	outNode: number;
	weight: number;
	enabled: boolean;
	innovation: number;
	constructor(inNode: number, outNode: number, weight: number, innovation: number, enabled: boolean) {
		this.inNode = inNode;
		this.outNode = outNode;
		this.weight = weight;
		this.innovation = innovation;
		this.enabled = enabled;
	}

	copy(): NeuralConnection {
		return new NeuralConnection(this.inNode, this.outNode, this.weight, this.innovation, this.enabled);
	}
}

export class NeuralNetwork {
	nodes: Map<number, NeuralNode> = new Map();
	connections: NeuralConnection[];

	fitness: number = 0;

	constructor(nodes: NeuralNode[], connections: NeuralConnection[]) {
		this.connections = [];

		for (let connection of connections) {
			this.connections.push(connection.copy());
		}

		for (let node of nodes) {
			this.nodes.set(node.id, node);
		}
	}

	topologicalSort(edges: Map<NeuralNode, NeuralNode[]>) {
		let visited = new Set<NeuralNode>();
		let order: NeuralNode[] = [];

		function visit(node: NeuralNode) {
			if (visited.has(node)) {
				return;
			}

			visited.add(node);

			let connectedNodes = edges.get(node);

			if (!connectedNodes) {
				return;
			}

			for (let connectedNode of connectedNodes) {
				visit(connectedNode);
			}

			order.push(node);
		}

		for (let node of edges.keys()) {
			visit(node);
		}

		return order.reverse();
	}

	evaluate(inputValues: number[]): number[] {
		let nodeValues = new Map<number, number>();
		let nodeInputs = new Map<number, NeuralConnection[]>();

		let inputNodes = [];
		let outputNodes = [];

		for (let nodeId of this.nodes.keys()) {
			let node = this.nodes.get(nodeId);

			if (node) {
				if (node.layer == NodeLayer.INPUT) {
					inputNodes.push(node);
				} else if (node.layer == NodeLayer.OUTPUT) {
					outputNodes.push(node);
				}
				nodeInputs.set(nodeId, []);
			}
		}

		for (let i = 0; i < inputNodes.length; i++) {
			nodeValues.set(inputNodes[i].id, inputValues[i]);
		}

		let edges = new Map<NeuralNode, NeuralNode[]>();

		for (let node of this.nodes.values()) {
			edges.set(node, []);
		}

		for (let connection of this.connections) {
			if (connection.enabled) {
				const inputNode = this.nodes.get(connection.inNode);
				const outputNode = this.nodes.get(connection.outNode);

				if (!inputNode || !outputNode) {
					continue;
				}

				edges.get(inputNode)?.push(outputNode);
				nodeInputs.get(connection.outNode)?.push(connection);
			}
		}

		const sortedNodes = this.topologicalSort(edges);

		for (let node of sortedNodes) {
			if (nodeValues.has(node.id)) {
				continue;
			}

			let incoming = nodeInputs.get(node.id);

			if (!incoming) {
				continue;
			}

			let totalInput = node.bias;

			for (let connection of incoming) {
				let inputValue = nodeValues.get(connection.inNode);

				if (inputValue !== undefined) {
					totalInput += inputValue * connection.weight;
				}
			}

			if (node.activation == ActivationFunction.SIGMOID) {
				nodeValues.set(node.id, sigmoid(totalInput));
			} else if (node.activation == ActivationFunction.RELU) {
				nodeValues.set(node.id, ReLU(totalInput));
			} else if (node.activation == ActivationFunction.NONE) {
				nodeValues.set(node.id, totalInput);
			}
		}

		let networkOutputs: number[] = [];

		for (let node of outputNodes) {
			networkOutputs.push(nodeValues.get(node.id) ?? 0);
		}

		return networkOutputs;
	}

	distanceTo(network2: NeuralNetwork, c1 = 1.0, c2 = 1.0, c3 = 0.4) {
		let genes1 = new Map<number, NeuralConnection>();
		let genes2 = new Map<number, NeuralConnection>();

		for (let connection of this.connections) {
			genes1.set(connection.innovation, connection);
		}

		for (let connection of network2.connections) {
			genes2.set(connection.innovation, connection);
		}

		let innovations1 = new Set(genes1.keys());
		let innovations2 = new Set(genes2.keys());

		let matching = innovations1.intersection(innovations2);
		let disjoint = innovations1.symmetricDifference(innovations2);

		let excess = new Set<number>();

		let maxInnovations1 = Math.max(...innovations1) ?? 0;
		let maxInnovations2 = Math.max(...innovations2) ?? 0;
		let maxInnovation = Math.min(maxInnovations1, maxInnovations2);

		let oldDisjoint = new Set([...disjoint]);

		for (let innovation of oldDisjoint) {
			if (innovation > maxInnovation) {
				excess.add(innovation);
				disjoint.delete(innovation);
			}
		}

		let averageWeightDifference = 0;

		if (matching.size != 0) {
			let weightDifference = 0;

			for (let match of matching) {
				const weight1 = genes1.get(match)?.weight ?? 0;
				const weight2 = genes2.get(match)?.weight ?? 0;
				weightDifference += Math.abs(weight1 - weight2);
			}

			averageWeightDifference = weightDifference / matching.size;
		}

		let N = Math.max(this.connections.length, network2.connections.length);

		if (N < 20) {
			N = 1;
		}

		const difference = (c1 * excess.size) / N + (c2 * disjoint.size) / N + c3 * averageWeightDifference;

		return difference;
	}

	nodePathExists(startingNodeId: number, endingNodeId: number, checkedNodes: Set<number> | undefined): boolean {
		if (!checkedNodes) {
			checkedNodes = new Set<number>();
		}

		if (startingNodeId == endingNodeId) {
			return true;
		}

		checkedNodes.add(startingNodeId);

		for (let connection of this.connections) {
			if (connection.enabled && connection.inNode == startingNodeId) {
				if (!checkedNodes.has(connection.outNode)) {
					if (this.nodePathExists(connection.outNode, endingNodeId, checkedNodes)) {
						return true;
					}
				}
			}
		}

		return false;
	}

	addConnectionMutate(innovationTracker: InnovationTracker): void {
		let nodeSet = new Set(this.nodes.values());

		const maxTries = 10;

		for (let i = 0; i < maxTries; i++) {
			const nodeArray = [...nodeSet];

			let node1 = nodeArray[Math.floor(Math.random() * nodeSet.size)];
			let node2;

			do {
				node2 = nodeArray[Math.floor(Math.random() * nodeSet.size)];
			} while (node1 == node2);

			if (node1.layer == NodeLayer.OUTPUT || (node1.layer == NodeLayer.HIDDEN && node2.layer == NodeLayer.INPUT)) {
				[node1, node2] = [node2, node1];
			}

			if (node1 == node2) {
				continue;
			}

			if (node1.layer == NodeLayer.OUTPUT || node2.layer == NodeLayer.INPUT) {
				continue;
			}

			let connectionExists = false;

			for (let connection of this.connections) {
				if ((connection.inNode == node1.id && connection.outNode == node2.id) || (connection.inNode == node2.id && connection.outNode == node1.id)) {
					connectionExists = true;
					break;
				}
			}

			if (connectionExists) {
				continue;
			}

			if (this.nodePathExists(node2.id, node1.id, undefined)) {
				continue;
			}

			let innovationNumber = innovationTracker.getConnectionInnovation(node1.id, node2.id);
			let newConnection = new NeuralConnection(node1.id, node2.id, signedRand(), innovationNumber, true);

			this.connections.push(newConnection);
			return;
		}
	}

	addNodeMutate(innovationTracker: InnovationTracker) {
		let enabledConnections = new Set<NeuralConnection>();

		for (let connection of this.connections) {
			if (connection.enabled) {
				enabledConnections.add(connection);
			}
		}

		if (enabledConnections.size == 0) {
			return;
		}

		let randomConnection = [...enabledConnections][Math.floor(Math.random() * enabledConnections.size)];
		randomConnection.enabled = false;

		let { nodeId, connection1Innovation, connection2Innovation } = innovationTracker.getNodeInnovation(randomConnection.innovation);

		let newNode = new NeuralNode(nodeId, NodeLayer.HIDDEN, ActivationFunction.SIGMOID, signedRand());

		let connection1 = new NeuralConnection(randomConnection.inNode, nodeId, 1, connection1Innovation, true);
		let connection2 = new NeuralConnection(nodeId, randomConnection.outNode, randomConnection.weight, connection2Innovation, true);

		this.nodes.set(nodeId, newNode);
		this.connections.push(connection1, connection2);
	}

	mutateWeights(rate = 0.8, power = 0.5) {
		for (let connection of this.connections) {
			if (Math.random() < rate) {
				if (Math.random() < 0.1) {
					connection.weight = signedRand();
				} else {
					connection.weight += gaussianRandom(0, power);
					connection.weight = Math.max(-5, Math.min(5, connection.weight));
				}
			}
		}
	}

	mutateBias(rate = 0.7, power = 0.5) {
		for (let node of this.nodes.values()) {
			if (node.layer != NodeLayer.INPUT && Math.random() < rate) {
				if (Math.random() < 0.1) {
					node.bias = signedRand();
				} else {
					node.bias += gaussianRandom(0, power);
					node.bias = Math.max(-5, Math.min(5, node.bias));
				}
			}
		}
	}

	mutate(innovationTracker: InnovationTracker, connectionMutationRate = 0.05, nodeMutationRate = 0.03, weightMutationRate = 0.8, biasMutationRate = 0.7) {
		this.mutateWeights(weightMutationRate);
		this.mutateBias(biasMutationRate);

		if (Math.random() < connectionMutationRate) {
			this.addConnectionMutate(innovationTracker);
		}

		if (Math.random() < nodeMutationRate) {
			this.addNodeMutate(innovationTracker);
		}
	}
}


class NodePair {
	inputNode: number;
	outputNode: number;

	constructor(input: number, output: number) {
		this.inputNode = input;
		this.outputNode = output;
	}

	toString(): string {
		return `${this.inputNode}<${this.outputNode}`;
	}
}

type InnovationTriplet = {
	nodeId: number;
	connection1Innovation: number;
	connection2Innovation: number;
}

class InnovationTracker {
	currentInnovation: number = 0;
	connectionInnovations: Map<string, number> = new Map();
	nodeInnovations: Map<number, InnovationTriplet> = new Map();
	nodeIdCounter: number = 0;

	getConnectionInnovation(inputNodeId: number, outputNodeId: number): number {
		let pair = new NodePair(inputNodeId, outputNodeId);
		let pairString = pair.toString();

		if (!this.connectionInnovations.has(pairString)) {
			this.connectionInnovations.set(pairString, this.currentInnovation++);
		}

		return this.connectionInnovations.get(pairString) ?? 0;
	}

	getNodeInnovation(connectionInnovation: number): InnovationTriplet {
		if (!this.nodeInnovations.has(connectionInnovation)) {
			const nodeId = this.nodeIdCounter++;

			const connection1Innovation = this.currentInnovation++;
			const connection2Innovation = this.currentInnovation++;

			const innovationTriplet: InnovationTriplet = {
				nodeId,
				connection1Innovation,
				connection2Innovation
			};

			this.nodeInnovations.set(connectionInnovation, innovationTriplet);
		}

		let triplet = this.nodeInnovations.get(connectionInnovation);

		if (triplet) {
			return triplet;
		} else {
			return {
				nodeId: 0,
				connection1Innovation: 0,
				connection2Innovation: 0
			};
		}
	}
}

class Species {
	representative: NeuralNetwork;
	members: NeuralNetwork[];

	adjustedFitness: number = 0
	bestFitness: number = -Infinity;
	stagnantGenerations: number = 0

	constructor(representative: NeuralNetwork) {
		this.representative = representative;
		this.members = [representative];
	}

	addMember(network: NeuralNetwork) {
		this.members.push(network);
	}

	clearMembers() {
		this.members = [];
	}

	updateFitnessStats() {
		if (this.members.length == 0) {
			this.adjustedFitness = 0;
			return;
		}

		let currentBestFitness = 0;
		let totalFitness = 0;

		for (let member of this.members) {
			currentBestFitness = Math.max(member.fitness, currentBestFitness);
			totalFitness += member.fitness;
		}

		if (currentBestFitness > this.bestFitness) {
			this.bestFitness = currentBestFitness;
			this.stagnantGenerations = 0;
		} else {
			this.stagnantGenerations++;
		}

		this.adjustedFitness = totalFitness / this.members.length;
	}
}

class Speciator {
	species: Species[] = [];

	compatibilityThreshold: number = 3.0;

	constructor(compatibilityThreshold = 3.0) {
		this.compatibilityThreshold = compatibilityThreshold;
	}

	speciate(population: NeuralNetwork[]) {
		for (let species of this.species) {
			species.clearMembers();
		}

		for (let network of population) {
			let foundSpecies = false;

			for (let species of this.species) {
				if (network.distanceTo(species.representative) < this.compatibilityThreshold) {
					species.addMember(network);
					foundSpecies = true;
					break;
				}
			}

			if (!foundSpecies) {
				const newSpecies = new Species(network);
				this.species.push(newSpecies);
			}
		}

		this.species = this.species.filter(species => species.members.length > 0);

		for (let species of this.species) {
			species.updateFitnessStats();

			// Get member with most fitness (epic hacker js)
			species.representative = species.members.reduce((acc, current) => acc.fitness > current.fitness ? acc : current);
		}
	}
}

export class NEAT {
	population: NeuralNetwork[] = [];
	innovationTracker: InnovationTracker = new InnovationTracker();
	speciator: Speciator = new Speciator();
	fitnessFunction: (network: NeuralNetwork) => number

	constructor(numberOfInputs: number, numberOfOutputs: number, size: number, fitnessFunction: (network: NeuralNetwork) => number) {
		this.fitnessFunction = fitnessFunction;

		for (let i = 0; i < size; i++) {
			let inputs: NeuralNode[] = [];
			let outputs: NeuralNode[] = [];
			let connections: NeuralConnection[] = [];

			for (let i = 0; i < numberOfInputs; i++) {
				inputs.push(new NeuralNode(i, NodeLayer.INPUT, ActivationFunction.NONE, 0));
			}

			for (let i = 0; i < numberOfOutputs; i++) {
				outputs.push(new NeuralNode(i + numberOfInputs, NodeLayer.OUTPUT, ActivationFunction.SIGMOID, signedRand()));
			}

			this.innovationTracker.nodeIdCounter = numberOfInputs + numberOfOutputs;

			for (let i = 0; i < numberOfInputs; i++) {
				for (let j = 0; j < numberOfOutputs; j++) {
					const inputNodeId = i;
					const outputNodeId = numberOfInputs + j;

					const innovation = this.innovationTracker.getConnectionInnovation(inputNodeId, outputNodeId);

					const connection = new NeuralConnection(inputNodeId, outputNodeId, signedRand(), innovation, true);

					connections.push(connection);
				}
			}

			let network = new NeuralNetwork(inputs.concat(outputs), connections);

			this.population.push(network);
		}
	}

	crossover(network1: NeuralNetwork, network2: NeuralNetwork): NeuralNetwork {
		// important: network1 is always the fitter one

		let offspringConnections = [];
		let offspringNodes = new Set<NeuralNode>();
		let allNodes = new Map<Number, NeuralNode>();

		for (let node of network1.nodes.values()) {
			const nodeCopy = node.copy();
			allNodes.set(node.id, nodeCopy);
			offspringNodes.add(nodeCopy);
		}

		for (let node of network2.nodes.values()) {
			if (!allNodes.has(node.id)) {
				allNodes.set(node.id, node.copy());
			}
		}

		let genes1 = new Map<number, NeuralConnection>();
		let genes2 = new Map<number, NeuralConnection>();

		for (let gene of network1.connections) {
			genes1.set(gene.innovation, gene);
		}

		for (let gene of network2.connections) {
			genes2.set(gene.innovation, gene);
		}

		let allInnovation = new Set(genes1.keys()).union(new Set(genes2.keys()));

		let sortedInnovations = Array.from(allInnovation).sort((a, b) => a - b);

		let geneCopy;

		for (let innovation of sortedInnovations) {
			let gene1 = genes1.get(innovation);
			let gene2 = genes2.get(innovation);

			if (gene1 && gene2) {
				let selected = [gene1, gene2][Math.floor(Math.random() * 2)];
				geneCopy = selected.copy();

				if (!gene1.enabled || !gene2.enabled) {
					if (Math.random() < 0.75) {
						geneCopy.enabled = false;
					}
				}
			} else if (gene1 && !gene2) {
				geneCopy = gene1.copy();
			} else {
				continue;
			}

			let inputNode = allNodes.get(geneCopy.inNode);
			let outputNode = allNodes.get(geneCopy.outNode);

			if (inputNode && outputNode) {
				offspringConnections.push(geneCopy);
				offspringNodes.add(inputNode);
				offspringNodes.add(outputNode);
			}
		}

		return new NeuralNetwork(Array.from(offspringNodes), Array.from(offspringConnections));
	}

	evolvePopulation(fitnessScores: number[], stagnationLimit = 15) {
		let newPopulation: NeuralNetwork[] = [];

		for (let i = 0; i < fitnessScores.length; i++) {
			this.population[i].fitness = fitnessScores[i];
		}

		this.speciator.speciate(this.population);

		let speciesList = this.speciator.species;
		speciesList.sort((a, b) => b.bestFitness - a.bestFitness);

		let survivingSpecies = [];

		if (speciesList.length != 0) {
			survivingSpecies.push(speciesList[0]);
		}

		for (let species of speciesList.slice(1)) {
			if (species.stagnantGenerations < stagnationLimit) {
				survivingSpecies.push(species);
			}
		}

		speciesList = survivingSpecies;

		let totalAdjustedFitness = 0;

		for (let species of speciesList) {
			totalAdjustedFitness += species.adjustedFitness;

			if (species.members.length != 0) {
				newPopulation.push(species.members.reduce((acc, curr) => acc.fitness > curr.fitness ? acc : curr));
			}
		}

		let remainingOffspring = this.population.length - newPopulation.length;

		for (let species of speciesList) {
			let offspringCount;

			if (totalAdjustedFitness > 0) {
				offspringCount = Math.trunc((species.adjustedFitness / totalAdjustedFitness) * remainingOffspring);
			} else {
				offspringCount = Math.floor(remainingOffspring / speciesList.length);
			}

			if (offspringCount > 0) {
				let offspring = this.reproduceSpecies(species, offspringCount);
				newPopulation.push(...offspring);
			}
		}

		while (newPopulation.length < this.population.length) {
			let bestSpecies = speciesList.reduce((acc, curr) => acc.adjustedFitness > curr.adjustedFitness ? acc : curr);
			let offspring = this.reproduceSpecies(bestSpecies, 1);
			newPopulation.push(...offspring);
		}

		this.population = newPopulation;
	}

	reproduceSpecies(species: Species, offspringCount: number) {
		let offspring: NeuralNetwork[] = [];

		if (species.members.length == 1) {
			for (let i = 0; i < offspringCount; i++) {
				let nodes = [];
				let connectors = [];

				for (let node of species.members[0].nodes.values()) {
					nodes.push(node.copy());
				}

				for (let connector of species.members[0].connections.values()) {
					connectors.push(connector.copy());
				}

				let child = new NeuralNetwork(nodes, connectors);

				child.mutate(this.innovationTracker);

				offspring.push(child);
			}
		} else {
			for (let i = 0; i < offspringCount; i++) {
				let speciesSize = species.members.length;

				let parent1 = species.members[Math.floor(Math.random() * speciesSize)];
				let parent2;

				do {
					parent2 = species.members[Math.floor(Math.random() * speciesSize)]
				} while (parent1 == parent2);

				if (parent1.fitness < parent2.fitness) {
					[parent1, parent2] = [parent2, parent1];
				}

				let child = this.crossover(parent1, parent2);

				child.mutate(this.innovationTracker);

				offspring.push(child);
			}
		}

		return offspring;
	}

	evolve() {
		let overallBestFitness = 0;

		for (let generation = 0; generation < 15000; generation++) {
			let fitnessScores = this.population.map(el => this.fitnessFunction(el));

			let bestFitness = Math.max(...fitnessScores);
			overallBestFitness = Math.max(overallBestFitness, bestFitness);
			let averageFitness = fitnessScores.reduce((acc, curr) => acc + curr, 0) / fitnessScores.length;

			console.log(`Generation ${generation}: Best=${bestFitness.toFixed(3)}, Avg=${averageFitness.toFixed(3)}, Species=${this.speciator.species.length}`);

			if (bestFitness >= 3.9 || generation == 14999) {
				console.log(`Problem solved in ${generation} generations!`);
				console.log(`Best fitness achieved was ${bestFitness}`);

				let best = fitnessScores.indexOf(bestFitness);

				console.log(this.population[best]);

				return;
			}

			this.evolvePopulation(fitnessScores);
		}

		console.log("Couldn't solve xor in 50 generations.");
		console.log(`The overall best fitness was ${overallBestFitness}`);
	}
}
