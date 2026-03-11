function signedRand() {
	return (Math.random() * 2) - 1;
}

function sigmoid(x: number): number {
	return 1 / (1 + Math.pow(Math.E, -x));
}

enum ActivationFunction {
	SIGMOID,
	TANH,
	NONE
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
}

class NeuralNetwork {
	nodes: Map<number, NeuralNode> = new Map();
	connections: NeuralConnection[];

	fitness: number = 0;

	constructor(nodes: NeuralNode[], connections: NeuralConnection[]) {
		this.connections = connections;

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
			if (nodeValues.get(node.id)) {
				continue;
			}

			let incoming = nodeInputs.get(node.id);

			if (!incoming) {
				continue;
			}

			let totalInput = node.bias;

			for (let connection of incoming) {
				let inputValue = nodeValues.get(connection.inNode);

				if (inputValue) {
					totalInput += inputValue * connection.weight;
				}
			}

			if (node.activation == ActivationFunction.SIGMOID) {
				nodeValues.set(node.id, sigmoid(totalInput));
			}
		}

		let networkOutputs: number[] = [];

		for (let node of outputNodes) {
			networkOutputs.push(nodeValues.get(node.id) ?? 0);
		}

		return networkOutputs;
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
	nodeInnovation: number;
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
				nodeInnovation: nodeId,
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
				nodeInnovation: 0,
				connection1Innovation: 0,
				connection2Innovation: 0
			};
		}
	}
}

export class NEAT {
	population: NeuralNetwork[] = [];
	innovationTracker: InnovationTracker = new InnovationTracker();

	constructor(numberOfInputs: number, numberOfOutputs: number, size: number) {
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
}
