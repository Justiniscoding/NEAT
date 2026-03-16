import { NEAT, NeuralNetwork } from "./lib/nn"

function fitness(network: NeuralNetwork) {
	const inputs = [[0, 0], [1, 0], [0, 1], [1, 1]];
	const outputs = [0, 1, 1, 0];

	let totalError = 0;

	for (let i = 0; i < 4; i++) {
		const output = network.evaluate(inputs[i]);

		if (output) {
			const error = Math.abs(output[0] - outputs[i]);
			totalError += error;
		} else {
			totalError += outputs[i];
		}
	}

	return Math.max(0, 4 - totalError);
}

const neat = new NEAT(2, 1, 50, fitness);

neat.evolve();
