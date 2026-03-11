import { NEAT } from "./lib/nn"

const neat = new NEAT(3, 1, 10);
for (let i = 0; i < 10; i++) {
	console.log(neat.population[i].evaluate([0, 1, 0]));
}
