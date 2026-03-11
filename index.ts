import { NEAT } from "./lib/nn"

const neat = new NEAT(2, 1, 10);

console.log(neat.distance(neat.population[0], neat.population[1]));

// for (let i = 0; i < 10; i++) {
// 	console.log(neat.population[i].evaluate([0, 1]));
// }
