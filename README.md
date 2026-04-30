# NEAT
This is an implementation of the NEAT algorithm in TypeScript. It is a genetic algorithm used for evolving/training Neural Networks.

This is heavily based off of the work of Carlos Redondo, where he implemented the NEAT algorithm in python and wrote about it [here](https://towardsdatascience.com/from-genes-to-neural-networks-understanding-and-building-neat-neuro-evolution-of-augmenting-topologies-from-scratch/)
It is currently able to get a fitness of about 2.7 on the XOR problem within 15,000 generations, and needs an intense bug fixing session to get that up above 3.9! It is still functional and a demo/proof of concept of the algorithm either way.

Future goals:
- Adjust algorithm to be able to get higher/optimal fitness values.
- Use this implementation to learn to play a game.

# Running This Project
To run this project, first clone the repository. You can either download it strictly from this repository by clicking Code and then "Download ZIP", or you can use the following command:
`git clone https://github.com/justiniscoding/neat`
And then to go in to the project directory:
`cd neat`

For the next step, make sure to have Node.js installed. If not, you can download and install it from [here](https://nodejs.org).

Run the command `npm install` in the terminal to install the dependencies required for this project, and then you can use `npm run dev` to run the project's code. It will then attempt to solve the XOR problem using the NEAT algorithm over the course of 15,000 generations.
