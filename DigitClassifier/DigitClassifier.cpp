#include "pch.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <stdlib.h>
#include <vector>
#include <array>
#include <algorithm>
#include <iterator>

#include "FileLoader.h"
#include "NeuralNetwork.h"

// JUST FOR DEBUGGING PURPOSES!
#define space std::cout << "\n"
#define print(X) std::cerr << X << "\n"


int main()
{
	system("title MNIST Digit Classifier");

	int train_iter;

	std::cout << "Insert number of learning iterations: ";
	std::cin >> train_iter;

	double completion_time = train_iter * 1.55 * 0.8;
	std::cout << "Estimated training time: " << completion_time << "s\n";

	system("pause");

	NeuralNetwork nn;

	nn.train(train_iter);

	space;
	space;

	char generateMore;
	int i = 0;

	do
	{
		nn.test_NN(i);
		space;

		std::cout << "\nAnother test? (y/n): ";
		std::cin >> generateMore;
		i++;

	} while (generateMore != 'n' && i < 10000);

	space;

	// do while iterator and mse > 0.001
}
