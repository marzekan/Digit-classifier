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

	/*
	*	TODO
	*	- add variable "epoch" - nums of training epochs
	*	- add variable "batchSize" - number of rows to read from dataFile
	*/

	int epochs = 1;
	int batchSize = 2000;

	//std::cout << "Insert number of epochs: ";
	//std::cin >> epochs;

	//std::cout << "Insert batch size:";
	//std::cin >> batchSize;

	

	//double completion_time = train_iter * 1.55 * 0.8;
	//std::cout << "Estimated training time: " << completion_time << "s\n";

	system("pause");
	
	int MAX_LINES = 59998;
	
	//{
	//	FileLoader fl;
	//	MAX_LINES = fl.max_lines("mnist_train.csv");
	//}


	NeuralNetwork nn(epochs,batchSize,MAX_LINES);

	nn.train();

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
