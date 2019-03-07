#include "pch.h"
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{
	std::cout << "NNetwork created!\n";

}


NeuralNetwork::~NeuralNetwork()
{
	std::cout << "Network deleted.\n";
	Delete();
}
