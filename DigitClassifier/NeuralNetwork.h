#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include<iterator>
#include<chrono>

#include "FileLoader.h"

#define print(X) std::cerr << X << "\n"

// Random weight value for network initialization.
#define RAND_WEIGHT	(((double)rand() / (double)RAND_MAX) - 0.5)

class NeuralNetwork
{
//TIME FUNCTIONS
#pragma region MyRegion
/***************************************************************************************************/

	typedef std::chrono::time_point<std::chrono::steady_clock> vreme;

	vreme timeNow()
	{
		return std::chrono::high_resolution_clock::now();
	}

	void timeElapsed(vreme& start, vreme& end)
	{
		double elap = (1e-9)*(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

		std::cout << "Time elapsed: " << elap << "\n";
	}

	/***************************************************************************************************/
#pragma endregion

// Variables
public:

private:
	const int input_n{ 785 };
	const int* input_p = &input_n; // Number of inputs (154 + 1, one for label, one for each dimension of a reduced dataset and one for bias).

	const int output_n{ 10 };
	const int* output_p = &output_n; // Number of outputs, one for each digit.

	const int hidden_n{ 519 };
	const int* hidden_p = &hidden_n;// Number of hidden neurons. 101 hidden neurons + 1 bias. Number of hidden layers and neurons is taken from an article, link in the Readme.

	const double lRate{ 0.01 };
	const double* learn_rate = &lRate; // Neural network learning rate.


	// Network inputs.
	std::vector<double>* inputs = new std::vector<double>(*input_p);

	// Network ouputs.
	std::vector<double>* outputs = new std::vector<double>(*output_p);

	// Hidden layer values.
	std::vector<double>* hidden = new std::vector<double>(*hidden_p);


	// Input-hidden weights.
	std::vector<std::vector<double>>* weights_I_H = new std::vector<std::vector<double>>(*hidden_p, std::vector<double>(*input_p));

	// Output-hidden weights.
	std::vector<std::vector<double>>* weights_H_O = new std::vector<std::vector<double>>(*output_p, std::vector<double>(*hidden_p));


	// Reference to output error vector.
	std::vector<double>* error_output = new std::vector<double>;

	// Reference to hidden error vector.
	std::vector<double>* error_hidden = new std::vector<double>;


	// Label vector.
	std::unique_ptr<std::vector<int>> out{ new std::vector<int> };

// Methods

private:

	// Sigmoid function function.
	double sigmoid(double& in)
	{
		double exp_value;

		// Calculating exponential
		exp_value = exp(-in);

		// Sigmoid funciton calculation
		return (1 / (1 + exp_value));
	}

	// Derivative of sigmoid function.
	double sigmoid_d(double& sig)
	{
		return sig * (1 - sig);
	}

	// Feeding inputs forward.
	void feed_forward()
	{
		// Calculation for hidden layer.
		for (int i = 0; i < *hidden_p - 1; i++)
		{
			for (int j = 0; j < *input_p; j++)
			{
				// Calculating weights.
				hidden->at(i) += (weights_I_H->at(i).at(j) * inputs->at(j));
			}
			hidden->at(i) = sigmoid(hidden->at(i));
		}

		// Calculation for output layer.
		for (int i = 0; i < *output_p; i++)
		{
			for (int j = 0; j < *hidden_p; j++)
			{
				// Calculation weigths.
				outputs->at(i) += (weights_H_O->at(i).at(j) * hidden->at(j));
			}
			outputs->at(i) = sigmoid(outputs->at(i));
		}
	}

	// Converts label to vector of size = 10 where 1 represents the given digit. 
	auto label_vector(double& label)
	{
		out->clear();
		out->resize(10);
		out->at(label) = 1;
		return *out;
	}

	// Backpropagating errors.
	void backpropagation()
	{
		error_output->resize(*output_p);
		error_hidden->resize(*hidden_p - 1);

		// Calculating errors of the output neurons.
		for (int i = 0; i < *output_p; i++)
		{
			error_output->at(i) = (label_vector(inputs->at(0)).at(i) - outputs->at(i)) * sigmoid_d(outputs->at(i));
		}

		// Calculating errors for the hidden neurons.
		for (int i = 0; i < *hidden_p - 1; i++)
		{
			for (int j = 0; j < *output_p; j++)
			{
				error_hidden->at(i) += error_output->at(j) * weights_H_O->at(j).at(i);
			}

			error_hidden->at(i) *= sigmoid_d(hidden->at(i));
		}

		// Ajusting weights from hidden to output layer.
		for (int i = 0; i < *output_p; i++)
		{
			for (int j = 0; j < *hidden_p; j++)
			{
				weights_H_O->at(i).at(j) += *learn_rate * error_output->at(i) * hidden->at(j);
			}
		}

		// Ajusting weights from input to hidden layer.
		for (int i = 0; i < *hidden_p - 1; i++)
		{
			for (int j = 0; j < *input_p; j++)
			{
				weights_I_H->at(i).at(j) += *learn_rate * error_hidden->at(i) * inputs->at(j);
			}
		}
	}

	// Calculates the Mean Square Error of a given input.
	double calculate_MSE()
	{
		double mse = 0.0;

		for (int i = 0; i < *output_p; i++)
		{
			mse += pow(label_vector(inputs->at(0)).at(i) - outputs->at(i), 2);
		}
		return (mse / static_cast<double>(*output_p));
	}


	// calculate Adam


	// Network initialization.
	void init_network()
	{
		// Setting the Bias neuron value.
		inputs->resize(*input_p);
		hidden->resize(*hidden_p);
		outputs->resize(*output_p);

		weights_H_O->resize(*output_p, std::vector<double>(*hidden_p));
		weights_I_H->resize(*hidden_p, std::vector<double>(*input_p));

		inputs->at(*input_p - 1) = 1.0;

		hidden->at(*hidden_p - 1) = 1.0;

		// Initializing input to hidden weights.
		for (int i = 0; i < *hidden_p - 1; i++)
		{
			for (int j = 0; j < *input_p; j++)
			{
				weights_I_H->at(i).at(j) = RAND_WEIGHT;
			}
		}

		// Initializing hidden to output weights.
		for (int i = 0; i < *output_p; i++)
		{
			for (int j = 0; j < *hidden_p; j++)
			{
				weights_H_O->at(i).at(j) = RAND_WEIGHT;
			}
		}
	}

	// Sets network inputs so that each neuron in the input layer matches one feature from .csv file.
	void set_inputs(std::vector<double>& row)
	{
		for (auto j = 0; j < inputs->size(); j++)
		{
			inputs->at(j) = row.at(j);
		}
	}

	// Deallocates all dynamically allocated objects.
	void Delete()
	{
		delete inputs;
		inputs = nullptr;
		if (inputs != NULL)
		{
			print("inputs - DEALLOCATION FAILED.");
		}

		delete outputs;
		outputs = nullptr;
		if (outputs != NULL)
		{
			print("outputs - DEALLOCATION FAILED.");
		}

		delete hidden;
		hidden = nullptr;
		if (hidden != NULL)
		{
			print("hidden - DEALLOCATION FAILED.");
		}

		delete weights_H_O;
		weights_H_O = nullptr;
		if (weights_H_O != NULL)
		{
			print("weights_H_O - DEALLOCATION FAILED.");
		}

		delete weights_I_H;
		weights_I_H = nullptr;
		if (weights_I_H != NULL)
		{
			print("weights_I_H - DEALLOCATION FAILED.");
		}

		delete error_output;
		error_output = nullptr;
		if (error_output != NULL)
		{
			print("error_output - DEALLOCATION FAILED.");
		}

		delete error_hidden;
		error_hidden = nullptr;
		if (error_hidden != NULL)
		{
			print("error_hidden - DEALLOCATION FAILED.");
		}
	}

	// Finds the largest output.
	int classifier()
	{
		// Largest element in vector.
		double best = *std::max_element(outputs->begin(), outputs->end());

		// Finds index of largerst element.
		auto iter = std::find(outputs->begin(), outputs->end(), best);

		// Returns index of largest element.
		return std::distance(outputs->begin(), iter);
	}



public:
	NeuralNetwork();

	// Training the neural network until the criteria is set.
	void train(int& iter)
	{
		auto start = timeNow();

		std::cout << "Training start...\n";

		FileLoader fileldr;
		double mse;

		// Single row from the train.csv file.
		std::vector<double> row;

		init_network();

		// Maximum number of lines in mnist_train.csv file.
		// When loop reaches end of file, it starts over untill the number of iterations is reached.
		const int max_lines = 59998;

		for (int i = 0; i < iter; i++)
		{
			if (i > max_lines)
			{
				i = 0;
			}

			row = fileldr.CSVFileRead("mnist_train.csv", i);

			set_inputs(row);

			feed_forward();

			backpropagation();

			mse = calculate_MSE();

			if (i % 50 == 0 && i != 0)
			{
				std::cout << "epoch: " << i << "...\n";
			}

		}
		/*int i = 0;
		do
		{
			row = fileldr.CSVFileRead("mnist_train.csv", i);

			set_inputs(row);
			//print("inputs set");

			feed_forward();
			//print("fed forward");

			backpropagation();
			//print("back-propragated");

			mse = calculate_MSE();
			//print("MSE calculated");

			UI.epoch_output(i);
			i++;

		} while (mse > 0.001);*/

		auto end = timeNow();

		timeElapsed(start, end);

		std::cout << "Training done.\n";

		/*FileLoader fl;
		fl.Save_weights(*weights_I_H, *weights_H_O);*/
	}

	void test_NN(int& i)
	{
		FileLoader fl;
		std::vector<double> row;

		row = fl.CSVFileRead("mnist_test.csv", i);

		set_inputs(row);

		feed_forward();

		int classified = classifier();

		std::cout << "\nReal value: " << row.at(0) << "\n";

		std::cout << "Classified as: " << classified << " , at " << outputs->at(classified) << "\n";

		for (auto i = 0; i < *output_p; i++)
		{
			std::cout << outputs->at(i) << ", ";
		}

		std::cout << "\n";
	}



	// Sets up previously trained network.
	void load_network()
	{
		// FileLoader...
	}

	// Saves weights of current running network.
	void SaveSession()
	{
		// FileLoader....
	}

	// Tests network for a given number of iterations
	void automated_testing(int& iter)
	{
		// iter = 1000
		// returns percetage of correctly classified digits.
	}
	

	~NeuralNetwork();
};

