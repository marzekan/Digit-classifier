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
	int epochs{ 1 }; // Number of training epochs, set to 1.
 
	int batch_size{ 2000 }; // Number of lines to read from file in one batch.

	int max_lines = 0; // Maximum number of rows in mnist_train.csv file.

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
	void calculate_adam()
	{


	}

	
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

	// Prints out given vector in a for-each loop.
	void print_vector(std::vector<char>& arr)
	{
		for (auto i : arr)
		{
			std::cout << i;
		}

		std::cout << "\n";
	}

	// Writes out progress bar of network training to console.
	void write_progress(int& iter)
	{
		std::vector<char> progressBar {'-', '-', '-', '-', '-', '-', '-', '-', '-', '-'};

		print_vector(progressBar);

		for (int i = 0; i < iter; i++)
		{
			if (int(i == (0.1 * iter)))
			{
				progressBar.at(0) = '=';
				print_vector(progressBar);
				
			}
			else if (int(i == (0.2 * iter)))
			{
				progressBar.at(1) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.3 * iter)))
			{
				progressBar.at(2) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.4 * iter)))
			{
				progressBar.at(3) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.5 * iter)))
			{
				progressBar.at(4) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.6 * iter)))
			{
				progressBar.at(5) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.7 * iter)))
			{
				progressBar.at(6) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.8 * iter)))
			{
				progressBar.at(7) = '=';
				print_vector(progressBar);
			}
			else if (int(i == (0.9 * iter)))
			{
				progressBar.at(8) = '=';
				print_vector(progressBar);
			}
			else if (i == (iter-1))
			{
				progressBar.at(9) = '=';
				print_vector(progressBar);
			}
		}
	}

public:
	NeuralNetwork(int& epochs_in, int& batch_in, int& maxLines)
	{

		epochs = epochs_in;
		batch_size = batch_in;
		max_lines = maxLines;

		std::cout << "NNetwork created!\n";

	}

	// Training the neural network until the criteria is set.
	void train()
	{
		auto start = timeNow();

		std::cout << "Training start...\n";

		FileLoader fileldr;
		double mse;

		// Single row from the train.csv file.
		std::vector<double> row;

		init_network();
		
		// Number of iterations in each epoch. Calculated by deviding maximum number of rows in file with the batch size.
		int iter = (max_lines / batch_size) + 1;
		//int iter = 2;

		for (int i = 0; i < epochs; i++)
		{
			/*
			*
			*	ZAMJENIT ITER SA PRAVIM BROJEVIMA!!
			*
			*/

			for (int k = 0; k < iter; k++)
			{
				if (k != 0 && k == iter - 1)
				{
					batch_size = max_lines % k;
				}

				row = fileldr.read_batch_CSV("mnist_train.csv", batch_size, k);

				for (int j = 0; j < row.size(); j++)
				{
					set_inputs(row);

					feed_forward();

					backpropagation();

					mse = calculate_MSE();

					//int bb = j * (k + 1);

					//write_progress(bb);

					if (j % 100 == 0 && j != 0)
					{
						std::cout << "Training iteration: " << j * (iter + 1)<< "...\n";
					}
				}
			}
		}

		/*
		do
		{
			row = fileldr.read_CSV("mnist_train.csv", i);

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

		row = fl.read_CSV("mnist_test.csv",i);

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
	void save_network()
	{
		// FileLoader....
	}

	// Tests network for a given number of iterations
	void automated_testing(int& iter)
	{
		// iter = 10.000
		// returns percetage of correctly classified digits.
	}
	

	~NeuralNetwork()
	{
		std::cout << "Network deleted.\n";
		Delete();
	};
};

