#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <sstream>
#include <chrono>
#include <ctime>

#pragma warning(disable : 4996)

class FileLoader
{
private:

	std::string timeNow()
	{
		auto time = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(time);

		return std::ctime(&in_time_t);
	}

public:
	// Methods
	FileLoader();
	

	//Save function. Saves weights to .txt file for later usage.
	void Save_weights(std::vector<std::vector<double>>& hidden_w, std::vector<std::vector<double>>& output_w)
	{
		std::string time_now = timeNow();
		std::string fileName = "Saves/weights_" + time_now;

		std::ifstream file(fileName);

		if (!file)
		{
			std::ofstream save_file("Saves/" + fileName);
			std::string s;
			double weight;

			for (auto i = 0; i < 519; i++)
			{
				for (auto j = 0; j < 785; j++)
				{
					weight = hidden_w.at(i).at(j);
					s = std::to_string(weight);
					save_file << s << "\n";
				}
			}

			for (auto i = 0; i < 10; i++)
			{
				for (auto j = 0; j < 519; j++)
				{
					weight = output_w.at(i).at(j);
					s = std::to_string(weight);
					save_file << s << "\n";
				}
			}

			std::cout << "File saved." << "\n";
		}

		// odredit do koje linije je koji. BITNO AF
	}

	//Load function. Loads weights from .txt file.
	void Load_weights(std::vector<std::vector<double>>& hidden, std::vector<std::vector<double>>& output)
	{
		// stavit linije fajla u vektore.
	}

	// Reads one .csv file line and returnes vector of values.
	std::vector<double> CSVFileRead(const char* filename, int& it)
	{
		std::ifstream file(filename);

		if (!file.is_open())
		{
			std::cerr << "File cannot be opened..." << "\n";
			return {};
		}
		else {
			
			std::vector<double> data_list;
			std::string line;

			char delimiter = ',';

			// Gets one line from file.
			for (int i = 0; i <= it; i++)
			{
				getline(file, line);
			}

			// Converting string (line) to stringstream.
			std::istringstream ss(line);
			std::string datapoint;

			// Saves every value as a seperate vector member.
			while (getline(ss, datapoint, delimiter))
			{
				data_list.emplace_back(std::stod(datapoint));
			}

			return data_list;
		}
	}


	~FileLoader();
};

