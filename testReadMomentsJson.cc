// example program showing how to read a vector of moment values from a JSON file
// compile with = g++ -std=c++17 -o testReadMomentsJson testReadMomentsJson.cc


#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"


// struct that holds a single moment value and its associated metadata
typedef struct {
	int momentIndex;
	int L;
	int M;
	double valRe;
	double uncertRe;
	double valIm;
	double uncertIm;
	std::map<std::string, double> binCenters;  // map with center values of the kinematic variables that define the data bin
} MomentValue;

// define JSON serialization for the MomentValue struct
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
	MomentValue,
	momentIndex,
	L,
	M,
	valRe,
	uncertRe,
	valIm,
	uncertIm,
	binCenters
)


int
main() {
	const std::string jsonFileName = "./test.json";
	std::cout << "Reading moment values from JSON file '" << jsonFileName << "'" << std::endl;
	std::ifstream inFile(jsonFileName);
	if (not inFile) {
		std::cerr << "Could not open file '" << jsonFileName << "' for reading." << std::endl;
		return 1;
	}
	nlohmann::json json;
	inFile >> json;

	// MomentValues from JSON object
	std::vector<MomentValue> moments = json.get<std::vector<MomentValue>>();
	std::cout << "Read " << moments.size() << " moment values from JSON file '" << jsonFileName << "'" << std::endl;
	for (const auto& m : moments) {
		std::cout << "momentIndex = " << m.momentIndex
		          << ", L = "         << m.L
		          << ", M = "         << m.M
		          << ", valRe = "     << m.valRe
		          << ", uncertRe = "  << m.uncertRe
		          << ", valIm = "     << m.valIm
		          << ", uncertIm = "  << m.uncertIm;
		std::cout << ", binCenters = {";
		bool first = true;
		for (const auto& [key, value] : m.binCenters) {
				if (not first)
					std::cout << ", ";
				std::cout << key << ": " << value;
				first = false;
		}
		std::cout << "}" << std::endl;
	}

	return 0;
}
