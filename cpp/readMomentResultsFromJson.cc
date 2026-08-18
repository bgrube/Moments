// example program showing how to read moment values from a JSON file
// compile from main project dir by running
//   g++ -std=c++17 -I. -o cpp/readMomentResultsFromJson{,.cc}


#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"


// define classes that represent Python dataclasses in the `MomentCalculation` module
struct QnMomentIndex
{
	int momentIndex;
	int L;
	int M;

	bool
	operator<(const QnMomentIndex& other) const {
		return std::tie(momentIndex, L, M) < std::tie(other.momentIndex, other.L, other.M);
	}
};

struct MomentValue
{
	int momentIndex;
	int L;
	int M;
	double valRe;
	double uncertRe;
	double valIm;
	double uncertIm;
	std::map<std::string, double> binCenters;  // map with center values of the kinematic variables that define the data bin
};

class MomentIndices
{
	public:
		MomentIndices() = default;

		MomentIndices(
			const int                         maxL,
			const bool                        polarized,
			const std::vector<QnMomentIndex>& indices
		) : maxL(maxL),
				polarized(polarized)
			{
				for (size_t flatIndex = 0; flatIndex < indices.size(); ++flatIndex) {
					const QnMomentIndex& qnIndex = indices[flatIndex];
					_flatToQnIndex[flatIndex] = qnIndex;
					_qnIndexToFlat[qnIndex]   = flatIndex;
				}
			}

		size_t nmbMoments() const
		{
			return _flatToQnIndex.size();
		}

		const QnMomentIndex&
		operator [](const int flatIndex) const
		{
			return _flatToQnIndex.at(flatIndex);
		}

		int
		operator [](const QnMomentIndex& qnIndex) const
		{
			return _qnIndexToFlat.at(qnIndex);
		}

		int  maxL;
		bool polarized;

	private:
		std::map<int, QnMomentIndex> _flatToQnIndex;  // maps flat index to QnMomentIndex
		std::map<QnMomentIndex, int> _qnIndexToFlat;  // maps QnMomentIndex to flat index
};

struct BinCenter
{
	std::string varName;
	double      centerValue;
	std::string label;
	std::string unit;
	int         nmbDigits;
};

struct MomentResult {
	MomentIndices                     indices;
	std::vector<BinCenter>            binCenters;
	std::vector<std::complex<double>> _valsFlatIndex;
};


// define JSON serialization for simple structs
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
	QnMomentIndex,
	momentIndex,
	L,
	M
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
	BinCenter,
	varName,
	centerValue,
	label,
	unit,
	nmbDigits
)

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


// define JSON serialization for classes that require a custom constructor or classes that contain such classes
void
from_json(
	const nlohmann::json& json,
	MomentIndices&        indices
) {
	const int  maxL                      = json.at("maxL"     ).get<int>();
	const bool polarized                 = json.at("polarized").get<bool>();
	std::vector<QnMomentIndex> qnIndices = json.at("qnIndices").get<std::vector<QnMomentIndex>>();
	indices = MomentIndices(maxL, polarized, qnIndices);
}

void
from_json(
	const nlohmann::json& json,
	MomentResult&         result
) {
	json.at("indices").get_to(result.indices);
	json.at("binCenters").at("binCenters").get_to(result.binCenters);
	// read moment values
	const auto&  jsonNdarray = json.at("_valsFlatIndex");
	const auto&  jsonData    = jsonNdarray.at("data");
	const size_t nmbValues   = jsonNdarray.at("shape").at(0).get<size_t>();
	if (jsonData.size() != nmbValues)
		throw std::runtime_error("Data size = " + std::to_string(jsonData.size())
			+ " does not match 'shape' = [" + std::to_string(nmbValues) + "]");
	result._valsFlatIndex.resize(nmbValues);
	for (size_t indexValue = 0; indexValue < nmbValues; ++indexValue) {
		const auto& jsonValue = jsonData.at(indexValue);
		result._valsFlatIndex[indexValue] = std::complex<double>(jsonValue.at("re").get<double>(), jsonValue.at("im").get<double>());
	}
}


int
main()
{
	const std::string jsonFilePath = "./moments.json";
	std::cout << "Reading moment values from JSON file '" << jsonFilePath << "'" << std::endl;
	std::ifstream inFile(jsonFilePath);
	if (not inFile) {
		std::cerr << "Could not open file '" << jsonFilePath << "' for reading." << std::endl;
		return 1;
	}
	nlohmann::json json;
	inFile >> json;

	std::vector<MomentResult> momentResults = json.at("results").get<std::vector<MomentResult>>();
	for (const auto& result : momentResults) {
		std::cout << "indices.maxL = "      << result.indices.maxL << std::endl;
		std::cout << "indices.polarized = " << result.indices.polarized << std::endl;
		for (const auto& binCenter : result.binCenters) {
			std::cout << "binCenter.varName = "       << binCenter.varName
			          << ", binCenter.centerValue = " << binCenter.centerValue
			          << ", binCenter.label = "       << binCenter.label
			          << ", binCenter.unit = "        << binCenter.unit
			          << ", binCenter.nmbDigits = "   << binCenter.nmbDigits
			          << std::endl;
		}
		assert(result.indices.nmbMoments() == result._valsFlatIndex.size() && "Mismatch between number of moments and values");
		for (size_t flatIndex = 0; flatIndex < result.indices.nmbMoments(); ++flatIndex) {
			const QnMomentIndex&        qnIndex = result.indices[flatIndex];
			const std::complex<double>& val     = result._valsFlatIndex[flatIndex];
			std::cout << "result._valsFlatIndex[" << flatIndex << "] = {momentIndex: " << qnIndex.momentIndex
			          << ", L: " << qnIndex.L << ", M: " << qnIndex.M
			          << ", val: " << val << "}" << std::endl;
		}
	}

	return 0;
}
