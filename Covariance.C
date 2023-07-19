// functor for RDataFrame that calculates the cross-covariance of 2 scalar columns
// from https://root-forum.cern.ch/t/help-to-extend-covariance-functor-for-rdataframe-functionality/45335
// see also https://root.cern/doc/master/df018__customActions_8C.html

#include <memory>

#include "ROOT/RDataFrame.hxx"


// T is the type of the scalar column
// Example usage with RDataFrame
// Covariance<double> Cov();
// auto covXY = df.Book<double,double>(std::move(Cov), {"xColumn", "yColumn"}))->GetValue();
template< typename T>
class Covariance : public ROOT::Detail::RDF::RActionImpl<Covariance<T>> {

private:

	std::shared_ptr<T>  _covariance;
	// one per data-processing slot
	std::vector<size_t> _nmbEntries;
	std::vector<T>      _xSum;
	std::vector<T>      _ySum;
	std::vector<T>      _xySum;

public:

	using Result_t = T;  // this type is required for every helper

	Covariance()
	{
		const auto nmbSlots = ROOT::IsImplicitMTEnabled() ? ROOT::GetThreadPoolSize() : 1;
		for (auto i : ROOT::TSeqU(nmbSlots)) {
			_nmbEntries.emplace_back(0);
			_xSum.emplace_back      (0);
			_ySum.emplace_back      (0);
			_xySum.emplace_back     (0);
		}
		_covariance = std::make_shared<double>(0);
	}
	Covariance(Covariance &&) = default;
	Covariance(const Covariance &) = delete;

	template <typename... ColumnTypes>
	void
	Exec(
		unsigned int   slot,
		ColumnTypes... values
	)	{
		std::array<double, sizeof...(ColumnTypes)> valuesArr{static_cast<double>(values)...};
		++_nmbEntries[slot];
		_xSum        [slot] += valuesArr[0];
		_ySum        [slot] += valuesArr[1];
		_xySum       [slot] += valuesArr[0] * valuesArr[1];
	}

	void
	Finalize()
	{
		//TODO improve algorithm; see https://www.wikiwand.com/en/Algorithms_for_calculating_variance#Online
		for (auto slot : ROOT::TSeqU(1, _xySum.size())) {
			_nmbEntries[0] += _nmbEntries[slot];
			_xSum      [0] += _xSum      [slot];
			_ySum      [0] += _ySum      [slot];
			_xySum     [0] += _xySum     [slot];
		}
		*_covariance = (1. / (_nmbEntries[0] - 1)) * (_xySum[0] - _xSum[0] * _ySum[0] / _nmbEntries[0]);
	}

	std::shared_ptr<T> GetResultPtr() const { return _covariance; }
	std::string GetActionName() { return "Covariance"; }
	void Initialize() { }
	void InitTask(TTreeReader*, unsigned int) { }
};
