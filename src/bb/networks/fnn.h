#pragma once

#include <vector>

#include "../layers/fully_connected_layer.h"

namespace bb
{
	class fnn
	{
	public:
		using training_set = std::vector<std::pair<vec, vec>>;
		fnn(const std::vector<size_t>& dim);
		fnn(const std::string& path);
		void set_input(const vec& in);
		const vec& get_output() const;
		void predict();
		void train(const training_set& sets, double lr, size_t epoch);
		void train(const vec& sample, const vec& label, double lr);
		void store_to_file(const std::string& path);
	private:
		void update_deltas(const vec& label);
		void update_weights(double lr);
		std::vector<fully_connected_layer> layers;
	};
}