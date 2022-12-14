#pragma once

#include "../../linalg/vector.h"
#include "../../linalg/matrix.h"

namespace bb
{
	using namespace linalg;
	class fully_connected_layer
	{
	public:
		fully_connected_layer(size_t _Prev, size_t _Dim);
		void set_input(const vec& _In);
		const vec& get_output() const;
		void forward();
		void backward(const vec& target);
		void backward(const fully_connected_layer& next);
		void update_weight(double lr);
		void init_weight();
		size_t dimension() const { return dim; }
		size_t prev_dimension() const { return prev_dim; }
		void store_weight_to_stream(std::ostream& stream);
		void load_weight_from_stream(std::istream& stream);
	private:
		static double activator(double x);
		static double activator_deriv(double x);
		size_t prev_dim, dim;
		vec in;
		vec out;
		vec delta;
		mat weight;
	};
}