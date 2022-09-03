#include "fully_connected_layer.h"
#include <random>

namespace bb
{
	// i tried to vectorize the implementation as much as possible, not the best for performance atm due to the excess copying of vectors and matrices

	fully_connected_layer::fully_connected_layer(size_t _Prev, size_t _Dim)
		:
		prev_dim(_Prev),
		dim(_Dim),
		in(prev_dim + 1),
		out(_Dim),
		delta(_Dim),
		weight(prev_dim + 1, dim) // account for the bias node
	{
		in[prev_dim] = 1.0;
	}
	void fully_connected_layer::set_input(const vec& _In)
	{
		assert(_In.dimension() == prev_dim);
		for (size_t i = 0; i < prev_dim; i++)
			in[i] = _In[i];
	}
	const vec& fully_connected_layer::get_output() const
	{
		return out;
	}
	void fully_connected_layer::forward()
	{
		out = transform(weight * in, activator);
	}
	void fully_connected_layer::backward(const vec& target)
	{
		delta = transform(out, activator_deriv) * (out - target);
	}
	void fully_connected_layer::backward(const fully_connected_layer& next)
	{
		vec a = multiply_transposed(next.weight, next.delta);
		vec b(dim);
		for (size_t i = 0; i < dim; i++)
			b[i] = a[i];
		delta = transform(out, activator_deriv) * b;
	}
	void fully_connected_layer::update_weight(double lr)
	{
		mat a_m(prev_dim + 1, 1, in.data());
		mat d_m(1, dim, delta.data());
		weight = weight - lr * d_m * a_m; // gradient descent
	}
	void fully_connected_layer::init_weight()
	{
		std::random_device rd;
		std::default_random_engine engine(rd());
		std::uniform_real_distribution<> distr(-0.1, 0.1);
		for (int j = 0; j < dim; j++)
			for (int i = 0; i < prev_dim + 1; i++)
				weight.get(i, j) = distr(engine);
	}
	double fully_connected_layer::activator(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}
	double fully_connected_layer::activator_deriv(double y) // needs to be represented using the output of the activator
	{
		return y * (1.0 - y);
	}
}