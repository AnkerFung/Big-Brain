#pragma once

#include <vector>

#include "../layers/fully_connected_layer.h"

namespace bb
{
	class fnn
	{
	public:
		fnn(const std::vector<size_t>& dim)
		{
			assert(dim.size() > 1);
			size_t prev_dim = dim[0];
			for (auto i = dim.cbegin() + 1; i != dim.cend(); i++)
			{
				layers.emplace_back(prev_dim, *i);
				prev_dim = *i;
			}
			for (auto& layer : layers)
				layer.init_weight();
		}
		void set_input(const vec& in)
		{
			layers[0].set_input(in);
		}
		const vec& get_output() const
		{
			return layers[layers.size() - 1].get_output();
		}
		void predict()
		{
			for (auto i = layers.begin() + 1; i != layers.end(); i++)
			{
				(i - 1)->forward();
				i->set_input((i - 1)->get_output());
			}
			layers[layers.size() - 1].forward();
		}
		void train(const std::vector<std::pair<vec, vec>>& sets, double lr, size_t epoch)
		{
			for (size_t i = 0; i < epoch; i++)
			{
				for (const auto& [sample, label] : sets)
				{
					train(sample, label, lr);
				}
			}
		}
		void train(const vec& sample, const vec& label, double lr)
		{
			set_input(sample);
			predict();
			update_deltas(label);
			update_weights(lr);
		}
	private:
		void update_deltas(const vec& label)
		{
			auto i = layers.rbegin();
			i->backward(label);
			i++;
			for (; i != layers.rend(); i++)
			{
				i->backward(*(i - 1));
			}
		}
		void update_weights(double lr)
		{
			for (auto& layer : layers)
			{
				layer.update_weight(lr);
			}
		}
		std::vector<fully_connected_layer> layers;
	};
}