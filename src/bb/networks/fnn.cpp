#include "fnn.h"

#include <fstream>

namespace bb
{

	fnn::fnn(const std::vector<size_t>& dim)
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
	
	fnn::fnn(const std::string& path)
	{
		std::fstream stream;
		stream.open(path, std::ios_base::in | std::ios_base::binary);
		uint64_t input_num, layer_num;
		stream.read((char*)&input_num, sizeof(uint64_t));
		stream.read((char*)&layer_num, sizeof(uint64_t));
		uint64_t prev_dim = input_num;
		for (uint64_t i = 0; i < layer_num; i++)
		{
			uint64_t dim;
			stream.read((char*)&dim, sizeof(uint64_t));
			layers.emplace_back(prev_dim, dim);
			prev_dim = dim;
		}
		for (auto& layer : layers)
		{
			layer.load_weight_from_stream(stream);
		}
		stream.close();
	}

	void fnn::set_input(const vec& in)
	{
		layers[0].set_input(in);
	}

	const vec& fnn::get_output() const
	{
		return layers[layers.size() - 1].get_output();
	}

	void fnn::predict()
	{
		for (auto i = layers.begin() + 1; i != layers.end(); i++)
		{
			(i - 1)->forward();
			i->set_input((i - 1)->get_output());
		}
		layers[layers.size() - 1].forward();
	}

	void fnn::train(const training_set& sets, double lr, size_t epoch)
	{
		for (size_t i = 0; i < epoch; i++)
		{
			for (const auto& [sample, label] : sets)
			{
				train(sample, label, lr);
			}
		}
	}

	void fnn::train(const vec& sample, const vec& label, double lr)
	{
		set_input(sample);
		predict();
		update_deltas(label);
		update_weights(lr);
	}

	void fnn::store_to_file(const std::string& path)
	{
		std::fstream stream;
		stream.open(path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
		uint64_t input_num = (uint64_t)layers.begin()->prev_dimension();
		stream.write((char*)&input_num, sizeof(uint64_t));
		uint64_t layer_num = (uint64_t)layers.size();
		stream.write((char*)&layer_num, sizeof(uint64_t));
		for (const auto& layer : layers)
		{
			uint64_t node_num = (uint64_t)layer.dimension();
			stream.write((char*)&node_num, sizeof(uint64_t));
		}
		for (auto& layer : layers)
		{
			layer.store_weight_to_stream(stream);
		}
		stream.close();
	}

	void fnn::update_deltas(const vec& label)
	{
		auto i = layers.rbegin();
		i->backward(label);
		i++;
		for (; i != layers.rend(); i++)
		{
			i->backward(*(i - 1));
		}
	}

	void fnn::update_weights(double lr)
	{
		for (auto& layer : layers)
		{
			layer.update_weight(lr);
		}
	}

}