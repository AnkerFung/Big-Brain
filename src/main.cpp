#include <iostream>
#include <fstream>

#include "bb/networks/fnn.h"

using namespace linalg;

std::vector<std::pair<vec, vec>> read_training_sets()
{
	constexpr const size_t sample_num = 60000;
	constexpr const size_t image_size = 28 * 28;
	std::vector<std::pair<vec, vec>> result;
	{
		std::fstream stream_samples;
		stream_samples.open("resources/mnist/train-images.idx3-ubyte", std::ios_base::in | std::ios_base::binary);
		stream_samples.seekg(16);
		char* buffer_samples = new char[image_size * sample_num];
		stream_samples.read(buffer_samples, image_size * sample_num);

		std::fstream stream_labels;
		stream_labels.open("resources/mnist/train-labels.idx1-ubyte", std::ios_base::in | std::ios_base::binary);
		stream_labels.seekg(8);
		char* buffer_labels = new char[sample_num];
		stream_labels.read(buffer_labels, sample_num);

		for (size_t i = 0; i < sample_num; i++)
		{
			result.push_back({ vec(image_size), vec(10) });

			for (size_t j = 0; j < image_size; j++)
			{
				result[i].first[j] = (unsigned char)(buffer_samples[i * image_size + j]);
			}

			result[i].second.set(0.1);
			result[i].second[buffer_labels[i]] = 0.9;
		}

		delete[] buffer_samples;
		stream_samples.close();

		delete[] buffer_labels;
		stream_labels.close();
	}
	return result;
}

std::vector<std::pair<vec, vec>> read_testing_sets()
{
	constexpr const size_t sample_num = 10000;
	constexpr const size_t image_size = 28 * 28;
	std::vector<std::pair<vec, vec>> result;
	{
		std::fstream stream_samples;
		stream_samples.open("resources/mnist/t10k-images.idx3-ubyte", std::ios_base::in | std::ios_base::binary);
		stream_samples.seekg(16);
		char* buffer_samples = new char[image_size * sample_num];
		stream_samples.read(buffer_samples, image_size * sample_num);

		std::fstream stream_labels;
		stream_labels.open("resources/mnist/t10k-labels.idx1-ubyte", std::ios_base::in | std::ios_base::binary);
		stream_labels.seekg(8);
		char* buffer_labels = new char[sample_num];
		stream_labels.read(buffer_labels, sample_num);

		for (size_t i = 0; i < sample_num; i++)
		{
			result.push_back({ vec(image_size), vec(10) });

			for (size_t j = 0; j < image_size; j++)
			{
				result[i].first[j] = (unsigned char)(buffer_samples[i * image_size + j]);
			}

			result[i].second.set(0.1);
			result[i].second[buffer_labels[i]] = 0.9;
		}

		delete[] buffer_samples;
		stream_samples.close();

		delete[] buffer_labels;
		stream_labels.close();
	}
	return result;
}

int main()
{
	auto training_sets = read_training_sets();
	auto testing_sets = read_testing_sets();
	bb::fnn network({ 784, 300, 10 });
	double prev_cr = 0.0;
	double cr = 0.0; // correct rate
	int epoch = 0;
	while (cr >= prev_cr)
	{
		size_t correct_num = 0;
		for (const auto& [sample, label] : testing_sets)
		{
			network.set_input(sample);
			network.predict();
			if (network.get_output().max_index() == label.max_index())
				correct_num += 1;
		}
		prev_cr = cr;
		cr = (double)correct_num / (double)testing_sets.size();
		system("cls");
		printf_s("On epoch: %d\nCorrect Rate: %d%%\n", epoch, (int)(cr * 100.0));
		network.train(training_sets, 0.01, 10);
		epoch += 1;
	}
	return 0;
}