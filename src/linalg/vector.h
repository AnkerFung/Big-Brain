#pragma once

#include <vector>
#include <cassert>
#include <functional>

namespace linalg
{
	class vec
	{
	public:
		vec(size_t _Dim)
			:
			dim(_Dim),
			v(_Dim)
		{}
		vec(size_t _Dim, const double* _V)
			:
			dim(_Dim),
			v(_Dim)
		{
			memcpy_s(v.data(), dim * sizeof(double), (const void*)_V, dim * sizeof(double));
		}
		vec(const std::vector<double>& _V)
			:
			dim(_V.size()),
			v(_V)
		{}
		vec(std::vector<double>&& _V)
			:
			dim(_V.size()),
			v(std::move(_V))
		{}
		size_t dimension() const
		{
			return dim;
		}
		void set(double _V)
		{
			for (double& value : v)
			{
				value = _V;
			}
		}
		size_t max_index() const
		{
			size_t index = 0;
			for (size_t i = 0; i < v.size(); i++)
			{
				if (v[i] > v[index])
					index = i;
			}
			return index;
		}
		double* data()
		{
			return v.data();
		}
		const double* data() const
		{
			return v.data();
		}
		double& operator[](size_t i)
		{
			assert(i < dim);
			return v[i];
		}
		double operator[](size_t i) const
		{
			assert(i < dim);
			return v[i];
		}
		vec operator=(const vec& obj)
		{
			assert(dim == obj.dim);
			v = obj.v;
			return *this;
		}
		friend vec operator+(const vec& a, const vec& b)
		{
			assert(a.dim == b.dim);
			size_t dim = a.dim;
			std::vector<double> _V(dim);
			for (size_t i = 0; i < dim; i++)
			{
				_V[i] = a[i] + b[i];
			}
			return vec(std::move(_V));
		}
		friend vec operator-(const vec& a, const vec& b)
		{
			assert(a.dim == b.dim);
			size_t dim = a.dim;
			std::vector<double> _V(dim);
			for (size_t i = 0; i < dim; i++)
			{
				_V[i] = a[i] - b[i];
			}
			return vec(std::move(_V));
		}
		friend vec operator*(const vec& a, double b)
		{
			size_t dim = a.dim;
			std::vector<double> _V(dim);
			for (size_t i = 0; i < dim; i++)
			{
				_V[i] = a[i] * b;
			}
			return vec(std::move(_V));
		}
		friend vec operator*(const vec& a, const vec& b)
		{
			assert(a.dim == b.dim);
			size_t dim = a.dim;
			std::vector<double> _V(dim);
			for (size_t i = 0; i < dim; i++)
			{
				_V[i] = a[i] * b[i];
			}
			return vec(std::move(_V));
		}
	private:
		const size_t dim;
		std::vector<double> v;
	};

	inline double dot(const vec& a, const vec& b)
	{
		assert(a.dimension() == b.dimension());
		double result = 0.0;
		size_t dim = a.dimension();
		for (size_t i = 0; i < dim; i++)
		{
			result += a[i] * b[i];
		}
		return result;
	}

	inline vec transform(vec v, std::function<double(double)> func)
	{
		vec result(v.dimension());
		for (size_t i = 0; i < v.dimension(); i++)
		{
			result[i] = func(v[i]);
		}
		return result;
	}
}