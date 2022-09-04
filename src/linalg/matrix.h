#pragma once

#include <vector>

#include "vector.h"

namespace linalg
{
	class mat
	{
	public:
		mat(size_t _Row, size_t _Column)
			:
			row(_Row),
			column(_Column),
			v(row * column)
		{}
		mat(size_t _Row, size_t _Column, const double* _V)
			:
			row(_Row),
			column(_Column),
			v(row * column)
		{
			rsize_t size = row * column * sizeof(double);
			memcpy_s(v.data(), row * column * sizeof(double), (const void*)_V, size);
		}
		mat(size_t _Row, size_t _Column, const std::vector<double>& _V)
			:
			row(_Row),
			column(_Column),
			v(row * column)
		{
			assert(_V.size() != row * column);
			v.assign(_V.cbegin(), _V.cend());
		}
		mat(size_t _Row, size_t _Column, std::vector<double>&& _V)
			:
			row(_Row),
			column(_Column),
			v(std::move(_V))
		{
			assert(v.size() != row * column);
		}
		std::pair<size_t, size_t> dimensions() const
		{
			return { row, column };
		}
		double& get(size_t i, size_t j)
		{
			assert(i < row
				&& j < column);
			return v[i + j * row];
		}
		double get(size_t i, size_t j) const
		{
			assert(i < row
				&& j < column);
			return v[i + j * row];
		}
		double* data()
		{
			return v.data();
		}
		const double* data() const
		{
			return v.data();
		}
		mat operator=(const mat& obj)
		{
			assert(row    == obj.row
				&& column == obj.column);
			v = obj.v;
			return *this;
		}
		friend mat operator+(const mat& a, const mat& b)
		{
			assert(a.row    == b.row
				&& a.column == b.column);
			size_t size = a.row * a.column;
			std::vector<double> _V(size);
			for (size_t i = 0; i < size; i++)
			{
				_V[i] = a.v[i] + b.v[i];
			}
			return mat(a.row, a.column, std::move(_V));
		}
		friend mat operator-(const mat& a, const mat& b)
		{
			assert(a.row == b.row
				&& a.column == b.column);
			size_t size = a.row * a.column;
			std::vector<double> _V(size);
			for (size_t i = 0; i < size; i++)
			{
				_V[i] = a.v[i] - b.v[i];
			}
			return mat(a.row, a.column, std::move(_V));
		}
		friend mat operator*(double scale, const mat& m)
		{
			size_t size = m.row * m.column;
			std::vector<double> _V(size);
			for (size_t i = 0; i < size; i++)
			{
				_V[i] = m.v[i] * scale;
			}
			return mat(m.row, m.column, std::move(_V));
		}
		friend mat operator*(const mat& m, double scale)
		{
			size_t size = m.row * m.column;
			std::vector<double> _V(size);
			for (size_t i = 0; i < size; i++)
			{
				_V[i] = m.v[i] * scale;
			}
			return mat(m.row, m.column, std::move(_V));
		}
		friend vec operator*(const mat& m, const vec& v)
		{
			assert(m.row == v.dimension());
			std::vector<double> _V(m.column);
			for (size_t i = 0; i < m.row; i++)
			{
				for (size_t j = 0; j < m.column; j++)
				{
					_V[j] += m.get(i, j) * v[i];
				}
			}
			return vec(std::move(_V));
		}
		friend mat operator*(const mat& a, const mat& b)
		{
			assert(a.row == b.column);
			std::vector<double> _V(b.row * a.column);
			for (size_t i = 0; i < b.row; i++)
			{
				for (size_t j = 0; j < a.column; j++)
				{
					for (size_t k = 0; k < b.column; k++)
					{
						_V[i + j * b.row] += a.get(k, j) * b.get(i, k);
					}
				}
			}
			return mat(b.row, a.column, std::move(_V));
		}
	private:
		const size_t row, column;
		std::vector<double> v;
	};

	inline vec multiply_transposed(const mat& m, const vec& v)
	{
		size_t row = m.dimensions().first;
		size_t column = m.dimensions().second;
		assert(column == v.dimension());
		std::vector<double> _V(row);
		for (size_t i = 0; i < column; i++)
		{
			for (size_t j = 0; j < row; j++)
			{
				_V[j] += m.get(j, i) * v[i];
			}
		}
		return vec(std::move(_V));
	}
}