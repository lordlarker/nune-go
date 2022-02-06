// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/lordlarker/nune/internal/cpd"
)

func (t *Tensor[T]) ReductOp(f func([]T) T) T {
	return cpd.Reduct(t.storage.Load(), f)
}

// Min returns the minimum value of all elements in the Tensor.
func (t *Tensor[T]) Min() T {
	return t.ReductOp(func(s []T) T {
		m := s[0]
		for i := 1; i < len(s); i++ {
			if s[i] < m {
				m = s[i]
			}
		}
		return m
	})
}

// Max returns the maximum value of all elements in the Tensor.
func (t *Tensor[T]) Max() T {
	return t.ReductOp(func(s []T) T {
		m := s[0]
		for i := 1; i < len(s); i++ {
			if s[i] > m {
				m = s[i]
			}
		}
		return m
	})
}

// Mean returns the mean value of all elements in the Tensor.
func (t *Tensor[T]) Mean() T {
	return t.ReductOp(func(s []T) T {
		var sum T
		for i := 0; i < len(s); i++ {
			sum += s[i]
		}
		return sum / T(len(s))
	})
}

// Sum returns the sum of all elements in the Tensor.
func (t *Tensor[T]) Sum() T {
	return t.ReductOp(func(s []T) T {
		var sum T
		for i := 0; i < len(s); i++ {
			sum += s[i]
		}
		return sum
	})
}

// Prod returns the product of all elements in the Tensor.
func (t *Tensor[T]) Prod() T {
	return t.ReductOp(func(s []T) T {
		var prod T = 1
		for i := 0; i < len(s); i++ {
			prod *= s[i]
		}
		return prod
	})
}
