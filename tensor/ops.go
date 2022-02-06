// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/lordlarker/nune/internal/cpd"
	"github.com/lordlarker/nune/internal/slice"
)

// Add takes a Tensor and performs element-wise addition,
// by reference, over the two Tensor's elements, and then
// returns the resulting Tensor.
func (t *Tensor[T]) Add(other *Tensor[T]) *Tensor[T] {
	if !slice.Equal(t.Shape(), other.Shape()) {
		panic("nune/tensor: Tensor.Add received a Tensor with a different shape than its own")
	}

	cpd.Op(t.storage.Load(), other.storage.Load(), t.storage.Load(), func(t1, t2 T) T {
		return t1 + t2
	})

	return t
}

// Sub takes a Tensor and performs element-wise subtraction,
// by reference, over the two Tensor's elements, and then
// returns the resulting Tensor.
func (t *Tensor[T]) Sub(other *Tensor[T]) *Tensor[T] {
	if !slice.Equal(t.Shape(), other.Shape()) {
		panic("nune/tensor: Tensor.Sub received a Tensor with a different shape than its own")
	}

	cpd.Op(t.storage.Load(), other.storage.Load(), t.storage.Load(), func(t1, t2 T) T {
		return t1 - t2
	})

	return t
}

// Mul takes a Tensor and performs element-wise multiplication,
// by reference, over the two Tensor's elements, and then
// returns the resulting Tensor.
func (t *Tensor[T]) Mul(other *Tensor[T]) *Tensor[T] {
	if !slice.Equal(t.Shape(), other.Shape()) {
		panic("nune/tensor: Tensor.Mul received a Tensor with a different shape than its own")
	}

	cpd.Op(t.storage.Load(), other.storage.Load(), t.storage.Load(), func(t1, t2 T) T {
		return t1 * t2
	})

	return t
}

// Div takes a Tensor and performs element-wise division,
// by reference, over the two Tensor's elements, and then
// returns the resulting Tensor.
func (t *Tensor[T]) Div(other *Tensor[T]) *Tensor[T] {
	if !slice.Equal(t.Shape(), other.Shape()) {
		panic("nune/tensor: Tensor.Div received a Tensor with a different shape than its own")
	}

	cpd.Op(t.storage.Load(), other.storage.Load(), t.storage.Load(), func(t1, t2 T) T {
		if t2 == T(0) {
			panic("nune/tensor: division by zero occurred in Tensor.Div")
		}

		return t1 / t2
	})

	return t
}
