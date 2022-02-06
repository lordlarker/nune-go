// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"math"

	"github.com/lordlarker/nune/internal/cpd"
)

// PwiseOp performs a pointwise operation
// over each element of the Tensor.
func (t Tensor[T]) PwiseOp(f func(T) T) Tensor[T] {
	cpd.Pointwise(t.storage.Load(), f)

	return t
}

// Abs computes the absolute value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Abs() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Abs(float64(x)))
	})

	return t
}

// Sin computes the sine value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Sin() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Sin(float64(x)))
	})

	return t
}

// Cos computes the cosine value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Cos() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Cos(float64(x)))
	})

	return t
}

// Tan computes the tan value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Tan() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Tan(float64(x)))
	})

	return t
}

// Log computes the natural log value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Log() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Log(float64(x)))
	})

	return t
}

// Log2 computes the binary log value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Log2() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Log2(float64(x)))
	})

	return t
}

// Log10 computes the decimal log value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Log10() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Log10(float64(x)))
	})

	return t
}

// Exp computes the base-e exponential value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Exp() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Exp(float64(x)))
	})

	return t
}

// Pow computes the base-value exponential of p of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Pow(p T) Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Pow(float64(x), float64(p)))
	})

	return t
}

// Sqrt computes the square root value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Sqrt() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Sqrt(float64(x)))
	})

	return t
}

// Round computes the nearest integer value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Round() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Round(float64(x)))
	})

	return t
}

// Floor computes the nearest lesser integer value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Floor() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Floor(float64(x)))
	})

	return t
}

// Ceil computes the nearest greater value of each
// element of the Tensor and returns the Tensor.
func (t Tensor[T]) Ceil() Tensor[T] {
	cpd.Pointwise(t.storage.Load(), func(x T) T {
		return T(math.Ceil(float64(x)))
	})

	return t
}
