// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"nune"
	"nune/internal/slice"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Numeric, U nune.Numeric](t Tensor[U]) Tensor[T] {
	c := slice.WithLen[T](len(t.data))
	for i := 0; i < len(c); i++ {
		c[i] = T(t.data[i])
	}

	return Tensor[T]{
		data:    c,
		shape:   t.shape,
		strides: t.strides,
	}
}

// Copy copies the Tensor's fields into
// a new Tensor and returns it.
func (t Tensor[T]) Copy() Tensor[T] {
	return Tensor[T]{
		data:    t.Ravel(),   // Ravel already copies the slice
		shape:   t.Shape(),   // Shape already copies the slice
		strides: t.Strides(), // Strides already copies the slice
	}
}

// AssignTo attempts to unwrap the given value and assign
// the Tensor's data buffer to it, if it matches the Tensor's shape.
func (t Tensor[T]) AssignTo(v any) Tensor[T] {
	defer func() {
		if r := recover(); r != nil {
			panic("nune/tensor: Tensor.AssignTo could not assign the Tensor's data buffer to the given value")
		}
	}()

	other := From[T](v)
	if slice.Equal(t.shape, other.shape) {
		t.data = other.data
		return t
	} else {
		panic("") // trigger a recover then panic with the same error above
	}
}

// Reshape modifies the Tensor's underlying shape buffer
// and returns the Tensor.
func (t Tensor[T]) Reshape(s ...int) Tensor[T] {
	if len(s) == 0 && len(t.data) <= 1 {
		t.shape = nil
	} else {
		if checkEmptyShape(s) {
			panic("nune/tensor: Tensor.Reshape received an empty shape")
		} else if !checkPosAxes(s) {
			panic("nune/tensor: Tensor.Reshape received a shape with non-strictly positive axes")
		} else if len(t.data) != slice.Prod(s) {
			panic("nune/tensor: Tensor.Reshape received a shape whose size doesn't match the Tensor's number of elements")
		}

		t.shape = slice.Copy(s)
		t.strides = stridesFromShape(t.shape)
	}
	return t
}

// Index returns a view over an index of the Tensor.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
	if len(indices) > len(t.shape) {
		panic("nune/tensor: Tensor.Index received more indices than its number of axes")
	}

	for i, idx := range indices {
		if !checkInRange(idx, 0, t.shape[i]) {
			panic("nune/tensor: Tensor.Index received an out of bounds index")
		}
	}

	var offset int

	for i, idx := range indices {
		offset += idx * t.strides[i]
	}

	return Tensor[T]{
		data:    t.data[offset : offset+t.strides[len(indices)-1]],
		shape:   slice.Copy(t.shape[len(indices):]),
		strides: stridesFromShape(t.shape[len(indices):]),
	}
}

// Slice returns a view over a slice of the Tensor.
func (t *Tensor[T]) Slice(start, end int) Tensor[T] {
	if len(t.shape) == 0 {
		panic("nune/tensor: Tensor.Slice cannot slice rank 0 Tensor")
	} else if start < 0 {
		panic("nune/tensor: Tensor.Slice received an endpoint less than zero")
	} else if start > end {
		panic("nune/tensor: Tensor.Slice received a descending interval")
	} else if end > t.Size(0) {
		panic("nune/tensor: Tensor.Slice received an endpoint greater than the Tensor's size at axis 0")
	}

	newshape := slice.WithLen[int](len(t.shape))
	newshape[0] = end - start
	copy(newshape[1:], t.shape[1:])

	return Tensor[T]{
		data:    t.data[start*t.strides[0] : end*t.strides[0]],
		shape:   newshape,
		strides: stridesFromShape(newshape),
	}
}

// Reverse reverses the order of the elements of the Tensor.
func (t Tensor[T]) Reverse() Tensor[T] {
	for i, j := 0, len(t.data)-1; i < j; i, j = i+1, j-1 {
		t.data[i], t.data[j] = t.data[j], t.data[i]
	}

	return t
}
