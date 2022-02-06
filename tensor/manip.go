// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/lordlarker/nune"
	"github.com/lordlarker/nune/internal/slice"
)

// Cast casts a Tensor's underlying type to the given numeric type.
func Cast[T nune.Numeric, U nune.Numeric](t Tensor[U]) Tensor[T] {
	c := slice.WithLen[T](t.Numel())
	for i := 0; i < len(c); i++ {
		c[i] = T(t.storage.Index(i))
	}

	return Tensor[T]{
		storage:    newStorage(c),
		layout:   t.layout,
	}
}

// Copy copies the Tensor's fields into
// a new Tensor and returns it.
func (t Tensor[T]) Copy() Tensor[T] {
	return Tensor[T]{
		storage: t.storage.Copy(),
		layout: t.layout.Copy(),
	}
}

// Assign attempts to unwrap the given value and assign
// the Tensor's data buffer to it, if it matches the Tensor's shape.
func (t Tensor[T]) Assign(v any) Tensor[T] {
	defer func() {
		if r := recover(); r != nil {
			panic("nune/tensor: Tensor.AssignTo could not assign the Tensor's data buffer to the given value")
		}
	}()

	other := From[T](v)
	if slice.Equal(t.Shape(), other.Shape()) {
		t.storage.Dump(other.storage.Load())
		return t
	} else {
		panic("") // trigger a recover then panic with the same error above
	}
}

// Reshape modifies the Tensor's underlying shape buffer
// and returns the Tensor.
func (t Tensor[T]) Reshape(s ...int) Tensor[T] {
	if len(s) == 0 && t.Numel() <= 1 {
		return Tensor[T]{
			storage: t.storage,
			layout: newLayout(nil),
		}
	} else {
		assertGoodShape(s...)
		assertArgsBounds(len(s), t.Rank()-1)

		return Tensor[T]{
			storage: t.storage,
			layout: newLayout(slice.Copy(s)),
		}
	}
}

// Index returns a view over an index of the Tensor.
func (t Tensor[T]) Index(indices ...int) Tensor[T] {
	assertArgsBounds(len(indices), t.Rank())

	for i, idx := range indices {
		assertAxisBounds(idx, t.Size(i))
	}

	var offset int

	for i, idx := range indices {
		offset += idx * t.Strides()[i]
	}

	return Tensor[T]{
		storage: t.storage,
		layout: newLayout(slice.Copy(t.Shape()[len(indices):])),
	}
	// return Tensor[T]{
	// 	data:    t.data[offset : offset+t.strides[len(indices)-1]],
	// 	shape:   slice.Copy(t.shape[len(indices):]),
	// 	strides: stridesFromShape(t.shape[len(indices):]),
	// }
}

// Slice returns a view over a slice of the Tensor.
func (t *Tensor[T]) Slice(start, end int) Tensor[T] {
	assertGoodShape(t.Shape()...) // make sure Tensor rank is not 0
	assertGoodInterval(start, end, [2]int{0, t.Size(0)})

	newshape := slice.WithLen[int](t.Rank())
	newshape[0] = end - start
	copy(newshape[1:], t.Shape()[1:])

	return Tensor[T]{
		storage: t.storage,
		layout: newLayout(newshape),
	}

	// return Tensor[T]{
	// 	data:    t.data[start*t.strides[0] : end*t.strides[0]],
	// 	shape:   newshape,
	// 	strides: stridesFromShape(newshape),
	// }
}

// Reverse reverses the order of the elements of the Tensor.
func (t Tensor[T]) Reverse() Tensor[T] {
	for i, j := 0, t.Numel()-1; i < j; i, j = i+1, j-1 {
		t.storage.SetIndex(i, t.storage.Index(j))
		t.storage.SetIndex(j, t.storage.Index(i))
	}

	return t
}
