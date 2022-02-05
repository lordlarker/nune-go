// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/lordlarker/nune/internal/slice"
	"github.com/lordlarker/nune/internal/utils"
	"unsafe"
)

// Ravel returns a copy of the Tensor's 1-dimensional data buffer.
func (t Tensor[T]) Ravel() []T {
	return slice.Copy(t.data)
}

// Numel returns the number of elements in the Tensor's data buffer.
func (t Tensor[T]) Numel() int {
	return len(t.data)
}

// Numby returns the size in bytes occupied by all elements
// of the Tensor's underlying data buffer.
func (t Tensor[T]) Numby() uintptr {
	var n T
	return unsafe.Sizeof(n) * uintptr(len(t.data))
}

// Rank returns the Tensor's rank
// (the number of axes in the Tensor's shape).
func (t Tensor[T]) Rank() int {
	return len(t.shape)
}

// Shape returns a copy of the Tensor's shape.
func (t Tensor[T]) Shape() []int {
	return slice.Copy(t.shape)
}

// Strides returns a copy of the Tensor's strides.
func (t Tensor[T]) Strides() []int {
	return slice.Copy(t.strides)
}

// Size returns the Tensor's total shape size.
// If axis is specified, the number of dimensions at
// that axis is returned.
func (t Tensor[T]) Size(axis ...int) int {
	args := len(axis)
	assertArgsBounds(args, 1)

	if args == 0 {
		return slice.Prod(t.shape)
	} else {
		assertAxisBounds(axis[0], t.Rank())
		return t.shape[axis[0]]
	}
}

// MemSize returns the size in bytes occupied by all fields
// that make up the Tensor.
func (t Tensor[T]) MemSize() uintptr {
	var i int
	shapeSize := unsafe.Sizeof(i) * uintptr(len(t.shape))
	stridesSize := unsafe.Sizeof(i) * uintptr(len(t.strides))

	return t.Numby() + shapeSize + stridesSize
}

// Broadable returns whether or not the Tensor can be
// broadcasted to the given shape.
func (t Tensor[T]) Broadable(shape ...int) bool {
	if utils.Panics(func() {
		assertGoodShape(shape...)
		assertArgsBounds(len(shape), t.Rank()-1)
	}) {
		return false
	}

	var s []int

	if len(t.shape) < len(shape) {
		s = slice.WithLen[int](len(shape))
		for i := 0; i < len(shape)-len(t.shape); i++ {
			s[i] = 1
		}
		copy(s[len(shape)-len(t.shape):], t.shape)
	} else {
		s = t.shape
	}

	for i := 0; i < len(shape); i++ {
		if s[i] != shape[i] && s[i] != 1 {
			return false
		}
	}

	return true
}
