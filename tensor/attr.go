// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"nune/internal/slice"
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
	if len(axis) == 0 {
		return slice.Prod(t.shape)
	} else if len(axis) == 1 {
		if axis := axis[0]; checkInRange(axis, 0, len(t.shape)) {
			return t.shape[axis]
		} else {
			panic("nune/tensor: Size received an out of bounds axis")
		}
	} else {
		panic("nune/tensor: Size received more than 1 axis")
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
	if checkEmptyShape(shape) {
		return false
	} else if len(t.shape) > len(shape) {
		return false
	} else if !checkPosAxes(shape) {
		return false
	}

	var s []int

	if len(t.shape) < len(shape) {
		s = slice.WithLen[int](len(shape))
		for i := 0; i < len(shape) - len(t.shape); i++ {
			s[i] = 1
		}
		copy(s[len(shape)-len(t.shape)+1:], t.shape)
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