// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"unsafe"

	"github.com/lordlarker/nune/internal/slice"
	"github.com/lordlarker/nune/internal/utils"
)

// Ravel returns a copy of the Tensor's 1-dimensional data buffer.
func (t *Tensor[T]) Ravel() []T {
	return t.storage.Load()
}

// Numel returns the number of elements in the Tensor's data buffer.
func (t *Tensor[T]) Numel() int {
	return t.storage.Numel()
}

// Numby returns the size in bytes occupied by all elements
// of the Tensor's underlying data buffer.
func (t *Tensor[T]) Numby() uintptr {
	return t.storage.Numby()
}

// Rank returns the Tensor's rank
// (the number of axes in the Tensor's shape).
func (t *Tensor[T]) Rank() int {
	return t.layout.Rank()
}

// Shape returns a copy of the Tensor's shape.
func (t *Tensor[T]) Shape() []int {
	return slice.Copy(t.layout.Shape())
}

// Strides returns a copy of the Tensor's strides.
func (t *Tensor[T]) Strides() []int {
	return slice.Copy(t.layout.Strides())
}

// Size returns the Tensor's total shape size.
// If axis is specified, the number of dimensions at
// that axis is returned.
func (t *Tensor[T]) Size(axis ...int) int {
	args := len(axis)
	assertArgsBounds(args, 1)

	if args == 0 {
		return slice.Prod(t.layout.Shape())
	} else {
		assertAxisBounds(axis[0], t.Rank())
		return t.Shape()[axis[0]]
	}
}

// MemSize returns the size in bytes occupied by all fields
// that make up the Tensor.
func (t *Tensor[T]) MemSize() uintptr {
	var i int
	shapeSize := unsafe.Sizeof(i) * uintptr(t.Rank())
	stridesSize := unsafe.Sizeof(i) * uintptr(t.Rank())

	return t.Numby() + shapeSize + stridesSize
}

// Broadable returns whether or not the Tensor can be
// broadcasted to the given shape.
func (t *Tensor[T]) Broadable(shape ...int) bool {
	if utils.Panics(func() {
		assertGoodShape(shape...)
		assertArgsBounds(len(shape), t.Rank()-1)
	}) {
		return false
	}

	var s []int

	if t.Rank() < len(shape) {
		s = slice.WithLen[int](len(shape))
		for i := 0; i < len(shape)-t.Rank(); i++ {
			s[i] = 1
		}
		copy(s[len(shape)-t.Rank():], t.layout.Shape())
	} else {
		s = t.Shape()
	}

	for i := 0; i < len(shape); i++ {
		if s[i] != shape[i] && s[i] != 1 {
			return false
		}
	}

	return true
}
