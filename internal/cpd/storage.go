// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpd

import (
	"unsafe"

	"github.com/lordlarker/nune"
	"github.com/lordlarker/nune/internal/slice"
)

type Storage[T nune.Numeric] struct {
	data []T
}

func NewStorage[T nune.Numeric](data []T) *Storage[T] {
	s := new(Storage[T])
	s.data = data

	return s
}

func (s *Storage[T]) Numel() int {
	return len(s.data)
}

func (s *Storage[T]) Numby() uintptr {
	return unsafe.Sizeof(T(0)) * uintptr(len(s.data))
}

func (s *Storage[T]) Load() []T {
	return s.data
}

func (s *Storage[T]) Dump(data []T) {
	s.data = data
}

func (s *Storage[T]) Index(idx int) T {
	return s.data[idx]
}

func (s *Storage[T]) SetIndex(idx int, x T) {
	s.data[idx] = x
}

func (s *Storage[T]) Slice(start, end int) []T {
	return s.data[start:end]
}

func (s *Storage[T]) SetSlice(start, end int, x []T) {
	copy(s.data[start:end], x)
}

func (s *Storage[T]) Copy() *Storage[T] {
	c := new(Storage[T])
	c.data = slice.Copy(s.data)

	return c
}
