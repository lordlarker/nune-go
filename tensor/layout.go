// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import "github.com/lordlarker/nune/internal/slice"

type layout struct {
	shape   []int
	strides []int
	offset int
}

func newLayout(shape []int) *layout {
	l := new(layout)
	l.shape = shape

	if len(shape) != 0 {
		strides := slice.WithLen[int](len(shape))
		strides[0] = slice.Prod(shape[1:])

		for i := 1; i < len(shape); i++ {
			strides[i] = strides[i-1] / shape[i]
		}

		l.strides = strides
	}

	return l
}

func (l *layout) Rank() int {
	return len(l.shape)
}

func (l *layout) Shape() []int {
	return l.shape
}

func (l *layout) SetShape(shape []int) {
	l.shape = shape
}

func (l *layout) Strides() []int {
	return l.strides
}

func (l *layout) SetStrides(strides []int) {
	l.strides = strides
}

func (l *layout) Offset() int {
	return l.offset
}

func (l *layout) SetOffset(offset int) {
	l.offset = offset
}

func (l *layout) Copy() *layout {
	c := new(layout)
	c.shape = slice.Copy(l.shape)
	c.strides = slice.Copy(l.strides)

	return c
}