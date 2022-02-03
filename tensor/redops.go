// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

// Min returns the minimum value of all elements in the Tensor.
func (t *Tensor[T]) Min() T {
	m := t.data[0]
	for i := 1; i < t.Numel(); i++ {
		if t.data[i] < m {
			m = t.data[i]
		}
	}

	return m
}

// Max returns the maximum value of all elements in the Tensor.
func (t *Tensor[T]) Max() T {
	m := t.data[0]
	for i := 1; i < t.Numel(); i++ {
		if t.data[i] > m {
			m = t.data[i]
		}
	}

	return m
}