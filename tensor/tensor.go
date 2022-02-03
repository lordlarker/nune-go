// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import "nune"

// A Tensor is a generic, n-dimensional numerical type.
type Tensor[T nune.Numeric] struct {
	data    []T   // the underlying data of the Tensor
	shape   []int // the shape of the Tensor
	strides []int // the number of elements to step in each axis
}
