// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/lordlarker/nune/internal/slice"
)

// stridesFromShape returns indexing strides for a given shape.
func stridesFromShape(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}

	strides := slice.WithLen[int](len(shape))
	strides[0] = slice.Prod(shape[1:])

	for i := 1; i < len(shape); i++ {
		strides[i] = strides[i-1] / shape[i]
	}

	return strides
}
