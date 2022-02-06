// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"github.com/lordlarker/nune"
	"github.com/lordlarker/nune/internal/cpd"
)

// A Tensor is a generic, n-dimensional numerical type.
type Tensor[T nune.Numeric] struct {
	storage *cpd.Storage[T] // the storage that holds the Tensor's data
	layout  *layout         // the layout that holds the Tensor's indexing scheme
}

func newStorage[T nune.Numeric](data []T) *cpd.Storage[T] {
	return cpd.NewStorage(data)
}
