// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor_test

import (
	"github.com/lordlarker/nune/tensor"
	"testing"
)

func TestFrom(t *testing.T) {
	if panics(func() {
		tensor.From[int]("world")
	}) {
		t.Fail()
	}

	if panics(func() {
		tensor.From[int](0.0)
	}) {
		t.Fail()
	}

	if panics(func() {
		tensor.From[int](tensor.Range[uint](0, 5, 1))
	}) {
		t.Fail()
	}

	if panics(func() {
		tensor.From[int]([3]uint{1, 2, 3})
	}) {
		t.Fail()
	}

	if panics(func() {
		tensor.From[int]([2][2]uint{{1, 2}, {3, 4}})
	}) {
		t.Fail()
	}

	if panics(func() {
		tensor.From[int]([]uint{1, 2, 3})
	}) {
		t.Fail()
	}

	if panics(func() {
		tensor.From[int]([][]uint{{1, 2}, {3, 4}})
	}) {
		t.Fail()
	}

	if !panics(func() {
		tensor.From[int]([][]uint{{1, 2}, {3, 4, 5}})
	}) {
		t.Fail()
	}
}

func TestFull(t *testing.T) {
	if !panics(func() {
		tensor.Full(0, []int{})
	}) {
		t.Fail()
	}

	if !panics(func() {
		tensor.Full(0, []int{0})
	}) {
		t.Fail()
	}

	tnsr := tensor.Full(0, []int{5, 2})
	if len(tnsr.Ravel()) != 10 {
		t.Fail()
	}
}
