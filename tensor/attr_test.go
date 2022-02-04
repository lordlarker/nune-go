// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor_test

import (
	"nune/tensor"
	"testing"
	"unsafe"
)

func TestNumby(t *testing.T) {
	tnsr := tensor.Range[int](0, 5, 1)
	var i int

	if tnsr.Numby() != 5*unsafe.Sizeof(i) {
		t.Fail()
	}
}

func TestSize(t *testing.T) {
	tnsr := tensor.Range[int](0, 10, 1)
	if tnsr.Size() != 10 {
		t.Fail()
	}

	tnsr = tnsr.Reshape(5, 2)
	if tnsr.Size() != 10 {
		t.Fail()
	}

	if tnsr.Size(0) != 5 {
		t.Fail()
	}

	if !panics(func() {
		tnsr.Size(-1)
	}) {
		t.Fail()
	}

	if !panics(func() {
		tnsr.Size(2)
	}) {
		t.Fail()
	}

	if !panics(func() {
		tnsr.Size(1, 2)
	}) {
		t.Fail()
	}
}

func TestMemSize(t *testing.T) {
	tnsr := tensor.Range[int](0, 5, 1)
	var i int

	if tnsr.MemSize() != (5+2)*unsafe.Sizeof(i) { // data: 5, other: 2
		t.Fail()
	}
}

func TestBroadable(t *testing.T) {
	tnsr := tensor.Range[int](0, 10, 1)

	if tnsr.Broadable() { // nil
		t.Fail()
	}

	if tnsr.Broadable(0) || tnsr.Broadable(-1) {
		t.Fail()
	}

	tnsr = tnsr.Reshape(10, 1)
	if tnsr.Broadable(5) {
		t.Fail()
	}

	if tnsr.Broadable(5, 10) {
		t.Fail()
	}

	if !tnsr.Broadable(10, 5) {
		t.Fail()
	}

	if !tnsr.Broadable(5, 10, 1) {
		t.Fail()
	}
}
