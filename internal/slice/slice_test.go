// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slice_test

import (
	"github.com/lordlarker/nune/internal/slice"
	"testing"
)

func TestProd(t *testing.T) {
	s1 := []int{5}
	s2 := []int{5, 5}

	if slice.Prod(s1) != 5 {
		t.Fail()
	}

	if slice.Prod(s2) != 25 {
		t.Fail()
	}
}

func TestEqual(t *testing.T) {
	s1 := []int{5}
	s2 := []int{5, 5}

	if slice.Equal(s1, s2) == true {
		t.Fail()
	}

	s1 = []int{0, 5}
	if slice.Equal(s1, s2) == true {
		t.Fail()
	}

	s1 = []int{5, 5}
	if slice.Equal(s1, s2) == false {
		t.Fail()
	}
}

func TestCopy(t *testing.T) {
	s1 := []int{5, 5}
	s2 := slice.Copy(s1)

	if len(s2) != len(s1) {
		t.Fail()
	}

	s1[0] = 0
	if s2[0] == 0 {
		t.Fail()
	}
}

func TestWithLen(t *testing.T) {
	s := slice.WithLen[int](5)

	if len(s) != 5 {
		t.Fail()
	}
}

func TestWithCap(t *testing.T) {
	s := slice.WithCap[int](5)

	if len(s) != 0 {
		t.Fail()
	}

	if cap(s) != 5 {
		t.Fail()
	}
}
