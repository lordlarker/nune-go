// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slice

// Prod returns the sum-product of the elements of a slice.
func Prod(s []int) int {
	p := 1
	for _, a := range s {
		p *= a
	}

	return p
}

// Equal returns whether or not two slices are equal.
func Equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// Copy makes a new slice and copies the given slice's elements into it.
func Copy[T any](s []T) []T {
	cp := make([]T, len(s))
	copy(cp, s)

	return cp
}

// WithLen makes a new slice with the given length and returns it.
func WithLen[T any](l int) []T {
	return make([]T, l)
}

// WithCap makes a new slice with the given capacity and returns it.
func WithCap[T any](c int) []T {
	return make([]T, 0, c)
}
