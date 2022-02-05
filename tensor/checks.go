// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

// List of errors.
var (
	
)

// checkEmptyShape returns true if the given
// shape s is empty and false otherwise.
func checkEmptyShape(s []int) bool {
	if len(s) == 0 {
		return true
	} else {
		return false
	}
}

// checkPosAxes returns true if the dimensions
// of the shape's axes are all strictly positive
// and false otherwise.
func checkPosAxes(s []int) bool {
	for _, a := range s {
		if a <= 0 {
			return false
		}
	}
	return true
}

// checkInRange returns true if a value x is in
// the interval [start, end) and false otherwise.
func checkInRange(x, start, end int) bool {
	if x < start || x >= end {
		return false
	} else {
		return true
	}
}
