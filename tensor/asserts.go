// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import "errors"

// List of errors.
var (
	// errBadShape occurs when a shape is nil or a has axes whose
	// dimensions are less than or equal to zero.
	errBadShape = errors.New("nune: received a bad shape")

	// errAxisBounds occurs when an axis is out of
	// (0, tensor rank) bounds.
	errAxisBounds = errors.New("nune: axis out of bounds")

	// errBadStep occurs when a step size is null or
	// is opposite to the inverval's order.
	errBadStep = errors.New("nune: received a bad step size")

	// errBadInterval occurs when a null interval, or a descending
	// interval, or an interval that doesn't fall within the allowed limits
	// is provided to a function like range or slice.
	errBadInterval = errors.New("nune: received a bad interval")

	// errUnwrapBacking occurs when a backing could not be
	// unwrapped into a 1-dimensional numeric buffer in order
	// to create a Tensor.
	errUnwrapBacking = errors.New("nune: could not unwrap backing to Tensor")

	// errArgsBounds occurs when a function receives more arguments
	// than it should.
	errArgsBounds = errors.New("nune: received more arguments than allowed")
)

// assertGoodShape makes sure a shape isn't empty,
// and that none of the shapes axes's dimensions
// are less than or equal to zero, and panics otherwise.
func assertGoodShape(s ...int) {
	if len(s) == 0 {
		panic(errBadShape)
	}

	for _, a := range s {
		if a <= 0 {
			panic(errBadShape)
		}
	}
}

// assertAxisBounds makes sure an axis is strictly positive
// and is less than the Tensor's rank.
func assertAxisBounds(axis, rank int) {
	if axis < 0 || axis > rank {
		panic(errAxisBounds)
	}
}

// assertGoodStep makes sure a step size isn't null,
// and whose sign matches the interval's order.
func assertGoodStep(s, start, end int) {
	if s == 0 {
		panic(errBadStep)
	} else if s > 0 && end < start || s < 0 && end > start{
		panic(errBadStep)
	}
}

// assertGoodInterval makes sure the interval is not null,
// is ascending, and falls within the given limits, inclusive.
// A value of nil for min or max means there is no limit.
func assertGoodInterval(start, end int, limits ...[2]int) {
	assertArgsBounds(len(limits), 1)

	if start >= end {
		panic(errBadInterval)
	}
	
	if len(limits) == 1 {
		if start < limits[0][0] || end > limits[0][1] {
			panic(errBadInterval)
		}
	}
}

// assertInRange makes sure a value x is in
// the interval [start, end), and panics otherwise.
func assertInRange(x, start, end int) {
	if x < start || x >= end {
		panic("")
	}
}

// assertArgsBounds makes sure the number of arguments
// doesn't exceed the number of allowed arguments.
func assertArgsBounds(nargs, max int) {
	if nargs > max {
		panic(errArgsBounds)
	}
}