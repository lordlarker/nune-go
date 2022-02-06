// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"math"
	"math/rand"
	"reflect"
	"time"

	"github.com/lordlarker/nune"
	"github.com/lordlarker/nune/internal/slice"
)

// init sets the rand package's rand.Source value to the local time.
func init() {
	rand.Seed(time.Now().UnixNano())
}

// Seed sets the rand package's rand.Source value to the given seed.
func Seed(seed int64) {
	rand.Seed(seed)
}

// From returns a Tensor from the given backing - be it a numeric type,
// a sequence, or nested sequences - with the corresponding shape.
//
// TODO: optimize the hell out of this function and
// its helper functions.
func From[T nune.Numeric](b any) *Tensor[T] {
	switch k := reflect.TypeOf(b).Kind(); k {
	case reflect.String:
		b = any([]byte(b.(string)))
		fallthrough
	case reflect.Array, reflect.Slice:
		v := reflect.ValueOf(b)

		c := make([]any, v.Len())
		for i := 0; i < v.Len(); i++ {
			c[i] = v.Index(i).Interface()
		}

		d, s := unwrapAny[T](c, []int{len(c)})

		return &Tensor[T]{
			storage: newStorage(d),
			layout:  newLayout(s),
		}
	default:
		if anyIsNumeric(b) {
			return &Tensor[T]{
				storage: newStorage(anyToNumeric[T](b)),
				layout:  newLayout(nil),
			}
		} else if c, ok := anyToTensor[T](b); ok {
			return c
		} else {
			panic(errUnwrapBacking)
		}
	}
}

// Full returns a Tensor filled with the given value and
// satisfying the given shape.
func Full[T nune.Numeric](x T, shape []int) *Tensor[T] {
	assertGoodShape(shape...)

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(x)
	}

	return &Tensor[T]{
		storage: newStorage(data),
		layout:  newLayout(slice.Copy(shape)),
	}
}

// Zeros returns a Tensor filled with zeros and satisfying the given shape.
func Zeros[T nune.Numeric](shape ...int) *Tensor[T] {
	return Full(T(0), shape)
}

// Ones returns a Tensor filled with ones and satisfying the given shape.
func Ones[T nune.Numeric](shape ...int) *Tensor[T] {
	return Full(T(1), shape)
}

// Range returns a rank 1 Tensor on the interval [start, end),
// and with the given step-size.
func Range[T nune.Numeric](start, end, step int) *Tensor[T] {
	assertGoodStep(step, start, end)

	d := math.Sqrt(math.Pow(float64(end-start), 2))   // distance
	l := int(math.Floor(d / math.Abs(float64(step)))) // length

	i := 0
	rng := slice.WithLen[T](l)
	for x := 0; x < l; x += 1 {
		rng[i] = T(start + x*step)
		i++
	}

	return &Tensor[T]{
		storage: newStorage(rng),
		layout:  newLayout([]int{len(rng)}),
	}
}

// Rand returns a Tensor filled with random numbers generated
// from a uniform distribution on the interval [0, 1).
func Rand[T nune.Numeric](shape ...int) *Tensor[T] {
	assertGoodShape(shape...)

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.Float64())
	}

	return &Tensor[T]{
		storage: newStorage(data),
		layout:  newLayout(slice.Copy(shape)),
	}
}

// Randn returns a Tensor filled with random numbers generated
// from a uniform distribution with mean 0 and variance 1
// (also known as the standard deviation).
func Randn[T nune.Numeric](shape ...int) *Tensor[T] {
	assertGoodShape(shape...)

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.NormFloat64())

	}

	return &Tensor[T]{
		storage: newStorage(data),
		layout:  newLayout(slice.Copy(shape)),
	}
}

// RandRange returns a tensor filled with random numbers
// generated uniformly on the ascending interval [start, end).
func RandRange[T nune.Numeric](start, end int, shape []int) *Tensor[T] {
	assertGoodStep(1, start, end) // make sure the interval is ascending
	assertGoodInterval(start, end)
	assertGoodShape(shape...)

	// rand.Intn requires strictly positive arguments
	if start == end || end == 0 {
		end++
	}

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.Intn(end-start) + start)
	}

	return &Tensor[T]{
		storage: newStorage(data),
		layout:  newLayout(slice.Copy(shape)),
	}
}

// Linspace returns a rank-1 Tensor of the given size whose values
// are evenly spaced on the interval [start, end].
func Linspace[T nune.Numeric](start, end, size int) *Tensor[T] {
	assertGoodShape(size)

	// if interval size is null
	if start == end {
		return Full(T(start), []int{size})
	}

	var x, step float64

	x = float64(start)
	// avoid division by zero
	if size == 1 {
		step = float64(end-start) / 1
	} else {
		step = float64(end-start) / float64(size-1)
	}

	data := slice.WithLen[T](size)
	for i := 0; i < size; i++ {
		data[i] = T(x)
		x += step
	}

	return &Tensor[T]{
		storage: newStorage(data),
		layout:  newLayout([]int{size}),
	}
}

// Logspace returns a rank-1 Tensor of the given size whose values
// are evenly spaced on the interval
// [math.Pow(base, start), math.Pow(base, end)].
func Logspace[T nune.Numeric](base, start, end float64, size int) *Tensor[T] {
	assertGoodShape(size)

	var x, step float64

	x = math.Pow(base, start)
	// avoid division by zero
	if size == 1 {
		step = math.Pow(base, (end-start)/1)
	} else {
		step = math.Pow(base, (end-start)/float64(size-1))
	}

	data := slice.WithLen[T](size)
	for i := 0; i < size; i++ {
		data[i] = T(x)
		x *= step
	}

	return &Tensor[T]{
		storage: newStorage(data),
		layout:  newLayout([]int{size}),
	}
}
