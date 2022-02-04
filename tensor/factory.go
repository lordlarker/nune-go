// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"math"
	"math/rand"
	"nune"
	"nune/internal/slice"
	"reflect"
	"time"
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
func From[T nune.Numeric](b any) Tensor[T] {
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

		d, s := unwrapAnySlice[T](c, []int{len(c)})

		return Tensor[T]{
			data:    d,
			shape:   s,
			strides: stridesFromShape(s),
		}
	default:
		if c, ok := anyToNumeric[T](b); ok {
			return Tensor[T]{
				data:    []T{c},
				shape:   nil,
				strides: nil,
			}
		} else if c, ok := anyToTensor[T](b); ok {
			return c
		} else {
			panic("nune/tensor: From failed to create a Tensor from the given backing")
		}
	}
}

// Full returns a Tensor filled with the given value and
// satisfying the given shape.
func Full[T nune.Numeric](x T, shape []int) Tensor[T] {
	if checkEmptyShape(shape) {
		panic("nune/tensor: Full received an empty shape")
	} else if !checkPosAxes(shape) {
		panic("nune/tensor: Full received a shape with non-strictly positive axes")
	}

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(x)
	}

	return Tensor[T]{
		data:    data,
		shape:   slice.Copy(shape),
		strides: stridesFromShape(shape),
	}
}

// Zeros returns a Tensor filled with zeros and satisfying the given shape.
func Zeros[T nune.Numeric](shape ...int) Tensor[T] {
	return Full[T](0, shape)
}

// Ones returns a Tensor filled with ones and satisfying the given shape.
func Ones[T nune.Numeric](shape ...int) Tensor[T] {
	return Full[T](1, shape)
}

// Range returns a rank-1 Tensor on the interval [start, end),
// and with the given step-size.
func Range[T nune.Numeric](start, end, step int) Tensor[T] {
	if step == 0 {
		panic("nune/tensor: Range received a null step size")
	} else if step > 0 && end < start {
		panic("nune/tensor: Range received a positive step size in a descending interval")
	} else if step < 0 && end > start {
		panic("nune/tensor: Range received a negative step size in an ascending interval")
	}

	// interval size must be strictly positive
	if start == end {
		end++
	}

	d := math.Sqrt(math.Pow(float64(end-start), 2))   // distance
	l := int(math.Floor(d / math.Abs(float64(step)))) // length

	i := 0
	rng := slice.WithLen[T](l)
	for x := 0; x < l; x += 1 {
		rng[i] = T(start + x*step)
		i++
	}

	return Tensor[T]{
		data:    rng,
		shape:   []int{len(rng)},
		strides: []int{1},
	}
}

// Rand returns a Tensor filled with random numbers generated
// from a uniform distribution on the interval [0, 1).
func Rand[T nune.Numeric](shape ...int) Tensor[T] {
	if checkEmptyShape(shape) {
		panic("nune/tensor: Rand received an empty shape")
	} else if !checkPosAxes(shape) {
		panic("nune/tensor: Rand received a shape with non-strictly positive axes")
	}

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.Float64())
	}

	return Tensor[T]{
		data:    data,
		shape:   slice.Copy(shape),
		strides: stridesFromShape(shape),
	}
}

// Randn returns a Tensor filled with random numbers generated
// from a uniform distribution with mean 0 and variance 1
// (also known as the standard deviation).
func Randn[T nune.Numeric](shape ...int) Tensor[T] {
	if checkEmptyShape(shape) {
		panic("nune/tensor: Randn received an empty shape")
	} else if !checkPosAxes(shape) {
		panic("nune/tensor: Randn received a shape with non-strictly positive axes")
	}

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.NormFloat64())

	}

	return Tensor[T]{
		data:    data,
		shape:   slice.Copy(shape),
		strides: stridesFromShape(shape),
	}
}

// RandRange returns a tensor filled with random numbers
// generated uniformly on the interval [start, end).
func RandRange[T nune.Numeric](start, end int, shape []int) Tensor[T] {
	if start > end {
		panic("nune/tensor: RandRange received a descending interval")
	} else if checkEmptyShape(shape) {
		panic("nune/tensor: RandRange received an empty shape")
	} else if !checkPosAxes(shape) {
		panic("nune/tensor: RandRange received a shape with non-strictly positive axes")
	}

	// rand.Intn requires strictly positive arguments
	if start == end || end == 0 {
		end++
	}

	data := slice.WithLen[T](slice.Prod(shape))
	for i := 0; i < len(data); i++ {
		data[i] = T(rand.Intn(end-start) + start)
	}

	return Tensor[T]{
		data:    data,
		shape:   slice.Copy(shape),
		strides: stridesFromShape(shape),
	}
}

// Linspace returns a rank-1 Tensor of the given size whose values
// are evenly spaced on the interval [start, end].
func Linspace[T nune.Numeric](start, end, size int) Tensor[T] {
	if size <= 0 {
		panic("nune/tensor: Linspace received a non-strictly positive size")
	}

	// interval size must be strictly positive
	if start == end {
		return Full(T(start), []int{size})
	}

	d := math.Sqrt(math.Pow(float64(end-start), 2)) // distance

	// make sure sizes greater than the interval's size are only
	// allowed when T is a floating point type.
	if T(d)/T(size) == 0 {
		panic("nune: Linspace received a size greater than the interval's size with a non-floating point type")
	}

	var x, step T

	x = T(start)
	if size == 1 {
		step = T(end-start) / 1
	} else {
		step = T(end-start) / T(size-1)
	}

	data := slice.WithLen[T](size)
	for i := 0; i < size; i++ {
		data[i] = x
		x += step
	}

	return Tensor[T]{
		data:    data,
		shape:   []int{size},
		strides: []int{1},
	}
}

// Logspace returns a rank-1 Tensor of the given size whose values
// are evenly spaced on the interval
// [math.Pow(base, start), math.Pow(base, end)].
func Logspace[T nune.Numeric](base, start, end float64, size int) Tensor[T] {
	if size <= 0 {
		panic("nune/tensor: Logspace received a non-strictly positive size")
	}

	d := math.Sqrt(math.Pow(float64(end-start), 2)) // distance

	// make sure sizes greater than the interval's size are only
	// allowed when T is a floating point type.
	if T(d)/T(size) == 0 {
		panic("nune: Logspace received a size greater than the interval's size with a non-floating point type")
	}

	var x, step float64

	x = math.Pow(base, start)
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

	return Tensor[T]{
		data:    data,
		shape:   []int{size},
		strides: []int{1},
	}
}
