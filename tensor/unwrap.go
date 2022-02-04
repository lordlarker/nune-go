// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"nune"
	"nune/internal/slice"
	"reflect"
)

// unwrapAnySlice attempts to recursively unwrap a slice
// or multiple nested slices of 'any' underlying type
// into a 1-dimensional contiguous numeric slice.
func unwrapAnySlice[T nune.Numeric](n []any, shape []int) ([]T, []int) {
	if len(n) == 0 {
		panic("nune/tensor: unwrapAnySlice received an empty slice")
	}

	if _, ok := anyToNumeric[T](n[0]); ok {
		c := slice.WithLen[T](len(n))
		for i := 0; i < len(n); i++ {
			c[i], _ = anyToNumeric[T](n[i])
		}

		return c, shape
	}

	if k := reflect.ValueOf(n[0]).Kind(); k == reflect.Array || k == reflect.Slice {
		d := reflect.ValueOf(n[0]).Len()

		for i := 1; i < len(n); i++ {
			if reflect.ValueOf(n[i]).Len() != d {
				panic("nune/tensor: unwrapAnySlice received a backing with unequal dimensions along the same axes")
			}
		}

		p := slice.WithLen[any](len(n) * d)
		for i := 0; i < len(n); i++ {
			r := reflect.ValueOf(n[i])
			for j := 0; j < d; j++ {
				p[i*d+j] = r.Index(j).Interface()
			}
		}

		shape = append(shape, len(p)/len(n))

		return unwrapAnySlice[T](p, shape)
	}

	panic("nune/tensor: unwrapAnySlice failed to unwrap the backing into a 1-dimensional numeric slice")
}

// anyToNumeric attemps to cast an interface{}
// to the given Numeric type.
func anyToNumeric[T nune.Numeric](a any) (T, bool) {
	switch a.(type) {
	case int:
		return T(a.(int)), true
	case int8:
		return T(a.(int8)), true
	case int16:
		return T(a.(int16)), true
	case int32:
		return T(a.(int32)), true
	case int64:
		return T(a.(int64)), true
	case uint:
		return T(a.(uint)), true
	case uint8:
		return T(a.(uint8)), true
	case uint16:
		return T(a.(uint16)), true
	case uint32:
		return T(a.(uint32)), true
	case uint64:
		return T(a.(uint64)), true
	case float32:
		return T(a.(float32)), true
	case float64:
		return T(a.(float64)), true
	default:
		return T(0), false
	}
}

// anyToTensor attempts to cast an interface{}
// to a Tensor of the given Numeric type.
func anyToTensor[T nune.Numeric](a any) (Tensor[T], bool) {
	switch a.(type) {
	case Tensor[int]:
		return Cast[T](a.(Tensor[int])), true
	case Tensor[int8]:
		return Cast[T](a.(Tensor[int8])), true
	case Tensor[int16]:
		return Cast[T](a.(Tensor[int16])), true
	case Tensor[int32]:
		return Cast[T](a.(Tensor[int32])), true
	case Tensor[int64]:
		return Cast[T](a.(Tensor[int64])), true
	case Tensor[uint]:
		return Cast[T](a.(Tensor[uint])), true
	case Tensor[uint8]:
		return Cast[T](a.(Tensor[uint8])), true
	case Tensor[uint16]:
		return Cast[T](a.(Tensor[uint16])), true
	case Tensor[uint32]:
		return Cast[T](a.(Tensor[uint32])), true
	case Tensor[uint64]:
		return Cast[T](a.(Tensor[uint64])), true
	case Tensor[float32]:
		return Cast[T](a.(Tensor[float32])), true
	case Tensor[float64]:
		return Cast[T](a.(Tensor[float64])), true
	default:
		return Tensor[T]{}, false
	}
}
