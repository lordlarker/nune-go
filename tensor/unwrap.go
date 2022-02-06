// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"reflect"

	"github.com/lordlarker/nune"
	"github.com/lordlarker/nune/internal/slice"
)

// unwrapAnySlice attempts to recursively unwrap a slice
// or multiple nested slices of 'any' underlying type
// into a 1-dimensional contiguous numeric slice.
//
// TODO: optimize the hell out of this
func unwrapAny[T nune.Numeric](s []any, shape []int) ([]T, []int) {
	if len(s) == 0 {
		panic(errUnwrapBacking)
	}

	if anyIsNumeric(s[0]) {
		return anyToNumeric[T](s...), shape
	}

	if k := reflect.ValueOf(s[0]).Kind(); k == reflect.Array || k == reflect.Slice {
		d := reflect.ValueOf(s[0]).Len()

		for i := 1; i < len(s); i++ {
			if reflect.ValueOf(s[i]).Len() != d {
				panic(errUnwrapBacking)
			}
		}

		p := slice.WithLen[any](len(s) * d)
		for i := 0; i < len(s); i++ {
			r := reflect.ValueOf(s[i])
			for j := 0; j < d; j++ {
				p[i*d+j] = r.Index(j).Interface()
			}
		}

		shape = append(shape, len(s)*d)

		return unwrapAny[T](p, shape)
	}

	panic(errUnwrapBacking)
}

// anyIsNumeric returns whether or not an interface
// is a numeric type.
func anyIsNumeric(a any) bool {
	switch a.(type) {
	case int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64:
		return true
	default:
		return false
	}
}

// anyToNumeric casts an interface{} to the given numeric type.
func anyToNumeric[T nune.Numeric](s ...any) []T {
	switch s[0].(type) {
	case int:
		return numericToNumeric[T, int](s)
	case int8:
		return numericToNumeric[T, int8](s)
	case int16:
		return numericToNumeric[T, int16](s)
	case int32:
		return numericToNumeric[T, int32](s)
	case int64:
		return numericToNumeric[T, int64](s)
	case uint:
		return numericToNumeric[T, uint](s)
	case uint8:
		return numericToNumeric[T, uint8](s)
	case uint16:
		return numericToNumeric[T, uint16](s)
	case uint32:
		return numericToNumeric[T, uint32](s)
	case uint64:
		return numericToNumeric[T, uint64](s)
	case float32:
		return numericToNumeric[T, float32](s)
	case float64:
		return numericToNumeric[T, float64](s)
	default:
		return nil
	}
}

// numericToNumeric casts a numeric type to another numeric type.
func numericToNumeric[T, U nune.Numeric](s []any) []T {
	ns := slice.WithLen[T](len(s))
	for i := 0; i < len(s); i++ {
		ns[i] = T(s[i].(U))
	}

	return ns
}

// anyToTensor attempts to cast an interface{}
// to a Tensor of the given numeric type.
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
