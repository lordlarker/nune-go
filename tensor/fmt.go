// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tensor

import (
	"nune"
	"fmt"
	"math"
	"reflect"
	"strings"
)

// String returns a string representation of the Tensor.
func (t Tensor[T]) String() string {
	template := "Tensor({})"
	f := newFmtState(template, t)

	s := strings.Replace(template, "{}", fmtTensor(t, f), 1)

	return fmt.Sprintf("%s", s)
}

// fmtTensor formats the Tensor into a string.
func fmtTensor[T nune.Numeric](t Tensor[T], s fmtState) string {
	var b strings.Builder

	if len(t.shape) == 0 {
		b.WriteString(fmtNum(t.data[0], s))
	} else {
		b.WriteString("[")

		if t.shape[0] > nune.FmtConfig.Excerpt {
			b.WriteString(fmtExcerpted(t, s))
		} else {
			b.WriteString(fmtComplete(t, s))
		}

		b.WriteString("]")
	}

	return b.String()
}

// fmtNum formats a numeric type into a string.
func fmtNum[T nune.Numeric](x T, s fmtState) string {
	switch reflect.ValueOf(x).Kind() {
	case reflect.Float32, reflect.Float64:
		return fmt.Sprintf("%*.*f", s.width, nune.FmtConfig.Precision, float64(x))
	case reflect.Uint8:
		if nune.FmtConfig.Btoa {
			return fmt.Sprintf("%s", string(byte(x)))
		}
		fallthrough
	default:
		return fmt.Sprintf("%*d", s.width, int64(x))
	}
}

// fmtExcerpted formats an excerpted representation of
// a Tensor into a string.
func fmtExcerpted[T nune.Numeric](t Tensor[T], s fmtState) string {
	var b strings.Builder

	var f string

	f = fmtTensor(t.Slice(0, nune.FmtConfig.Excerpt/2), s)
	f = f[1 : len(f)-1]
	b.WriteString(f)

	if len(t.shape) == 1 {
		b.WriteString(", ..., ")
	} else {
		b.WriteString("\n")
		b.WriteString(strings.Repeat(" ", s.pad+1))
		b.WriteString("...,\n")
		b.WriteString(strings.Repeat(" ", s.pad+1))
	}

	f = fmtTensor(t.Slice(t.shape[0]-nune.FmtConfig.Excerpt/2, t.shape[0]), s)
	f = f[1 : len(f)-1]
	b.WriteString(f)

	return b.String()
}

// fmtComplete formats a complete representation of
// a Tensor into a string.
func fmtComplete[T nune.Numeric](t Tensor[T], s fmtState) string {
	var b strings.Builder

	for i := 0; i < t.shape[0]; i++ {
		if len(t.shape) == 1 {
			b.WriteString(fmtTensor(t.Index(i), s))

			if i < t.shape[0]-1 {
				b.WriteString(", ")
			}
		} else {
			b.WriteString(fmtTensor(t.Index(i), s.update()))

			if i < t.shape[0]-1 {
				b.WriteString(strings.Repeat("\n", s.esc))
				b.WriteString(strings.Repeat(" ", s.pad+1))
			}
		}
	}

	return b.String()
}

// A fmtState holds the format configurations while formatting a Tensor.
type fmtState struct {
	depth, esc, pad, width int
}

// update prepares all the fmtState configurations for the next format call.
func (f fmtState) update() fmtState {
	f.depth += 1
	f.esc -= 1
	f.pad += 1

	return f
}

// newFmtState returns a new fmtState configured to
// a base Tensor representation.
func newFmtState[T nune.Numeric](fmt string, t Tensor[T]) fmtState {
	s := fmtState{
		depth: 0,
		esc:   len(t.shape) - 1,
	}

	s.pad = cfgPad(fmt)
	s.width = cfgWidth(t)

	return s
}

// cfgPad configures the padding from a base Tensor representation.
func cfgPad(s string) int {
	return len(strings.Split(s, "{}")[0])
}

// cfgWidth configures the numeric types' width from a given Tensor.
func cfgWidth[T nune.Numeric](t Tensor[T]) int {
	min, max := float64(t.Min()), float64(t.Max()) // find min and max numbers
	x := T(math.Max(math.Abs(min), math.Abs(max))) // find which has more numbers

	switch reflect.ValueOf(x).Kind() {
	case reflect.Float32, reflect.Float64:
		return len(fmt.Sprintf("%.*f", nune.FmtConfig.Precision, float64(x)))
	case reflect.Uint8:
		if nune.FmtConfig.Btoa {
			return 1
		}
		fallthrough
	default:
		return len(fmt.Sprintf("%d", int64(min)))
	}
}
