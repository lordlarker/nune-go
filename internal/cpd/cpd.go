// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpd

import (
	"math"
	"nune"
	"runtime"
	"sync"
)

var nCPU = runtime.NumCPU()

func Pointwise[T nune.Numeric](buf []T, f func(T) T) {
	var wg sync.WaitGroup

	chunk := int(math.Floor(float64(len(buf)) / float64(nCPU)))

	for i := 0; i < len(buf); i += chunk {
		var end int
		if i+chunk > len(buf) {
			end = len(buf)
		} else {
			end = i + chunk
		}

		go func(s []T) {
			for i := 0; i < len(s); i++ {
				buf[i] = f(buf[i])
			}

			wg.Done()
		}(buf[i:end])

		wg.Add(1)
	}
	wg.Wait()
}

func Op[T nune.Numeric](buf1, buf2, res []T, f func(T, T) T) {
	var wg sync.WaitGroup

	chunk := int(math.Floor(float64(len(res)) / float64(nCPU)))

	for i := 0; i < len(res); i += chunk {
		var end int
		if i+chunk > len(res) {
			end = len(res)
		} else {
			end = i + chunk
		}

		go func(s1, s2, s3 []T) {
			for i := 0; i < len(s3); i++ {
				s3[i] = f(s1[i], s2[i])
			}

			wg.Done()
		}(buf1[i:end], buf2[i:end], res[i:end])

		wg.Add(1)
	}
	wg.Wait()
}

func Reduct[T nune.Numeric](buf []T, f func([]T) T) T {
	var wg sync.WaitGroup

	chunk := int(math.Floor(float64(len(buf)) / float64(nCPU)))
	if chunk == 0 { chunk++ }

	res := struct {
		sync.RWMutex
		b []T
	}{
		b: make([]T, int(math.Ceil(float64(len(buf)) / float64(chunk)))),
	}

	var bIdx int
	for i := 0; i < len(buf); i += chunk {
		var end int
		if i+chunk >= len(buf) {
			end = len(buf)
		} else {
			end = i + chunk
		}

		go func(s []T, idx int) {
			y := f(s)

			res.Lock()
			res.b[idx] = y
			res.Unlock()

			wg.Done()
		}(buf[i:end], bIdx)

		wg.Add(1)
		bIdx++
	}
	wg.Wait()

	return f(res.b)
}
