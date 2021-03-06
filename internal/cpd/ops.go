// Copyright © Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpd

import (
	"math"
	"runtime"
	"sync"

	"github.com/lordlarker/nune"
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
	nChunks := int(math.Ceil(float64(len(buf))/float64(nCPU)))

	var wg sync.WaitGroup

	res := make([]T, 0, nChunks)
	ch := make(chan T, nChunks)

	for i := 0; i < nChunks; i++ {
    	min := (i * len(buf) / nChunks)
		max := ((i + 1) * len(buf)) / nChunks

		wg.Add(1)
		go func(s []T, c chan T) {
			c <- f(s)

			wg.Done()
		}(buf[min:max], ch)
	}

	wg.Wait()
	close(ch)

	for v := range ch {
		res = append(res, v)
	}

	return f(res)
}
