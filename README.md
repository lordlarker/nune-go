# Nune (v0.1)
**Nu**merical engi**ne** is a library for performing numerical computation in Go, relying on generic tensors.

## Table of contents
- [Installation](#Installation)
- [Design](#Design)
- [Usage](#Usage)
- [Roadmap](#Roadmap)
- [License](#License)

## Installation
**Nune** requires Go v1.18 as it heavily relies on generics in order to achieve a flexible interface.
Go v1.18 is currently only available in beta version, which can be downloaded [here](https://go.dev/dl/).

## Design
**Nune** follows Go's principles and design philosophies of simplicity and minimalism.
Therefore, going forward, **Nune** will always be a compact library providing only the minimal and foundational functions to deal with numerical data and computation.

## Usage
Creating tensors was never easier:
```go
package main

import (
	"github.com/lordlarker/nune/tensor"
)

func  main() {
	// Nune can create tensors of any shapes.
	_ = tensor.Zeros[int](5, 5) // create a 5x5 tensor
	_ = tensor.Ones[int](5, 10, 5, 25) // or a weird one
	_ = tensor.Full[int](4, []int{5, 2}) // (value, shape)

	// From a range?
	_ = tensor.Range[uint](0, 100, 2) // (start, end, step)
	// Maybe in reverse?
	_ = tensor.Range[uint](100, 0, -1)

	// Even a geometrical space?
	_ = tensor.Linspace[float32](-10, 10, 5) // (start, end, size)
	_ = tensor.Logspace[float32](2, 0, 1, 20) // (base, start, end, size)

	// Random facilities as well?
	_ = tensor.Rand[float64](3, 3, 3) // (shape ...int)
	_ = tensor.RandRange[float64](10, 100, []int{4, 4}) // (start, end, shape)

	// In fact Nune can create tensors from (almost) anything.
	_ = tensor.From[int](5)
	_ = tensor.From[float32]([]int{1, 2, 3}) // implicitly casts the type
	_ = tensor.From[float64]("nune") // even strings

	type Point int
	_ = tensor.From[Point](3) // or custom types

	// It can also take any n-dimensional array/slice.
	// The tensor will implicitly have a shape of (2, 2).
	_ = tensor.From[byte]([][]uint{{1, 2}, {3, 4}})
}
```
Or performing **parallel** operations with **multithreaded** math:
```go
package main

import (
	"math"
	"github.com/lordlarker/nune/tensor"
)

func main() {
	t := tensor.Range[float64](-100, 100, 1).Reshape(2, 4, 25)
	
	// Built in methods of various kinds
	_ = t.Abs()
	_ = t.Sum()
	_ = t.Add(t.Copy())
	
	_ = t.Sqrt().Floor()

	_, _, _ = t.Min(), t.Max(), t.Mean()
	_, _, _ = t.Sin(), t.Cos(), t.Tan()
	
	// Or make your own, and automatically
	// get parallelization on the way
	_ = t.PwiseOp(func(x float64) float64 {
		return 1 / (1 + math.Exp(-x)) // parallel sigmoid function
	})
}
```
And  what's better than some fancy terminal output:
```go
package main

import (
	"fmt"
	"github.com/lordlarker/nune"
	"github.com/lordlarker/nune/tensor"
)

func main() {
	scalar := tensor.From[int](5)
	fmt.Println(scalar)
	// Prints:
	// 
	// Tensor(5)

	nune.FmtConfig.Precision = 2 // number of decimal points
	t := tensor.Range[float32](-5, 5, 1).Reshape(2, 5)
	fmt.Println(t)
	// Prints:
	//
	// Tensor([[-5.00, -4.00, -3.00, -2.00, -1.00]
	//         [ 0.00,  1.00,  2.00,  3.00,  4.00]])

	nune.FmtConfig.Excerpt = 4 // number of elements shown per axis
	t = tensor.Range[float32](0, 256, 1).Reshape(16, 16)
	fmt.Println(t)
	// Prints:
	//
	//  Tensor([[  0.00,   1.00, ...,  14.00,  15.00]
        //          [ 16.00,  17.00, ...,  30.00,  31.00]
        //          ...,
        //          [224.00, 225.00, ..., 238.00, 239.00]
        //          [240.00, 241.00, ..., 254.00, 255.00]])
	
	nune.FmtConfig.Btoa = true // bytes to ASCII
	b := tensor.From[byte]("nune")
	fmt.Println(b)
	// Prints:
	//
	// Tensor([n, u, n, e])
}
```

## Roadmap
Since **Nune** will always be designed to provide only the minimal foundational numerical computing facilities, the roadmap isn't that long, and **Nune** already is close to stabilizing.
Things that still need work before this is a rock-stable library are the following, in order:
 - Create a backend to handle all the tensor manipulation routines and managing layout and memory.
 - Optimize the API for maximum performance.
 - Rigorously test the API.
 - Stabilize the API.
 - Write examples to ease the use of this library.
 
## License
**Nune** has a BSD-style license, as found in the [LICENSE](https://github.com/lordlarker/nune/blob/main/LICENSE) file.
