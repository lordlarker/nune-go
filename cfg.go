// Copyright © Lord Larker. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nune

// FmtConfig holds Nune's formatting configuration.
var FmtConfig = struct {
	Excerpt int // limit of the number of elements formatted
	Precision int  // limit of the number of decimals formatted
	Btoa bool // convert bytes to ASCII
}{
	Excerpt: 6,
	Precision: 4,
	Btoa: false,
}