# tfjs-forwardsolve
Forward substitution for rank 2 tensors in tensorflow.js

Currently experimental and for learning purposes. Use at own risk.

```js
const tfc = require("@tensorflow/tfjs-core")
const forwardSolve = require("tfjs-forwardsolve")
const L = tfc.tensor2d([[1, 0, 0], [3, 4, 0], [3, 4, 5]])
const b = tfc.tensor2d([[20], [300], [40]])
const result = forwardSolve(L, b)
// [ [20], [60], [-52] ]
```
