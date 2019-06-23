"use strict"

const tfc = require("@tensorflow/tfjs-core")

const forwardSolve = (L, b) => {
  return tfc.tidy(() => {
    const n = +b.shape[0]
    let tensorOut = tfc.tensor1d([])
    // TODO: make it a bit more readable and use a column vector throughout
    tensorOut = tensorOut.concat(tfc.div(b.gather([0]), L.flatten().slice(0, 1)).as1D())
    for (let i = 1; i < n; i++) {
      const Ls = tfc.slice(tfc.gather(L, i), 0, i).reshape([1, i])
      const t = tfc.matMul(Ls, tensorOut.reshape([i, 1])).asScalar()
      const diag = tfc.gather(L, i).as1D().gather(i).asScalar()
      const bt = b.gather([i]).asScalar()
      const val = tfc.div(tfc.sub(bt, t), diag).as1D()
      tensorOut = tensorOut.concat(val)
    }
    return tensorOut.reshape([n, 1])
  })
}

module.exports = forwardSolve