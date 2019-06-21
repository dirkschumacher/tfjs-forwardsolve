"use strict"

const tfc = require("@tensorflow/tfjs-core")

const forwardSolve = async (L, b) => {
  const n = +b.shape[0]
  const buffer = await b.buffer()
  const bufferOut = tfc.buffer([n], "float32")

  // Just learning
  // Feel free to solve this without buffers
  bufferOut.set(buffer.get(0), 0)
  for (let i = 1; i < n; i++) {
    const Ls = tfc.slice(tfc.gather(L, i), 0, i).reshape([1, i])
    const bs = bufferOut.toTensor().slice(0, i).reshape([i, 1])
    const t = tfc.matMul(Ls, bs).asScalar()
    const diag = tfc.gather(L, i).as1D().gather(i).asScalar()
    const bt = tfc.tensor(buffer.get(i), [])
    const val = tfc.div(tfc.sub(bt, t), diag)
    bufferOut.set((await val.buffer()).get(0), i)
    Ls.dispose()
    bs.dispose()
    t.dispose()
    diag.dispose()
    bt.dispose()
    val.dispose()
  }
  return bufferOut.toTensor()
}

module.exports = forwardSolve