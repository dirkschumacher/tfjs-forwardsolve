"use strict"

const tfc = require("@tensorflow/tfjs-core")

const forwardSolve = async (L, b) => {
  const n = +b.shape[0]
  let buffer = await b.clone().buffer()

  // Just learning
  // Feel free to solve this without buffers
  for (let i = 1; i < n; i++) {
    const Ls = tfc.slice(tfc.gather(L, i), 0, i).reshape([1, i])
    const bs = b.slice(0, i).reshape([i, 1])
    const t = tfc.matMul(Ls, bs).asScalar()
    Ls.dispose()
    bs.dispose()
    const diag = tfc.gather(L, i).as1D().gather(i).asScalar()
    const bt = tfc.tensor(buffer.get(i), [])
    const val = tfc.div(tfc.sub(bt, t), diag)
    t.dispose()
    diag.dispose()
    bt.dispose()
    buffer.set((await val.buffer()).get(0), i)
    val.dispose()
  }
  return buffer.toTensor()
}

module.exports = forwardSolve