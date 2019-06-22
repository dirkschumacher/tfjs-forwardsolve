"use strict"

const test = require("tape")
const forwardSolve = require(".")
const tfc = require("@tensorflow/tfjs-core")

const makeArrayEqual = (t) => (a, b) => {
  for (let i = 0; i < a.length; i++) {
    t.equal(a[i], b[i])
  }
}

test("solve example #1", async (t) => {
  const L = tfc.tensor2d([[5, 0, 0], [3, 4, 0], [3, 4, 5]])
  const b = tfc.tensor1d([20, 300, 40])
  const expected = tfc.tensor1d([4, 72, -52], "float32").arraySync()
  const result = (await forwardSolve(L, b)).arraySync()
  const arrayEqual = makeArrayEqual(t)
  arrayEqual(result, expected)
  t.end()
})


test("solve example #2", async (t) => {
  const L = tfc.tensor2d([[5, 0, 0], [3, 4, 0], [3, 4, 5]])
  const b = tfc.tensor1d([20, 300, 40])
  const expected = tfc.tensor1d([4, 72, -52], "float32").arraySync()
  await forwardSolve(L, b) // should not affect the result
  const result = await forwardSolve(L, b)
  const arrayEqual = makeArrayEqual(t)
  arrayEqual(result.arraySync(), expected)
  t.end()
})