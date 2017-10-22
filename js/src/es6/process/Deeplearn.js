import * as deeplearn from 'deeplearn'
import Dom from '../utils/Dom.js'

export default class Deeplearn {

    constructor(trainX=[], trainY=[], testX=[], testY=[]) {
        this.output = null
        const SHAPE = trainX[0].length
        const LABEL = trainY[0].length
        const LEARNING_RATE = 1
        const NUM_BATCHES = 10
        const math = new deeplearn.NDArrayMathGPU()
        const graph = new deeplearn.Graph()
        const session = new deeplearn.Session(graph, math)

        // 入力層
        const inputTensor = graph.placeholder("input", [SHAPE])
        console.log(`inputTensor = ${inputTensor.shape}`);       // [3]
        const labelTensor = graph.placeholder('label', [LABEL])
        console.log(`labelTensor = ${labelTensor.shape}`);       // [1]

        // 入力層 - 隠れ層
        const multiplier = graph.variable('multiplier', deeplearn.Array2D.randNormal([LABEL, SHAPE]));
        console.log(`multiplier = ${multiplier.shape}`);         // [1,3]

        // 隠れ層 - 出力層
        const outputTensor = graph.matmul(multiplier, inputTensor);
        console.log(`outputTensor = ${outputTensor.shape}`);     // [1]

        // // 入力層 - 隠れ層
        // const w0 = graph.variable("w0", deeplearn.Array2D.randNormal([LABEL, SHAPE]))
        // const b0 = graph.variable("b0", deeplearn.Array2D.randNormal([LABEL]))
        // const mat0 = graph.matmul(inputTensor, w0)               //行列乗算
        // const add0 = graph.add(mat0, b0)                         //行列加算
        // const hiddenTensor = graph.relu(add0)
        // console.log(`hiddenTensor = ${hiddenTensor.shape}`);     // [1]

        // // 隠れ層 - 出力層
        // const w1 = graph.variable("w1", deeplearn.Array2D.randNormal([SHAPE, 1]))
        // const b1 = graph.variable("b1", deeplearn.Array2D.randNormal([LABEL]))
        // const matmul = graph.matmul(hiddenTensor, w1)
        // const add1 = graph.add(matmul, b1)
        // const outputTensor = graph.sigmoid(graph.reshape(add1, [LABEL]))
        // console.log(`outputTensor = ${outputTensor.shape}`);    // [1]

        // // 損失関数（二乗誤差）
        const cost = graph.meanSquaredCost(outputTensor, labelTensor)
        console.log(`cost = ${cost.shape}`); // []
        
        math.scope((keep, track) => {
            const xs = trainX.map(x => track(deeplearn.Array1D.new(x)))
            const ys = trainY.map(x => track(deeplearn.Array1D.new(x)))
            const shuffledInputProviderBuilder = new deeplearn.InCPUMemoryShuffledInputProviderBuilder([xs, ys])
            const [xProvider, yProvider] = shuffledInputProviderBuilder.getInputProviders()
            const optimizer = new deeplearn.SGDOptimizer(LEARNING_RATE)
            // 計算
            for (let i = 0; i < NUM_BATCHES; i++) {
                const data = [{tensor: inputTensor, data: xProvider}, {tensor: labelTensor, data: yProvider}]
                const costValue = session.train(cost, data, xs.length, optimizer, deeplearn.CostReduction.MEAN)
                console.log(`${i}: Average cost: ${costValue.get()}`)
            }
            // 推定
            // for(let i = 0; i < testX.length; i++) {
                const d = [4,5,6]
                const a = [1,0,0]
                const testT = [{tensor: inputTensor, data: track(deeplearn.Array1D.new(d))}]
                const result = session.eval(outputTensor, testT);
                console.log(result.getValues()[0]);
                // const testT = [{tensor: inputTensor, data: track(deeplearn.Array1D.new(testX[i]))}]
                // const result = session.eval(outputTensor, testT);
                // let r = result.getValues()[0] > 0.5 ? 1 : 0;
                // console.log(`${i}: ${r === testY[i]}`, result.getValues()[0]);
            // }
        })

    }

}