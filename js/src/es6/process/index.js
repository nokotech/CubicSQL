import PreProcess from './PreProcess.js'
import Deeplearn from './Deeplearn.js'

export const preProcess = (data) => new PreProcess().execute(data)
export const deeplearn = (args1, args2, args3, args4) => new Deeplearn(args1, args2, args3, args4)