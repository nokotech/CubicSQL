import _ from 'underscore';
import Util from './utils/Util.js';
import Dom from './utils/Dom.js';
import {preProcess, deeplearn} from './process/index.js';

(async function main() {

  // let data = await Util.getCSV('/assets/train.csv')
  // data = preProcess(data)
  // Dom.viewTable("#table_result", data)

  const randInt = (min=5 , max=100) => (Math.floor( Math.random() * (max - min + 1) ) + min)
  const ans = (num) => (num == 0) ? [1, 0, 0] : (num == 1) ? [0, 1, 0] : [0, 0, 1]
  let learnX=[], learnY=[], testX=[], testY=[]
  for(let i = 0; i < 300; i++) {
    let num = randInt()
    learnX.push([num, num+1, num+2])
    learnY.push(ans((num+5) % 3));
  }
  for(let i = 0; i < 30; i++) {
    let num = randInt()
    testX.push([num, num+1, num+2])
    testY.push(ans((num+5) % 3));
  }
  console.log(learnY)
  Dom.viewTable("#table_result", testX)
  // console.log(testY)
  deeplearn(learnX, learnY, testX, testY)
  // deeplearn([[1,2,3],[2,3,4],[3,4,5]], [1,2,3], [[4,5,6]], [4])

})();
