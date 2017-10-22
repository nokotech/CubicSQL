import _ from 'underscore';

export default class PreProcess {

    // 実行
    execute(data) {
        data = this.completion(data)
        return data
    }

    // 補完
    completion(data) {
        let totalNum = 0, total = 0
        _.each(data, (d, i) => {
          if( d[5]!="" && 0 <= Number(d[5]) && Number(d[5]) <= 150 ) {
            totalNum++
            total += Number(d[5])
          }
        })
        const aveAge = Math.round(total / totalNum)
        _.each(data, (d, i) => { if(d[5] == "") data[i][5] = aveAge } )
        console.log(`num=${totalNum}, total=${total}, ave=${aveAge}`);
        return data
    }

}
