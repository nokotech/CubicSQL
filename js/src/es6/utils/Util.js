import request from 'superagent';
import _ from 'underscore';

class Util {

    getCSV(url) {
        return new Promise((resolve) => {
            request
            .get(url)
            .end((err, res) => {
                let result = [];
                _.each((res.text || []).split("\n"), (v, i) => result[i] = this.parseCsv(v));
                resolve(result);
            });
        });
    }

    parseCsv(text) {
        let formartText = ""
        text = text.replace( /(""|""")/g , "^" ).split('\"')
        text.length == 3 ? text[1] = text[1].replace( "," , "_" ) : null
        _.each(text, r => formartText += r)
        return formartText.split(',')
    }

}
export default new Util();