import $ from 'jquery';
import _ from 'underscore';

class Dom {

    viewText(dom, text) {
        $(dom).text(text);
    }

    viewTable(dom, table) {
        _.each( table, col => {
            let line = ""
            _.each( col, v => {
                let className = (v=="") ? "class='red'" : ""
                line += `<td ${className}>${v}</td>`
            })
            $(dom).append(`<tr>${line}</tr>`)
        });
    }

}
export default new Dom();