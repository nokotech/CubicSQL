/** MongoDB **/
const log = require("../util/Log");

class Mongo{

  // get collection
  constructor (db, name) {
    this.collection = db.collection(name);
  }

  // insert in document
  insert(obj) {
    this.collection.insertOne(obj, this.end);
  }

  // search in document
  search(param) {
    this.collection.find(param).toArray((error, docs) => {
      //log(JSON.stringify(docs), 2);
      return docs;
    });
  }

  /** end **/
  end(error, result) {
    // log(error);
    log(JSON.stringify(result.result));
  }
}

module.exports = Mongo;