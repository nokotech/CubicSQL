// import
const MongoClient = require("mongodb").MongoClient;
const log = require("./util/Log");
const MN = require("./operator/Mongo");
const setting = require("./config");

// MongoDB
MongoClient.connect(setting.url, (error, db) => {
  log("Connect MongoDB");
  __main__(db);
  db.close();
});

function __main__(db) {
  let mn = new MN(db, "products");
  mn.insert({
    "name": "Hack MongoDB",
    "price": 1280
  });
  mn.search();
}

