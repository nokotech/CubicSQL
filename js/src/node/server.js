// Import lib
const express = require('express');
const bodyParser = require('body-parser');

// express setting
const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.listen(3000);
console.log(__dirname);
console.log('server running on http://localhost:3000/');

// SITE routing
app.use(express.static(__dirname + '/../../dest/'));