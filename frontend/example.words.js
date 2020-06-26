function get_json(jsonpath){
  var arr = null;
  $.ajax({
      'async': false,
      'global': false,
      'url': jsonpath,
      'dataType': "json",
      'success': function (data) {
          arr = data;
      }
  })
  return arr;
};
// convert json data to array
function json_to_array(jsonobj){
  var columns = [];
  var data_m = [];
  $.each(jsonobj[0], function (index, value) {
      columns.push(index);
  });
  for (var i=0; i<columns.length; i++){
    data_m[i] = jsonobj.map(d => d[columns[i]]);
  }
  return(data_m)
};
// Reads csv file and returns 2 array first is columns and the second one is the data converted to an array
function read_csv(csvpath){
  var request = new XMLHttpRequest();
  request.open("GET", csvpath, false);
  request.send(null);

  var csvData = new Array();
  var jsonObject = request.responseText.split(/\r?\n|\r/);
  for (var i = 0; i < jsonObject.length; i++) {
    csvData.push(jsonObject[i].split(','));
  }
  var columns = csvData[0]
  var data_matrix = [];
  csvData = csvData.slice(1,csvData.length);
  for (var i=0; i<columns.length; i++){
    data_matrix[i] = csvData.map(d => d[i]);
    data_matrix[i] = data_matrix[i].slice(0,data_matrix[i].length-1);
  }

  var data = [columns,data_matrix]

  return(data)
}
// Defining words
mywords = get_json("../server/files/words.json");

// select number of words to show
function words_to_show(word_count){
  return(mywords.slice(0, word_count));
}
// Getting word predictions from file
var predicted_words = read_csv("../server/files/word_predictions.csv")
predicted_words_columns = predicted_words[0];
var predicted_words = predicted_words[1];


sentence = 'this is a sentence'
console.log(sentence.includes('this'));

// get tweet predictions from json
var predicted_tweets = get_json("../server/files/predicted_tweets.json");
predicted_tweets = json_to_array(predicted_tweets);

// return color of words based on their label
function colorvar(word){
      var color ;
      for (var i = 0; i < predicted_words[0].length; i++) {
        if (word == predicted_words[0][i]){
          if (predicted_words[4][i]==0)
            color = 'red';
          else
            color = 'blue';
        }

    }
    return (color);
  };
  // console.log(colorvar('people'));
  function clicked(){
    d3.select("svg").remove();
    words = words_to_show($("#words_num").val());
    create_vis(words);
  }

// return text to tooltip
function tooltip_text(word){
  var occour = 0;
  var precentage = 0;
  for (var i=0; i<predicted_words[0].length;i++){
    if (word == predicted_words[0][i]){
      occour = parseInt(predicted_words[1][i]) + parseInt(predicted_words[2][i]);
      precentage = parseInt(predicted_words[3][i]);
    }
  }
  data = [occour,100-precentage]
  return(data)
}
// Retutn tweets for alert
function alert_text(word){
  word =  word;
  function fan_nonfan(value){
    if (value == 1){
      return ('red');
    }
    else{
      return ('blue');
    }
  };
  var counter = 1;
  var tweets = '<div style="width: 500px; height: 200px; overflow-y: scroll;">'
    for(var i=0; i < predicted_tweets[0].length; i++){
      if(predicted_tweets[0][i].includes(word)){
        // tweets += 'Tweet: \n'+ '\t•' + predicted_tweets[0][i] + '\n\t• Label: '+ fan_nonfan(predicted_tweets[1][i]) +'\n\n';
        tweets += '<p align="left" style="color:'+fan_nonfan(predicted_tweets[1][i])+';">'+'<span style="color:black;">'+ counter +'- </span>'+predicted_tweets[0][i]+'</p>'
        counter +=1;
      }

  }
  tweets += '</div>'
  return tweets
}


console.log(console.log(predicted_words,predicted_words_columns))

// SPARE CODES ARE HERE

// function alert_text(word){
//   var tweets='';
//   function fan_nonfan(value){
//     if (value == 1){
//       return ('fanboy');
//     }
//     else{
//       return ('related');
//     }
//   };
//     for(var i=0; i < predicted_tweets[0].length; i++){
//       if(predicted_tweets[0][i].toLowerCase().includes(word)){
//         tweets += 'Tweet: \n'+ '\t•' + predicted_tweets[0][i] + '\n\t• Label: '+ fan_nonfan(predicted_tweets[1][i]) +'\n\n';
//       }
//
//   }
//   return tweets
// }


// Retrived data from csv file content
// console.log(JSONItems);
// console.log(rows);
// var reader = new FileReader('word_predictions.csv');

//parse the data file
//     var csvfile = "word_predictions.csv";
//     Papa.parse(csvfile, {
// 	complete: function(results) {
// 		console.log("Finished:", results.data);
// 	}
// });
// let arr = ["one", "two", "three", "four", "five"];
// var arr2 = [];
// function sample(){
//   if(arr.includes("two")){
//     console.log('amir')
//   };
// };
//

// var csvarray = [];
// var client = new XMLHttpRequest();
// client.open('GET', 'word_predictions.csv');
// client.onreadystatechange = function() {
//     var rows = client.responseText.split('\n');
//     for(var i = 0; i < rows.length; i++){
//         csvarray.push(rows[i].split(','));
//     }
// }
// console.log(csvarray);
// var Papa = require('papaparse');
//
// Papa.parse(content, {
//     header: false,
//     delimiter: "\t",
//     complete: function(results) {
//     rows = results.data;
//     }
// });
// sample();
// var words_arr=[];
// // Load the dataset
// data = d3.csv("english_tweets.csv", function(error, data) {
//   if (error) throw error;
//   words1(data);
// });
// function words1(d){
//   var variables = d.columns;
//   tweet = d.map(function(d){return d[variables[5]]});
//   var x = tweet.toString()
// //     vnew = x.trim().split(" ");
//
// //    console.log(vnew);
//   // console.log(results)
//
//   var raw;
//   raw = x.split(/\W+/);
//
//   // var keys = [];
//   // count the word frequency
//   var counts = raw.reduce(function(obj, word) {
//       if(!obj[word]) {
//           obj[word] = 0;
//           // keys.push(word);
//
//       }
//       obj[word]++;
//       return obj;
//   }, {});
//   // var result1 = Object.keys(counts).map(function(key) {
//   //     return [keys[key], counts[key]];
//   //   });
//   // console.log(counts);
//
//   // sort the keys from highest to lowest
//   // keys.sort(function(a,b) {
//   //     return counts[b] - counts[a];
//   // })
//   var sortable = [];
//   for (var i in counts) {
//       sortable.push([i, counts[i]]);
//   }
//   sortable.sort(function(a, b) {
//       return b[1] - a[1];
//   });
// //     texts = sortable.map(function(d){return d[0]});
// //     var returnObj = new Object();
// //     returnObj.text = texts;
// //    console.log(returnObj);
// function initArray(sortable){
//   myArray = [];
//   for(var i=0; i<sortable.length ; i++) {
//         var newElement = {}
//     newElement.text = sortable[i][0];
//     newElement.size = sortable[i][1];
//   //   var obj = {};
//     myArray[i] = newElement
//     }
//     return myArray;
// }
// // console.log(initArray(sortable));
// return (initArray(sortable));
// }
// console.log(words_arr)
