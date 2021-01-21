var fn_array = [];  // File name array for image similarity
var ref_array;  // Reference features array for image similarity
var fn_array_ex = [];  // File name array for image similarity EXAMPLE images
var ref_array_ex;  // Reference features array for image similarity EXAMPLE images
var imgList = [0, 0, 0, 0];
var imgListEmpty = 4;  // Number of available slots in the imgList
var b64o = [0, 0, 0, 0];
var b64e = 0;
var b64temp = 0;
// Create off-screen image elements
var tempImg = new Array();
tempImg[0] = new Image();
tempImg[1] = new Image();
tempImg[2] = new Image();
tempImg[3] = new Image();

// Grab elements, create settings, etc.
var video = document.getElementById('videoElement');
// Elements for taking the snapshot
var webCamCanvas = document.getElementById('webCamCanvas');
var wCCcontext = webCamCanvas.getContext('2d');

function aboutModal() {
  // Display the modal
  var modal = document.getElementById("aboutModal");
  modal.style.display = "block";

  // Get the <span> element that closes the modal
  var span = document.getElementById("closeModal");
  // Close the modal on click
  span.onclick = function() {
    modal.style.display = "none";
  }
  
  // When the user clicks anywhere outside of the modal, close it
  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
}

function populateTable(i, tableData) {
  var cardBody = document.getElementById("resultsDiv"+i).getElementsByClassName('card-body')[0];
  cardBody.innerHTML = "<p class='card-text'>Image Similarity</p>";
  tableData.forEach(function(rowData) {
    var item = document.createElement('div');
    item.classList.add("item");
    var img = document.createElement('img');
    img.src = 'https://cvbp-secondary.z19.web.core.windows.net/html_demo/small-150/' + rowData[0];
    var txt = document.createElement('p');
    txt.innerHTML = rowData[0] + "<br/><i>Dist.: " + rowData[1] + "</i>";
    item.appendChild(img);
    item.appendChild(txt);
    cardBody.appendChild(item);
  });
}


function eucDistance(a, b) {
  return a
  .map((x, i) => Math.abs( x - b[i] ) ** 2) // square the difference
  .reduce((sum, now) => sum + now) // sum
  ** (1/2)
}

function calcSimilar(top, queryFeatures, simType) {
  var dist_array = [];
  var rows = 0;
  if (simType == "example") {
    rows = ref_array_ex.length;
  } else {
    rows = ref_array.length;
  }
  var retImg = "-1";
  if (!queryFeatures) {
    var queryRow = Math.floor(Math.random() * (rows - 0 + 1) + 0);
    var queryimg = ref_array[queryRow];
    retImg = 'https://cvbp-secondary.z19.web.core.windows.net/html_demo/small-150/' + fn_array[queryRow];
  } else {
    var queryimg = queryFeatures;
  }
    
  for (i = 0; i < rows; i++) {
    if (simType == "example") {
      let euc = eucDistance(queryimg, ref_array_ex[i]).toFixed(2);
      var arr = [fn_array_ex[i],euc];
    } else {
      let euc = eucDistance(queryimg, ref_array[i]).toFixed(2);
      var arr = [fn_array[i],euc];
    }
    dist_array.push(arr); 
  }
  var topValues = dist_array.sort((a,b) => a[1]-b[1]).slice(0,top);
  return [topValues, retImg];
}

// Process zip file of filenames and parse into array
async function parseSimFileNames(fileType) {
  return new Promise(async function(res,rej) {
    new JSZip.external.Promise(function (resolve, reject) {
      zipFile_fn = 'data/ref_filenames.zip';
      if (fileType == "example") 
        zipFile_fn = 'https://cvbp-secondary.z19.web.core.windows.net/html_demo/data/ref_filenames.zip';
      JSZipUtils.getBinaryContent(zipFile_fn, function(err, data) {
          if (err) {
            reject(err);
          } else {
            resolve(data);
          }
      });
    }).then(function (data) {
      return JSZip.loadAsync(data);
    }).then(function (zip) {
      if (zip.file("../visualize/data/ref_filenames.txt")) {
        return zip.file("../visualize/data/ref_filenames.txt").async("string");
      } else {
        return zip.file("ref_filenames.txt").async("string");
      }
    }).then(function (text) {
      if (fileType == "example") 
        fn_array_ex = JSON.parse(text);
      else 
        fn_array = JSON.parse(text);
      res();
    });
  })
  
}

// Process zip file of reference image features and parse into array
async function parseSimFileFeatures(fileType) {
  return new Promise(async function(res,rej) {
    new JSZip.external.Promise(function (resolve, reject) {
      zipFile_ref = 'data/ref_features.zip';
      if (fileType == "example") 
        zipFile_ref = 'https://cvbp-secondary.z19.web.core.windows.net/html_demo/data/ref_features.zip';
      JSZipUtils.getBinaryContent(zipFile_ref, function(err, data) {
          if (err) {
              reject(err);
          } else {
              resolve(data);
          }
      });
    }).then(function (data) {
      return JSZip.loadAsync(data);
    }).then(function (zip) {
      if (zip.file("../visualize/data/ref_features.txt")) {
        return zip.file("../visualize/data/ref_features.txt").async("string");
      } else {
        return zip.file("ref_features.txt").async("string");
      }
    }).then(function (text) {
      if (fileType == "example") 
        ref_array_ex = JSON.parse(text);
      else 
        ref_array = JSON.parse(text);
      res();
    });
  })
}

// Handle sample image clicks - need this unsual syntax to accomodate the async nature of the "photoSave" process
document.querySelectorAll('.sImg').forEach(item => {
  item.addEventListener('click', () => handleSamples(item), false)
});
document.querySelectorAll('.sImg').forEach(item => {
  item.addEventListener('click', () => custom_close(), false)
});

function custom_close(){
    $('#sampleModal').modal('hide');
    }

// Handle sample image clicks - actual work
async function handleSamples(imgItem) {
  if (imgListEmpty == 0) {
    displayError(1);
    return;
  }
  var tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = imgItem.naturalWidth;
  tmpCanvas.height = imgItem.naturalHeight;
  var tmpCtx = tmpCanvas.getContext("2d");
  // Below 2 lines required to access images from external domain
  // Else the canvas is "tainted" by the external content and cannot be Base64 converted 
  var imgTemp = new Image;
  imgTemp.crossOrigin = "anonymous";
  imgTemp.onload = async function(){
    tmpCtx.drawImage(imgTemp, 0, 0);
    b64temp = tmpCanvas.toDataURL();
    await photoSave(0,b64temp);
    b64temp = 0;
  };
  imgTemp.src = imgItem.src;
 
}

// Handle example image clicks - need this unsual syntax to accomodate the async nature of the process
document.querySelectorAll('.eImg').forEach(item => {
  item.addEventListener('click', () => exampleClick(item), false)
});

// Handle example image clicks - actual work
async function exampleClick(imgItem) {
  var tmpCanvas = document.createElement("canvas");
  var tmpCanvas = document.getElementById("resultsCanvas8");
  tmpCanvas.width = imgItem.naturalWidth;
  tmpCanvas.height = imgItem.naturalHeight;
  var tmpCtx = tmpCanvas.getContext("2d");
  // Below 2 lines required to access images from external domain
  // Else the canvas is "tainted" by the external content and cannot be Base64 converted 
  var imgTemp = new Image;
  imgTemp.crossOrigin = "anonymous";
  imgTemp.onload = async function(){
    tmpCtx.drawImage(imgTemp, 0, 0);
    b64e = tmpCanvas.toDataURL();

    // Image classification
    let exampleId = imgItem.getAttribute("data-eid");
    var exampleData = exampleIC[exampleId];
    var showExample = await jsonParser(exampleData, 7);
  
    // Object detection
    exampleData = exampleOD[exampleId];
    showExample = await jsonParser(exampleData, 8);

    // Image similarity
    exampleData = exampleIS[exampleId];
    showExample = await jsonParser(exampleData, 9);
  };
  imgTemp.src = imgItem.src;
}

function exampleModels() {
var icCheck = document.getElementById("icCheck").checked;
  var odCheck = document.getElementById("odCheck").checked;
  var isCheck = document.getElementById("isCheck").checked;

  var icDiv = document.getElementById("resultsDiv7");
  var odDiv = document.getElementById("resultsDiv8");
  var isDiv = document.getElementById("resultsDiv9");

  if (icCheck) icDiv.classList.remove("hide");
  else icDiv.classList.add("hide");
  if (odCheck) odDiv.classList.remove("hide");
  else odDiv.classList.add("hide");
  if (isCheck) isDiv.classList.remove("hide");
  else isDiv.classList.add("hide");
}


// Trigger photo take
document.getElementById("snap").addEventListener("click", function() {
  webCamCanvas.classList.remove("hide");
  var width = video.videoWidth;
  var height = video.videoHeight;
  wCCcontext.canvas.width = width;
  wCCcontext.canvas.height = height;
  wCCcontext.drawImage(video, 0, 0, width, height);
});


// Trigger photo save - need this unsual syntax to accomodate the async nature of the "photoSave" process
document.getElementById("useImage").addEventListener("click", () => photoSave(), false);


async function photoSave(saveType, b64i) {
  $("#imageaddedmsg").toggleClass("show");
  if (saveType == 0)
    var dataURL = b64i;
  else {
    // Basically this gets called only when an attempt is made to save the webcam image
    if (imgListEmpty == 0) {
      displayError(1);
      return;
    }
    var dataURL = webCamCanvas.toDataURL();
  }
  var thumbnailURL = await resizeImg(dataURL, 150);
  var fullimgURL = await resizeImg(dataURL, 480);
  for (let i = 0; i < 4; i++) {
      if (imgList[i] == 0) {
      console.log("imgList has empty slot at: " + i);   
      document.getElementById("b64img-" + i).src=thumbnailURL;
      document.getElementById("clear-" + i).classList.remove("hide");
      document.getElementById("b64imgwrap-" + i).classList.remove("img-wrap-ph");
      imgList[i] = 1;
      b64o[i] = fullimgURL;
      imgListEmpty--;
      i = 4;
      }
   
      if (("b64o"+i) == 0) {
        console.log("b64 object " + i + "is empty (set to 0)");
      }
  }
  
  if (video.srcObject) {
    $('#multiCollapseWebcam').collapse('hide');
  }
    
}

$('#multiCollapseWebcam').on('hide.bs.collapse', function () {
  webcamStop();
  webCamCanvas.classList.add("hide");
  document.getElementById("btnWebcam").classList.remove("active");
  document.getElementById("btnWebcam").innerText = "Webcam";
})

$('#multiCollapseWebcam').on('shown.bs.collapse', function () {
  webcamActivate1();
})

$('#multiCollapseSample').on('hidden.bs.collapse', function () {
  document.getElementById("btnSample").classList.remove("active");
  document.getElementById("btnSample").innerText = "Choose From Samples";
})

$('#multiCollapseSample').on('shown.bs.collapse', function () {
  document.getElementById("btnSample").classList.add("active");
  document.getElementById("btnSample").innerText = "Hide Samples";
})

function sampleClose() {
  $('#multiCollapseSample').collapse('hide');
}

function handleFiles(files) {
  num_file = files.length;
  var j = num_file;
  console.log("num_file: " + num_file);
  if (num_file > 4) 
    num_file = 4;
  
  if (imgListEmpty == 0) {
    displayError(1);
    return 0;
  }
  
  for (let i = 0; i < num_file; i++) {        
    const file = files[i];  
    const reader = new FileReader();
    if (!reader) {
      console.log("sorry, change the browser.");
      return
    }
    // Save the images
    reader.onload = ( function(aImg) { return function(e) { 
      aImg.src = e.target.result;
      console.log("image #" + i + " processed")
      j--;
      if(j == 0)  // After all files have been processed, call fnc to display them
        saveFiles(num_file);
      }; })(tempImg[i]);
    reader.readAsDataURL(file);
  }
}


async function saveFiles(numFiles) {
  for (let k = 0; k < numFiles; k++) {
    await photoSave(0, tempImg[k].src);
    tempImg[k].src = "";  // Clear photo from temp storage after saving it
  }
  console.log("imgListEmpty: " + imgListEmpty);
}

// Delete image from display and img list
function removeImg(imgNumber) {
  console.log("Remove Image Number: " + imgNumber);
  document.getElementById("b64img-" + imgNumber).src="";
  document.getElementById("clear-" + imgNumber).classList.add("hide");
  document.getElementById("b64imgwrap-" + imgNumber).classList.add("img-wrap-ph");
  imgList[imgNumber] = 0;
  b64o[imgNumber] = 0;
  imgListEmpty++;
  console.log("imgListEmpty in removeImg: " + imgListEmpty);
}

function resizeImg(b64Orig, newHeight) {
  return new Promise(async function(resolve,reject){
    // Create an off-screen canvas
    var rIcanvas = document.createElement('canvas'),
    rIctx = rIcanvas.getContext('2d');
    // Create an off-screen image element
    var rImage = new Image();
    // WHen the image is loaded, process it
    rImage.onload = function() {
    // Original dimensions of image
    var width = rImage.naturalWidth;
    var height = rImage.naturalHeight;
    var ratio = width / height;
    // Dimensions of resized image (via canvas): rIwidth and newHeight
    var rIwidth = ratio * newHeight;
    // Set canvas dimensions to resized image dimensions
    rIcanvas.width = rIwidth;
    rIcanvas.height = newHeight;
    // Draw the image on teh canvas at the new size
      rIctx.drawImage(rImage, 0, 0, width, height, 0, 0, rIcanvas.width, rIcanvas.height);
      // Export the new image as Base64 and return to calling function
      var rIdu = rIcanvas.toDataURL();
      resolve(rIdu);
    }
    // Load the image from the original Base64 source (passed into this function)
    rImage.src = b64Orig;
  })
}

function APIValidation(url) {
  var pattern = /(ftp|http|https):\/\/(\w+:{0,1}\w*@)?(\S+)(:[0-9]+)?(\/|\/([\w#!:.?+=&%@!\-\/]))?/;
  if (pattern.test(url)) {
    console.log("url is valid")
    return true;
  } else {
    displayError(3);
    return false;
  }
}


function webcamActivate1(){
  document.getElementById("btnWebcam").classList.add("active");
  document.getElementById("btnWebcam").innerText = "Hide Webcam";
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
  }
}

function webcamStop(){
  var stream = video.srcObject;
  var tracks = stream.getTracks();
  for (var i = 0; i < tracks.length; i++) {
    var track = tracks[i];
    track.stop();
  }
  video.srcObject = null;
}


function displayError(errno) {
  var errtext = "";
  switch (errno) {
    case 1:
      errtext = "Only 4 images can be uploaded at a time. To use different images, delete one of your existing thumbnails.";
      break;
    case 2:
      errtext = "Error during API request.";
      break;
    case 3:
      errtext = "Invalid API url.";
      break;
    default:
      errtext = "An error occured.";
      break;
  }
  var alertDiv = document.getElementById("alertdiv");
  alertDiv.innerHTML = '<div id="alert" class="alert alert-danger alert-dismissible fade hide" role="alert"><strong>Alert!</strong> <span id="progress">You should check in on some of those fields below.</span><button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button></div>'
  var progress = document.getElementById("progress");
  progress.innerHTML = errtext;
  var alert = document.getElementById("alert");
  alert.classList.remove("hide");
  alert.classList.add("show");
}

function renderImage(i) {
  return new Promise(async function(resolve,reject){
    var img = document.getElementById("resultsImg" + i);
    
    img.onload = function(){
      var width = img.naturalWidth;
      var height = img.naturalHeight;

      var c = document.getElementById("resultsCanvas" + i);
      var ctx = c.getContext("2d");

      ctx.canvas.height = height;
      ctx.canvas.width = width; 

      var scale = Math.min(c.width / width, c.height / height);
      // get the top left position of the image
      var imgx = (c.width / 2) - (width / 2) * scale;
      var imgy = (c.height / 2) - (height / 2) * scale;
      ctx.drawImage(img, imgx, imgy, width * scale, height * scale);
      img.src = "";
      img.classList.add("hide");
      resolve();
    };
    if (i < 7)
      img.src = b64o[i];
    else 
      img.src = b64e; 
  })
}

function imgdetection(i, x, y, xwidth, xheight, label) {
  var c = document.getElementById("resultsCanvas" + i);
  var ctx = c.getContext("2d");
  
  ctx.lineWidth = 5;
  ctx.strokeStyle = "#FF0000";
  ctx.fillStyle = "#FF0000";
  ctx.font = "20px Verdana";
  ctx.strokeRect(x, y, xwidth, xheight);
  ctx.fillText(label, 10 + parseInt(x), 20 + parseInt(y));
}

function imgclassification(i, label, probability) {
  var c = document.getElementById("resultsCanvas" + i);
  var ctx = c.getContext("2d");

  ctx.lineWidth = 5;
  ctx.strokeStyle = "#FF0000";
  ctx.fillStyle = "#FF0000";
  ctx.font = "20px Verdana";

  ctx.fillText(label, 10, 30);
  ctx.fillText(parseFloat(probability).toFixed(2), 10, 60);
}

// "count" indicates the number of similar results to return; use 5 for now
// Call with no "queryFeatures" to use a random image from the existing thumbnails
// So example call without queryFeatures:  imgsimilarity(0,5)
async function imgsimilarity(i, count, queryFeatures) {
  // Do work here
  simType = "mymodel";
  if (i > 6)
    simType = "example";
  if (fn_array.length == 0 && i < 7) {
    // The zip files for the similarity comparison have't been processed yet
    await parseSimFileNames(simType);
    await parseSimFileFeatures(simType);
  } else if (fn_array_ex.length == 0 && i > 6) {
    // The zip files for the similarity comparison have't been processed yet
    await parseSimFileNames(simType);
    await parseSimFileFeatures(simType);
  }
  var results = calcSimilar(count, queryFeatures, simType);
  // results: [topResults from image matching, path to query image if no queryFeatures]
  populateTable(i, results[0]);
}

async function jsonParser(jString, ovr) {
  let resp = JSON.parse(jString)
  if (Array.isArray(resp[0])) {
    if (resp[0][0].hasOwnProperty("top")) {
      // "[[top: #, ]]"
      // will need to target a different feature if another scenario ends up doing rectangle boxes
      for (let i in resp) {
        let j = i
        if (ovr) {
          j = ovr
        }
        await renderImage(j);
        for (let box of resp[i]) {
          let x = box.left
          let y = box.top
          let width = box.right - box.left
          let height = box.bottom - box.top
          let label = box.label_name
          imgdetection(j, x, y, width, height, label)
        }
      }
      return "detection"
    }
    return "err"
  }

  // '[{"label":"asdasd","probability":"0.21354"},{"label": "klsdfjkdsfjklsdf","probability":"0.4512457"}]'
  if(resp[0].hasOwnProperty("probability")) {
    for (let i in resp) {
      let j = i
      if (ovr) {
        j = ovr
      }
      await renderImage(j);
      let label = resp[i].label
      let prob = resp[i].probability
      imgclassification(j, label, prob);
    }
    return "classification"
  }

  if(resp[0].hasOwnProperty("features")) {
    for (let i in resp) {
      let j = i
      if (ovr) {
        j = ovr
      }
      await renderImage(j);
      let features = resp[i].features;

      // parse the json into an array
      let featuresArray = JSON.parse(features)

      imgsimilarity(j, 5, featuresArray);
    }
    return "similarity"
  }
  return "err"
}

//whatever calls this should have a timeout
function APIRequest() {
  let url = document.getElementById("url").value;
  if (!APIValidation(url))
    return 0;

  var uplBtn = document.getElementById("uploadbtn");
  var uplStatus = document.getElementById("uploadstatus");
  uplBtn.disabled = "true";
  uplStatus.classList.remove("hide");
  uplStatus.innerHTML = 'Loading... <div class="spinner-border ml-auto spinner-border-sm" role="status" aria-hidden="true"></div>';

  let xhr = new XMLHttpRequest();

  xhr.onload = function() {
    console.log("request completed")
    if (xhr.readyState == 4) {
      if (xhr.status == 200) {
        console.log(xhr.responseXML);
        //loading(2);
        jsonParser(xhr.responseText);
        //loading(3);
      } else {
        displayError(0);  // Display generic error message in bold, red text
        console.log("Error: " + xhr.status + " response. " + xhr.responseText);
      }
      uplBtn.disabled = false;
      uplStatus.innerHTML = '<span class="text-muted font-weight-light font-italic">Complete</span>';
    }
  }

  xhr.onerror = function() {
    console.log(xhr.status)
    displayError(2);  // Display generic API error message in bold, red text
    uplBtn.disabled = false;
    uplStatus.innerHTML = '<span class="text-muted font-weight-light font-italic">Complete</span>';
  }

  xhr.open("POST", url, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  //add b64 strings to payload list at key "data"
  console.log("sending request")
  let dataList = []
  for (let i in b64o) {
    if (b64o[i] != 0) {
      dataList.push(b64o[i].split(',')[1]);
    }
  }
  xhr.send(JSON.stringify({"data": dataList}));
}