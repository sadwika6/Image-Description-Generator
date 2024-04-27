function generateDescription(file) {
    var imagePreview = document.getElementById("imagePreview");
    var generatedText = document.getElementById("generatedText");

    if (file) {
        var reader = new FileReader();
        reader.onload = function (e) {
            var img = new Image();
            img.src = e.target.result;
            imagePreview.innerHTML = '';
            imagePreview.appendChild(img);
        };
        reader.readAsDataURL(file);
        generatedText.textContent = "Uploaded image: " + file.name;
    } else {
        imagePreview.innerHTML = "";
        generatedText.textContent = "No image uploaded.";
    }
}

function speakDescription() {
    var generatedText = document.getElementById("generatedText").textContent;
    var speechSynthesis = window.speechSynthesis;
    var speechMsg = new SpeechSynthesisUtterance(generatedText);
    speechSynthesis.speak(speechMsg);
}

document.getElementById("imageUpload").addEventListener("change", function(e) {
    generateDescription(e.target.files[0]);
});

document.getElementById("captureBtn").addEventListener("click", function() {
    var fileInput = document.getElementById("imageUpload");
    if (fileInput.files.length > 0) {
        generateDescription(fileInput.files[0]);
    }
});

document.getElementById("generateBtn").addEventListener("click", function() {
    var fileInput = document.getElementById("imageUpload");
    if (fileInput.files.length > 0) {
        generateDescription(fileInput.files[0]);
        // Send AJAX request to Flask endpoint to generate caption
        var formData = new FormData();
        formData.append('file', fileInput.files[0]);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) // Parse response as JSON
        .then(data => {
            // Update generated text with the received caption from Flask
            document.getElementById("generatedText").textContent = data.caption;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert("Please upload an image first.");
    }
});

document.getElementById("audioBtn").addEventListener("click", speakDescription);


















// function generateDescription(file) {
//     var imagePreview = document.getElementById("imagePreview");
//     var generatedText = document.getElementById("generatedText");

//     if (file) {
//         var reader = new FileReader();
//         reader.onload = function (e) {
//             var img = new Image();
//             img.src = e.target.result;
//             imagePreview.innerHTML = '';
//             imagePreview.appendChild(img);
//         };
//         reader.readAsDataURL(file);
//         generatedText.textContent = "Uploaded image: " + file.name;
//     } else {
//         imagePreview.innerHTML = "";
//         generatedText.textContent = "No image uploaded.";
//     }
// }

// function speakDescription() {
//     var generatedText = document.getElementById("generatedText").textContent;
//     var speechSynthesis = window.speechSynthesis;
//     var speechMsg = new SpeechSynthesisUtterance(generatedText);
//     speechSynthesis.speak(speechMsg);
// }

// document.getElementById("imageUpload").addEventListener("change", function(e) {
//     generateDescription(e.target.files[0]);
// });

// document.getElementById("captureBtn").addEventListener("click", function() {
//     var fileInput = document.getElementById("imageUpload");
//     if (fileInput.files.length > 0) {
//         generateDescription(fileInput.files[0]);
//     }
// });

// document.getElementById("generateBtn").addEventListener("click", function() {
//     var fileInput = document.getElementById("imageUpload");
//     if (fileInput.files.length > 0) {
//         generateDescription(fileInput.files[0]);
//         // Send AJAX request to Flask endpoint to generate caption
//         var formData = new FormData();
//         formData.append('file', fileInput.files[0]);
//         fetch('/predict', {
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.text())
//         .then(caption => {
//             document.getElementById("generatedText").textContent = caption;
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     } else {
//         alert("Please upload an image first.");
//     }
// });

// document.getElementById("audioBtn").addEventListener("click", speakDescription);


















// function generateDescription(file) {
//     var imagePreview = document.getElementById("imagePreview");
//     var generatedText = document.getElementById("generatedText");

//     if (file) {
//         var reader = new FileReader();
//         reader.onload = function (e) {
//             var img = new Image();
//             img.src = e.target.result;
//             imagePreview.innerHTML = '';
//             imagePreview.appendChild(img);
//         };
//         reader.readAsDataURL(file);
//         generatedText.textContent = "Uploaded image: " + file.name;
//     } else {
//         imagePreview.innerHTML = "";
//         generatedText.textContent = "No image uploaded.";
//     }
// }

// function speakDescription() {
//     var generatedText = document.getElementById("generatedText").textContent;
//     var speechSynthesis = window.speechSynthesis;
//     var speechMsg = new SpeechSynthesisUtterance(generatedText);
//     speechSynthesis.speak(speechMsg);
// }

// document.getElementById("imageUpload").addEventListener("change", function(e) {
//     generateDescription(e.target.files[0]);
// });

// document.getElementById("captureBtn").addEventListener("click", function() {
//     var fileInput = document.getElementById("imageUpload");
//     if (fileInput.files.length > 0) {
//         generateDescription(fileInput.files[0]);
//     }
// });

// document.getElementById("generateBtn").addEventListener("click", function() {
//     var fileInput = document.getElementById("imageUpload");
//     if (fileInput.files.length > 0) {
//         generateDescription(fileInput.files[0]);
//         // Send AJAX request to Flask endpoint to generate caption
//         var formData = new FormData();
//         formData.append('file', fileInput.files[0]);
//         fetch('/predict', {
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.text())
//         .then(caption => {
//             document.getElementById("generatedText").textContent = caption;
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     } else {
//         alert("Please upload an image first.");
//     }
// });

// document.getElementById("audioBtn").addEventListener("click", speakDescription);

// function generateDescription(file) {
//     var imagePreview = document.getElementById("imagePreview");
//     var generatedText = document.getElementById("generatedText");

//     if (file) {
//         var reader = new FileReader();
//         reader.onload = function (e) {
//             var img = new Image();
//             img.src = e.target.result;
//             imagePreview.innerHTML = '';
//             imagePreview.appendChild(img);
//         };
//         reader.readAsDataURL(file);
//         generatedText.textContent = "Uploaded image: " + file.name;
//     } else {
//         imagePreview.innerHTML = "";
//         generatedText.textContent = "No image uploaded.";
//     }
// }

// function speakDescription() {
//     var generatedText = document.getElementById("generatedText").textContent;
//     var speechSynthesis = window.speechSynthesis;
//     var speechMsg = new SpeechSynthesisUtterance(generatedText);
//     speechSynthesis.speak(speechMsg);
// }

// document.getElementById("imageUpload").addEventListener("change", function(e) {
//     generateDescription(e.target.files[0]);
// });

// document.getElementById("captureBtn").addEventListener("click", function() {
//     var fileInput = document.getElementById("imageUpload");
//     if (fileInput.files.length > 0) {
//         generateDescription(fileInput.files[0]);
//     }
// });

// document.getElementById("generateBtn").addEventListener("click", function() {
//     var fileInput = document.getElementById("imageUpload");
//     if (fileInput.files.length > 0) {
//         generateDescription(fileInput.files[0]);
//         // Send AJAX request to Flask endpoint to generate caption
//         var formData = new FormData();
//         formData.append('file', fileInput.files[0]);
//         fetch('/predict', {
//             method: 'POST',
//             body: formData
//         })
//         .then(response => response.text())
//         .then(caption => {
//             document.getElementById("generatedText").textContent = caption;
//         })
//         .catch(error => {
//             console.error('Error:', error);
//         });
//     } else {
//         alert("Please upload an image first.");
//     }
// });

// document.getElementById("audioBtn").addEventListener("click", speakDescription);
