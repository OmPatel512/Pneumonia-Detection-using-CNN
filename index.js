console.log("started");

async function loadModel() {
  const model = await tf.loadLayersModel('dl_model/model.json');
  return model;
}

var preds;
var isPneumonia;

async function predictImage(fileInput) {
  try {
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function (e) {
      const img = new Image();
      img.src = e.target.result;
      
      img.onload = async function () {
        const tensor = tf.browser.fromPixels(img).resizeBilinear([224, 224]).expandDims();
        const normalizedImg = tensor.toFloat().div(tf.scalar(255.0));

        console.log(tensor);
        const model = await loadModel();
        const predictions = model.predict(normalizedImg);

        // Extract values from the predictions tensor
        const predictionsValues = await predictions.array();

        preds = predictionsValues[0][0];

        var res = document.getElementById('res');
        if (preds < 0.5) {
            res.innerText = "No Pneumonia Detected";
            isPneumonia = false;
        } else {
            res.innerText = "Pneumonia Detected";
            isPneumonia = true;
        }
      };
    };

    reader.readAsDataURL(file);
  } catch (error) {
    console.error('Error processing image:', error);
  }
}

// Assuming 'img' is the ID of file input element
const fileInput = document.getElementById('img');
fileInput.addEventListener('change', () => predictImage(fileInput));



